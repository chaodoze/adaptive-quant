#!/usr/bin/env python3
"""
adaptive-quant: Adaptive mixed-precision quantization for GGUF models.

Profiles per-layer quantization sensitivity, solves a knapsack optimization
for a target memory budget, and generates mixed-precision GGUF files.

Usage:
    adaptive-quant catalog list
    adaptive-quant catalog show <model>
    adaptive-quant optimize <model> --budget <GB> [--dry-run] [--script]
    adaptive-quant optimize <model> --device <device> [--dry-run] [--script]
    adaptive-quant benchmark <model> --budget <GB>
    adaptive-quant profile <model_path> [--output <profile.json>] [--simulate]
    adaptive-quant generate <profile.json> --budget <GB> [--output <model.gguf>]
    adaptive-quant demo
"""

import argparse
import json
import sys
import os

from .profiler import SensitivityProfiler, SimulatedProfiler
from .knapsack import KnapsackSolver
from .generator import GGUFGenerator
from .visualizer import (
    print_sensitivity_heatmap, print_allocation_table, print_comparison,
    print_catalog_list, print_benchmark, print_kv_analysis,
)
from .kvcache import kv_analysis
from .devices import DEVICE_PRESETS
from . import catalog


def _resolve_budget(args):
    """Resolve budget from --budget or --device flag."""
    if args.device:
        if args.device not in DEVICE_PRESETS:
            print(f"Unknown device '{args.device}'. Available: {', '.join(DEVICE_PRESETS.keys())}")
            sys.exit(1)
        preset = DEVICE_PRESETS[args.device]
        budget_gb = preset["memory_gb"] - preset["os_overhead_gb"]
        print(f"\n   Target device: {preset['name']}")
        print(f"   Total memory: {preset['memory_gb']}GB, "
              f"OS overhead: {preset['os_overhead_gb']}GB, "
              f"available for model: {budget_gb:.1f}GB")
        return budget_gb
    elif args.budget:
        print(f"\n   Target memory budget: {args.budget}GB")
        return args.budget
    else:
        print("Error: specify --budget <GB> or --device <preset>")
        sys.exit(1)


# === New commands: catalog, optimize, benchmark ===

def cmd_catalog_list(args):
    """List all models in the catalog."""
    models = catalog.list_models()
    print_catalog_list(models)


def cmd_catalog_show(args):
    """Show detailed profile for a catalog model."""
    try:
        profile = catalog.get_profile(args.model)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n   Model: {profile['model_name']}")
    print(f"   Architecture: {profile.get('architecture', 'dense')}")
    print(f"   BF16 size: {profile.get('bf16_size_gb', '?')}GB")
    print(f"   Sources: {', '.join(profile.get('sources', []))}")
    print(f"   Confidence: {profile.get('confidence', 'N/A')}")

    special = profile.get("special_tensors", {})
    if special:
        # Summarize special tensors
        shexp_count = sum(1 for k in special if "shexp" in k)
        other = {k: v for k, v in special.items() if "shexp" not in k}
        if shexp_count:
            print(f"   Special tensors: {shexp_count} shared-expert tensors pinned to BF16")
        for k, v in other.items():
            print(f"   Special tensors: {k} → {v}")

    print_sensitivity_heatmap(profile)


def cmd_optimize(args):
    """Optimize a catalog model for a target budget/device."""
    try:
        profile = catalog.get_profile(args.model)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    total_budget_gb = _resolve_budget(args)
    architecture = profile.get("architecture", "dense")
    cache_type = getattr(args, "cache_type", "q4_0") or "q4_0"
    context_length = getattr(args, "context", None)

    # Reserve memory for KV cache if context target specified
    weight_budget_gb = total_budget_gb
    kv_reserved_gb = 0
    if context_length and profile.get("kv_architecture"):
        from .kvcache import kv_cache_size_gb
        kv_reserved_gb = kv_cache_size_gb(context_length, cache_type, profile)
        weight_budget_gb = total_budget_gb - kv_reserved_gb
        print(f"\n   KV cache reservation: {kv_reserved_gb:.2f}GB "
              f"for {context_length // 1024}K context @ {cache_type}")
        print(f"   Weight budget: {weight_budget_gb:.2f}GB "
              f"(total {total_budget_gb:.1f}GB - {kv_reserved_gb:.2f}GB KV)")

    solver = KnapsackSolver(profile)
    allocation = solver.solve(budget_gb=weight_budget_gb)

    print_allocation_table(allocation, profile)

    if args.script:
        generator = GGUFGenerator()
        script = generator.generate_script(
            source_model=f"<path-to-{profile['model_name']}-BF16.gguf>",
            allocation=allocation,
            output_path=f"adaptive_{profile['model_name']}_{total_budget_gb:.0f}gb.gguf",
            architecture=architecture,
            cache_type=cache_type,
        )
        print("\n   Shell script:\n")
        print(script)
    elif not args.dry_run:
        if not args.source:
            print("\n   To generate the GGUF, provide --source <path-to-BF16.gguf>")
            print("   Or use --script to emit a standalone shell script.")
        else:
            generator = GGUFGenerator(
                llama_quantize=args.llama_quantize,
            )
            output = args.output or f"adaptive_{profile['model_name']}_{total_budget_gb:.0f}gb.gguf"
            generator.generate(
                source_model=args.source,
                allocation=allocation,
                output_path=output,
                architecture=architecture,
            )
            print(f"\n   Adaptive GGUF saved to: {output}")
    else:
        print("\n   Dry run — no GGUF generated.")

    # TurboQuant KV cache analysis
    if profile.get("kv_architecture"):
        kv_results = kv_analysis(profile, allocation["total_size_gb"], total_budget_gb)
        print_kv_analysis(kv_results, profile["model_name"],
                          allocation["total_size_gb"], total_budget_gb)

        print("\n   Recommended llama.cpp flags for TurboQuant:")
        print(f"   --cache-type-k {cache_type} --cache-type-v {cache_type}")

    return allocation


def cmd_benchmark(args):
    """Benchmark adaptive vs uniform quantization for a catalog model."""
    try:
        profile = catalog.get_profile(args.model)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    total_budget_gb = args.budget
    cache_type = getattr(args, "cache_type", "q4_0") or "q4_0"
    context_length = getattr(args, "context", None)

    # Reserve memory for KV cache if context target specified
    weight_budget_gb = total_budget_gb
    if context_length and profile.get("kv_architecture"):
        from .kvcache import kv_cache_size_gb
        kv_reserved_gb = kv_cache_size_gb(context_length, cache_type, profile)
        weight_budget_gb = total_budget_gb - kv_reserved_gb
        print(f"\n   KV cache reservation: {kv_reserved_gb:.2f}GB "
              f"for {context_length // 1024}K context @ {cache_type}")
        print(f"   Weight budget: {weight_budget_gb:.2f}GB")

    print(f"\n   Benchmarking {profile['model_name']} @ {weight_budget_gb:.1f}GB weight budget"
          f" ({total_budget_gb:.1f}GB total)")

    solver = KnapsackSolver(profile)

    # Adaptive allocation
    adaptive = solver.solve(budget_gb=weight_budget_gb)

    # Best uniform quant that fits
    uniform = solver.best_uniform_for_budget(weight_budget_gb)

    # Print benchmark comparison
    print_benchmark(adaptive, uniform, profile, weight_budget_gb)

    # Also show the allocation details
    print_allocation_table(adaptive, profile)

    # TurboQuant KV cache analysis
    if profile.get("kv_architecture"):
        kv_results = kv_analysis(profile, adaptive["total_size_gb"], total_budget_gb)
        print_kv_analysis(kv_results, profile["model_name"],
                          adaptive["total_size_gb"], total_budget_gb)

    return adaptive, uniform


# === Original commands ===

def cmd_profile(args):
    """Profile per-layer quantization sensitivity."""
    print(f"\n   Profiling quantization sensitivity: {args.model}")
    print("   This quantizes each layer individually and measures perplexity delta.\n")

    if args.simulate:
        profiler = SimulatedProfiler(args.model)
    else:
        profiler = SensitivityProfiler(
            model_path=args.model,
            llama_perplexity=args.llama_perplexity,
            llama_quantize=args.llama_quantize,
            reference_text=args.reference_text,
        )

    profile = profiler.run()

    output = args.output or f"{os.path.splitext(os.path.basename(args.model))[0]}_sensitivity.json"
    with open(output, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n   Sensitivity profile saved to: {output}")
    print_sensitivity_heatmap(profile)

    return profile


def cmd_generate(args):
    """Generate an adaptive mixed-precision GGUF from a sensitivity profile."""
    with open(args.profile) as f:
        profile = json.load(f)

    budget_gb = _resolve_budget(args)

    solver = KnapsackSolver(profile)
    allocation = solver.solve(budget_gb=budget_gb)

    print_allocation_table(allocation, profile)

    if not args.dry_run:
        generator = GGUFGenerator(
            llama_quantize=args.llama_quantize,
        )
        output = args.output or f"adaptive_{budget_gb:.0f}gb.gguf"
        architecture = profile.get("architecture", "dense")
        generator.generate(
            source_model=profile["model_path"],
            allocation=allocation,
            output_path=output,
            architecture=architecture,
        )
        print(f"\n   Adaptive GGUF saved to: {output}")
    else:
        print("\n   Dry run — no GGUF generated. Use without --dry-run to build the model.")

    return allocation


def cmd_demo(args):
    """Run the full pipeline with simulated Qwen3.5-9B data."""
    print("=" * 70)
    print("  adaptive-quant demo: Qwen3.5-9B Adaptive Mixed-Precision")
    print("=" * 70)

    # Step 1: Profile
    print("\n" + "─" * 70)
    print("  STEP 1: Layer Sensitivity Profiling")
    print("─" * 70)

    profiler = SimulatedProfiler("Qwen3.5-9B")
    profile = profiler.run()
    print_sensitivity_heatmap(profile)

    # Step 2: Generate allocations for multiple devices
    solver = KnapsackSolver(profile)

    devices = [
        ("M5 Max 128GB", 128 - 12),
        ("M5 Pro 64GB", 64 - 8),
        ("M4 Air 32GB", 32 - 6),
        ("M4 Air 16GB", 16 - 5),
        ("iPhone 17 Pro 8GB", 8 - 4),
    ]

    allocations = {}
    for device_name, budget in devices:
        print(f"\n{'─' * 70}")
        print(f"  STEP 2: Knapsack Optimization -> {device_name} ({budget:.0f}GB available)")
        print(f"{'─' * 70}")

        alloc = solver.solve(budget_gb=budget)
        allocations[device_name] = alloc
        print_allocation_table(alloc, profile)

    # Step 3: Compare against uniform quantization
    print(f"\n{'─' * 70}")
    print("  STEP 3: Adaptive vs Uniform Quantization Comparison")
    print(f"{'─' * 70}")
    print_comparison(allocations, profile)

    # Save the full demo output
    demo_output = {
        "profile": profile,
        "allocations": {name: alloc for name, alloc in allocations.items()},
    }
    with open("adaptive_quant_demo.json", "w") as f:
        json.dump(demo_output, f, indent=2)

    print("\n   Full demo data saved to: adaptive_quant_demo.json")
    print("\n   To use community-sourced profiles instead:")
    print("   adaptive-quant catalog list")
    print("   adaptive-quant benchmark Qwen3.5-35B-A3B --budget 26")


def main():
    parser = argparse.ArgumentParser(
        prog="adaptive-quant",
        description="Adaptive mixed-precision quantization for GGUF models",
    )
    sub = parser.add_subparsers(dest="command")

    # catalog
    p_catalog = sub.add_parser("catalog", help="Browse community-sourced model profiles")
    catalog_sub = p_catalog.add_subparsers(dest="catalog_action")
    catalog_sub.add_parser("list", help="List all available models")
    p_catalog_show = catalog_sub.add_parser("show", help="Show profile details for a model")
    p_catalog_show.add_argument("model", help="Model name (e.g., Qwen3.5-35B-A3B)")

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimize a catalog model for a target budget")
    p_opt.add_argument("model", help="Model name from catalog")
    p_opt.add_argument("--budget", type=float, help="Memory budget in GB")
    p_opt.add_argument("--device", help="Target device preset (e.g., m4-air-32gb)")
    p_opt.add_argument("--output", "-o", help="Output GGUF path")
    p_opt.add_argument("--source", help="Path to BF16 source GGUF for generation")
    p_opt.add_argument("--dry-run", action="store_true", help="Show allocation without generating")
    p_opt.add_argument("--script", action="store_true", help="Output a shell script instead of generating")
    p_opt.add_argument("--cache-type", default="q4_0",
                       choices=["f16", "q8_0", "q4_1", "q4_0"],
                       help="KV cache quantization type (default: q4_0)")
    p_opt.add_argument("--context", type=int, default=None,
                       help="Target context length in tokens (reserves KV cache memory from budget)")
    p_opt.add_argument("--llama-quantize", default="llama-quantize", help="Path to llama-quantize binary")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Compare adaptive vs uniform quantization")
    p_bench.add_argument("model", help="Model name from catalog")
    p_bench.add_argument("--budget", type=float, required=True, help="Total memory budget in GB")
    p_bench.add_argument("--cache-type", default="q4_0",
                         choices=["f16", "q8_0", "q4_1", "q4_0"],
                         help="KV cache quantization type (default: q4_0)")
    p_bench.add_argument("--context", type=int, default=None,
                         help="Target context length in tokens (reserves KV cache memory)")

    # profile (original)
    p_profile = sub.add_parser("profile", help="Profile per-layer quantization sensitivity")
    p_profile.add_argument("model", help="Path to BF16/F16 GGUF model file")
    p_profile.add_argument("--output", "-o", help="Output JSON path")
    p_profile.add_argument("--simulate", action="store_true", help="Use simulated data (for demo)")
    p_profile.add_argument("--llama-perplexity", default="llama-perplexity", help="Path to llama-perplexity binary")
    p_profile.add_argument("--llama-quantize", default="llama-quantize", help="Path to llama-quantize binary")
    p_profile.add_argument("--reference-text", default=None, help="Reference text for perplexity (default: wikitext-2)")

    # generate (original)
    p_gen = sub.add_parser("generate", help="Generate adaptive mixed-precision GGUF")
    p_gen.add_argument("profile", help="Path to sensitivity profile JSON")
    p_gen.add_argument("--budget", type=float, help="Memory budget in GB")
    p_gen.add_argument("--device", help="Target device preset (e.g., m5-pro-64gb)")
    p_gen.add_argument("--output", "-o", help="Output GGUF path")
    p_gen.add_argument("--dry-run", action="store_true", help="Show allocation without generating")
    p_gen.add_argument("--llama-quantize", default="llama-quantize", help="Path to llama-quantize binary")

    # demo (original)
    sub.add_parser("demo", help="Run full pipeline with simulated Qwen3.5-9B data")

    args = parser.parse_args()

    if args.command == "catalog":
        if args.catalog_action == "list":
            cmd_catalog_list(args)
        elif args.catalog_action == "show":
            cmd_catalog_show(args)
        else:
            p_catalog.print_help()
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "profile":
        cmd_profile(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
