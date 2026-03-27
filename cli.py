#!/usr/bin/env python3
"""
adaptive-quant: Adaptive mixed-precision quantization for GGUF models.

Profiles per-layer quantization sensitivity, solves a knapsack optimization
for a target memory budget, and generates mixed-precision GGUF files.

Usage:
    adaptive-quant profile <model_path> [--output <profile.json>] [--bits 8,4,2]
    adaptive-quant generate <profile.json> --budget <GB> [--output <model.gguf>]
    adaptive-quant compare <original.gguf> <adaptive.gguf> [--perplexity] [--bench]
    adaptive-quant demo  # Run with simulated Qwen3.5-9B data
"""

import argparse
import json
import sys
import os

from .profiler import SensitivityProfiler, SimulatedProfiler
from .knapsack import KnapsackSolver
from .generator import GGUFGenerator
from .visualizer import print_sensitivity_heatmap, print_allocation_table, print_comparison
from .devices import DEVICE_PRESETS


def cmd_profile(args):
    """Profile per-layer quantization sensitivity."""
    print(f"\n🔬 Profiling quantization sensitivity: {args.model}")
    print(f"   This quantizes each layer individually and measures perplexity delta.\n")

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

    print(f"\n✅ Sensitivity profile saved to: {output}")
    print_sensitivity_heatmap(profile)

    return profile


def cmd_generate(args):
    """Generate an adaptive mixed-precision GGUF from a sensitivity profile."""
    with open(args.profile) as f:
        profile = json.load(f)

    budget_gb = args.budget
    if args.device:
        if args.device not in DEVICE_PRESETS:
            print(f"❌ Unknown device '{args.device}'. Available: {', '.join(DEVICE_PRESETS.keys())}")
            sys.exit(1)
        budget_gb = DEVICE_PRESETS[args.device]["memory_gb"]
        overhead_gb = DEVICE_PRESETS[args.device]["os_overhead_gb"]
        budget_gb = budget_gb - overhead_gb
        print(f"\n🎯 Target device: {args.device}")
        print(f"   Total memory: {DEVICE_PRESETS[args.device]['memory_gb']}GB, "
              f"OS overhead: {overhead_gb}GB, available for model: {budget_gb:.1f}GB")
    else:
        print(f"\n🎯 Target memory budget: {budget_gb}GB")

    solver = KnapsackSolver(profile)
    allocation = solver.solve(budget_gb=budget_gb)

    print_allocation_table(allocation, profile)

    if not args.dry_run:
        generator = GGUFGenerator(
            llama_quantize=args.llama_quantize,
        )
        output = args.output or f"adaptive_{budget_gb:.0f}gb.gguf"
        generator.generate(
            source_model=profile["model_path"],
            allocation=allocation,
            output_path=output,
        )
        print(f"\n✅ Adaptive GGUF saved to: {output}")
    else:
        print(f"\n📋 Dry run — no GGUF generated. Use without --dry-run to build the model.")

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
        ("M5 Air 24GB", 24 - 6),
        ("M4 Air 16GB", 16 - 5),
        ("iPhone 17 Pro 8GB", 8 - 4),
    ]

    allocations = {}
    for device_name, budget in devices:
        print(f"\n{'─' * 70}")
        print(f"  STEP 2: Knapsack Optimization → {device_name} ({budget:.0f}GB available)")
        print(f"{'─' * 70}")

        alloc = solver.solve(budget_gb=budget)
        allocations[device_name] = alloc
        print_allocation_table(alloc, profile)

    # Step 3: Compare against uniform quantization
    print(f"\n{'─' * 70}")
    print(f"  STEP 3: Adaptive vs Uniform Quantization Comparison")
    print(f"{'─' * 70}")
    print_comparison(allocations, profile)

    # Save the full demo output
    demo_output = {
        "profile": profile,
        "allocations": {name: alloc for name, alloc in allocations.items()},
    }
    with open("adaptive_quant_demo.json", "w") as f:
        json.dump(demo_output, f, indent=2)

    print(f"\n✅ Full demo data saved to: adaptive_quant_demo.json")
    print(f"\n💡 To run this on a real model:")
    print(f"   1. Install llama.cpp: brew install llama.cpp")
    print(f"   2. Profile:  adaptive-quant profile path/to/Qwen3.5-9B-BF16.gguf")
    print(f"   3. Generate: adaptive-quant generate sensitivity.json --device m5-pro-64gb")


def main():
    parser = argparse.ArgumentParser(
        prog="adaptive-quant",
        description="Adaptive mixed-precision quantization for GGUF models",
    )
    sub = parser.add_subparsers(dest="command")

    # profile
    p_profile = sub.add_parser("profile", help="Profile per-layer quantization sensitivity")
    p_profile.add_argument("model", help="Path to BF16/F16 GGUF model file")
    p_profile.add_argument("--output", "-o", help="Output JSON path")
    p_profile.add_argument("--simulate", action="store_true", help="Use simulated data (for demo)")
    p_profile.add_argument("--llama-perplexity", default="llama-perplexity", help="Path to llama-perplexity binary")
    p_profile.add_argument("--llama-quantize", default="llama-quantize", help="Path to llama-quantize binary")
    p_profile.add_argument("--reference-text", default=None, help="Reference text for perplexity (default: wikitext-2)")

    # generate
    p_gen = sub.add_parser("generate", help="Generate adaptive mixed-precision GGUF")
    p_gen.add_argument("profile", help="Path to sensitivity profile JSON")
    p_gen.add_argument("--budget", type=float, help="Memory budget in GB")
    p_gen.add_argument("--device", help="Target device preset (e.g., m5-pro-64gb)")
    p_gen.add_argument("--output", "-o", help="Output GGUF path")
    p_gen.add_argument("--dry-run", action="store_true", help="Show allocation without generating")
    p_gen.add_argument("--llama-quantize", default="llama-quantize", help="Path to llama-quantize binary")

    # demo
    sub.add_parser("demo", help="Run full pipeline with simulated Qwen3.5-9B data")

    args = parser.parse_args()

    if args.command == "profile":
        cmd_profile(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
