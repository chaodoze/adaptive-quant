"""
Terminal visualizations for adaptive-quant.

Renders sensitivity heatmaps, allocation tables, and comparison reports
using Unicode box drawing and ANSI colors.
"""

from .knapsack import QUANT_OPTIONS, QUANT_ORDER


# ANSI color helpers
def _color(text, code):
    return f"\033[{code}m{text}\033[0m"

def _red(t): return _color(t, "91")
def _yellow(t): return _color(t, "93")
def _green(t): return _color(t, "92")
def _cyan(t): return _color(t, "96")
def _bold(t): return _color(t, "1")
def _dim(t): return _color(t, "2")


def _sensitivity_bar(score, max_score=2.0, width=30):
    """Render a horizontal bar for sensitivity score."""
    normalized = min(score / max_score, 1.0)
    filled = int(normalized * width)

    if normalized > 0.7:
        color = "91"  # Red
    elif normalized > 0.4:
        color = "93"  # Yellow
    else:
        color = "92"  # Green

    bar = "█" * filled + "░" * (width - filled)
    return f"\033[{color}m{bar}\033[0m"


def print_sensitivity_heatmap(profile):
    """
    Print a visual heatmap of per-layer quantization sensitivity.

    Shows the U-shaped curve with sensitive layers highlighted.
    """
    layers = profile["layers"]
    num_layers = profile["num_layers"]

    max_score = max(ly["sensitivity_score"] for ly in layers)

    print(f"\n   {'─' * 60}")
    print(f"   Layer Sensitivity Heatmap: {profile['model_name']}")
    print(f"   {'─' * 60}")
    print(f"   {'Layer':>8}  {'Score':>6}  {'Sensitivity':^30}  {'Zone'}")
    print(f"   {'─' * 60}")

    for layer in layers:
        idx = layer["index"]
        score = layer["sensitivity_score"]
        bar = _sensitivity_bar(score, max_score)

        # Classify zone
        if idx <= 2 or idx >= num_layers - 3:
            zone = _red("◆ EDGE (preserve)")
        elif score > max_score * 0.5:
            zone = _yellow("◇ elevated")
        else:
            zone = _green("○ compressible")

        # Only show every layer for small models, or sample for large ones
        if num_layers <= 40 or idx % max(1, num_layers // 40) == 0 or idx <= 2 or idx >= num_layers - 3:
            print(f"   {idx:>5}    {score:>5.3f}  {bar}  {zone}")

    print(f"   {'─' * 60}")
    most_sensitive = max(layers, key=lambda ly: ly["sensitivity_score"])
    least_sensitive = min(layers, key=lambda ly: ly["sensitivity_score"])
    min_score = least_sensitive["sensitivity_score"]
    print(f"   Max sensitivity: {_red(f'{max_score:.3f}')} (layer {most_sensitive['index']})")
    print(f"   Min sensitivity: {_green(f'{min_score:.3f}')} (layer {least_sensitive['index']})")

    # Show the U-curve pattern
    n = num_layers
    thirds = [
        ("First 3 layers (input)", layers[:3]),
        ("Middle layers", layers[n//4:3*n//4]),
        ("Last 3 layers (output)", layers[-3:]),
    ]

    print("\n   U-Curve Summary:")
    for label, group in thirds:
        avg = sum(ly["sensitivity_score"] for ly in group) / len(group)
        bar = _sensitivity_bar(avg, max_score, width=15)
        print(f"     {label:<26} avg={avg:.3f} {bar}")


def print_allocation_table(allocation, profile):
    """Print the per-layer quantization allocation."""
    layers = allocation["layers"]
    num_layers = len(layers)

    print("\n   Allocation Summary:")
    print(f"   ├── Strategy: {allocation['strategy']}")
    print(f"   ├── Budget: {allocation['budget_gb']:.1f}GB")
    print(f"   ├── Model size: {_bold(f'{allocation['total_size_gb']:.2f}GB')}")
    print(f"   ├── Headroom: {allocation['headroom_gb']:.2f}GB (for KV cache + OS)")
    print(f"   ├── Avg bits/weight: {_bold(f'{allocation['avg_bits_per_weight']:.1f}')}")
    print(f"   ├── Compression: {allocation['compression_ratio']}x from BF16")
    print(f"   └── Quality cost: {allocation['total_quality_cost']:.4f}")

    # Distribution summary
    dist = allocation["level_distribution"]
    print("\n   Quantization Distribution:")
    for quant in QUANT_ORDER:
        if quant in dist:
            count = dist[quant]
            pct = count / num_layers * 100
            bar = "█" * int(pct / 2)
            color_icon = QUANT_OPTIONS[quant]["color"]
            bits = QUANT_OPTIONS[quant]["bits"]
            print(f"     {color_icon} {quant:<8} ({bits:>4.1f}b): {count:>3} layers ({pct:>5.1f}%) {bar}")

    # Comparison vs uniform Q4_K
    comp = allocation.get("comparison_vs_uniform_q4k", {})
    if comp:
        improvement = comp.get("adaptive_quality_improvement_pct", 0)
        size_diff = comp.get("adaptive_size_difference_gb", 0)
        print("\n   vs Uniform Q4_K:")
        if improvement > 0:
            print(f"     {_green(f'✓ {improvement:.1f}% better quality')} at {size_diff:+.2f}GB size difference")
        else:
            print(f"     {_yellow(f'{improvement:.1f}% quality difference')} at {size_diff:+.2f}GB size")

    # Compact layer view
    print("\n   Per-layer view (compact):")
    row = "   "
    for layer in layers:
        q = layer["quant"]
        bits = QUANT_OPTIONS[q]["bits"]

        if bits >= 8:
            char = _green("▓")
        elif bits >= 5:
            char = _cyan("▒")
        elif bits >= 4:
            char = _yellow("░")
        elif bits >= 3:
            char = _red("·")
        else:
            char = _red(" ")

        row += char

    print(row)
    print(f"   {'↑':>{3}}{'Input':>0}{'↑':>{num_layers-8}}{'Output':>0}")
    print(f"   {_green('▓')}=8b+ {_cyan('▒')}=5-6b {_yellow('░')}=4b {_red('·')}=2-3b")


def print_comparison(allocations, profile):
    """Print comparison of adaptive allocations against uniform quantization."""

    bf16_size = profile.get("bf16_size_gb", 18.0)

    print(f"\n   {'Device':<25} {'Size':>7} {'Avg Bits':>9} {'Compress':>9} {'Quality':>12} {'vs Q4_K':>10}")
    print(f"   {'─' * 75}")

    # Uniform baselines
    uniform_configs = {
        "Uniform BF16": (16.0, bf16_size),
        "Uniform Q8_0": (8.0, bf16_size * 8/16),
        "Uniform Q4_K_M": (4.5, bf16_size * 4.5/16),
        "Uniform Q2_K": (2.6, bf16_size * 2.6/16),
    }

    for name, (bits, size) in uniform_configs.items():
        compress = bf16_size / size if size > 0 else 0
        print(f"   {_dim(f'{name:<25}')} {size:>6.1f}G {bits:>8.1f}b {compress:>8.1f}x {'(baseline)':>12} {'—':>10}")

    print(f"   {'─' * 75}")

    for device_name, alloc in allocations.items():
        size = alloc["total_size_gb"]
        bits = alloc["avg_bits_per_weight"]
        compress = alloc["compression_ratio"]
        cost = alloc["total_quality_cost"]
        vs_q4k = alloc.get("comparison_vs_uniform_q4k", {}).get("adaptive_quality_improvement_pct", 0)

        vs_str = f"{_green(f'+{vs_q4k:.1f}%')}" if vs_q4k > 0 else f"{_red(f'{vs_q4k:.1f}%')}"

        print(f"   {device_name:<25} {size:>6.1f}G {bits:>8.1f}b {compress:>8.1f}x {cost:>12.4f} {vs_str:>10}")

    print("\n   Quality cost: lower = better (sensitivity-weighted quantization error)")
    print("   vs Q4_K: positive = adaptive preserves more quality at same avg bitrate")


def print_catalog_list(models):
    """Print a table of all cataloged models."""
    print(f"\n   {'─' * 80}")
    print("   Available Models")
    print(f"   {'─' * 80}")
    print(f"   {'Model':<28} {'Arch':<12} {'Params':>8} {'BF16 Size':>10} {'Layers':>7} {'Sources'}")
    print(f"   {'─' * 80}")

    for m in models:
        params = f"{m['params_b']:.0f}B"
        if m.get("active_params_b"):
            params += f" ({m['active_params_b']:.0f}B active)"
        size = f"{m['bf16_size_gb']:.0f}GB"
        sources = ", ".join(m.get("sources", []))
        print(f"   {m['name']:<28} {m.get('architecture', 'dense'):<12} {params:>8} {size:>10} {m['num_layers']:>7} {sources}")

    print(f"   {'─' * 80}")


def print_benchmark(adaptive_alloc, uniform_result, profile, budget_gb):
    """Print side-by-side benchmark: adaptive vs best uniform quant."""
    model_name = profile["model_name"]
    bf16_size = profile.get("bf16_size_gb", 0)

    print(f"\n   {'═' * 72}")
    print(f"   BENCHMARK: {model_name} @ {budget_gb}GB budget")
    print(f"   BF16 reference size: {bf16_size:.1f}GB")
    print(f"   {'═' * 72}")

    # Side by side
    print(f"\n   {'Metric':<30} {'Adaptive':>18} {'Uniform ' + uniform_result['quant']:>18}")
    print(f"   {'─' * 68}")

    a = adaptive_alloc
    u = uniform_result

    print(f"   {'Total size':<30} {a['total_size_gb']:>17.2f}G {u['total_size_gb']:>17.2f}G")
    print(f"   {'Avg bits/weight':<30} {a['avg_bits_per_weight']:>18.2f} {u['bits']:>18.2f}")
    print(f"   {'Compression ratio':<30} {a['compression_ratio']:>17.2f}x {u['compression_ratio']:>17.2f}x")
    print(f"   {'Quality cost':<30} {a['total_quality_cost']:>18.6f} {u['total_quality_cost']:>18.6f}")

    # Quality improvement
    if u["total_quality_cost"] > 0:
        improvement = (1 - a["total_quality_cost"] / u["total_quality_cost"]) * 100
        if improvement > 0:
            result_str = _green(f"+{improvement:.1f}% better quality")
        else:
            result_str = _red(f"{improvement:.1f}% quality")
    else:
        improvement = 0
        result_str = "N/A (uniform cost = 0)"

    print(f"   {'─' * 68}")
    print(f"   Adaptive advantage: {result_str}")

    # Quant distribution
    dist = a["level_distribution"]
    levels_used = len(dist)
    print(f"   Quant levels used: {levels_used} ({', '.join(f'{q}:{n}' for q, n in sorted(dist.items(), key=lambda x: QUANT_OPTIONS.get(x[0], {}).get('bits', 0), reverse=True))})")

    if u.get("exceeds_budget"):
        print(f"\n   {_red('⚠ WARNING')}: Even the most aggressive uniform quant ({u['quant']}) exceeds the {budget_gb}GB budget.")
        print(f"   Uniform {u['quant']} size: {u['total_size_gb']:.2f}GB — model cannot fit uniformly.")

    print(f"   {'═' * 72}")


def print_kv_analysis(kv_results, model_name, model_weight_gb, total_memory_gb):
    """Print TurboQuant KV cache analysis across cache types."""
    headroom_gb = total_memory_gb - model_weight_gb

    print(f"\n   {'═' * 76}")
    print(f"   TURBOQUANT KV CACHE: {model_name}")
    print(f"   Model weights: {model_weight_gb:.2f}GB | "
          f"Total memory: {total_memory_gb:.1f}GB | "
          f"KV headroom: {headroom_gb:.2f}GB")
    print(f"   {'═' * 76}")

    print(f"\n   {'Cache Type':<26} {'Compress':>8} {'KV@16K':>8} {'KV@32K':>8} {'Max Context':>12}")
    print(f"   {'─' * 66}")

    for r in kv_results:
        kv_16k = r["cache_at_ref"].get(16384, 0)
        kv_32k = r["cache_at_ref"].get(32768, 0)
        max_ctx = r["max_context_k"]

        # Color code by context length
        max_tokens = r["max_context_length"]
        if max_tokens >= 32768:
            ctx_str = _green(f"{max_ctx:>12}")
        elif max_tokens >= 8192:
            ctx_str = _yellow(f"{max_ctx:>12}")
        elif max_tokens > 0:
            ctx_str = _red(f"{max_ctx:>12}")
        else:
            ctx_str = _red(f"{'NO FIT':>12}")

        compress = f"{r['compression_vs_f16']:.1f}x"
        print(f"   {r['label']:<26} {compress:>8} {kv_16k:>7.2f}G {kv_32k:>7.2f}G {ctx_str}")

    print(f"   {'─' * 66}")

    # Show the impact story
    f16_result = kv_results[0]  # f16 is first
    q4_result = kv_results[-1]  # q4_0 is last

    f16_ctx = f16_result["max_context_length"]
    q4_ctx = q4_result["max_context_length"]

    if f16_ctx > 0 and q4_ctx > 0:
        multiplier = q4_ctx / f16_ctx
        print("\n   TurboQuant impact (Q4_0 vs FP16):")
        print(f"     FP16 cache: {f16_result['max_context_k']} max context")
        print(f"     Q4_0 cache: {_green(q4_result['max_context_k'])} max context "
              f"({_green(f'{multiplier:.1f}x longer')})")
    elif q4_ctx > 0 and f16_ctx == 0:
        print(f"\n   {_green('TurboQuant enables context that FP16 cache cannot fit!')}")
        print(f"     FP16 cache: {_red('no room')} (model weights fill all memory)")
        print(f"     Q4_0 cache: {_green(q4_result['max_context_k'])} max context")
    elif q4_ctx == 0:
        print(f"\n   {_red('Model weights exceed memory budget — no room for KV cache at any setting.')}")

    print(f"   {'═' * 76}")
