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

    max_score = max(l["sensitivity_score"] for l in layers)

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
    print(f"   Max sensitivity: {_red(f'{max_score:.3f}')} (layer {max(layers, key=lambda l: l['sensitivity_score'])['index']})")
    print(f"   Min sensitivity: {_green(f'{min(l['sensitivity_score'] for l in layers):.3f}')} (layer {min(layers, key=lambda l: l['sensitivity_score'])['index']})")

    # Show the U-curve pattern
    n = num_layers
    thirds = [
        ("First 3 layers (input)", layers[:3]),
        ("Middle layers", layers[n//4:3*n//4]),
        ("Last 3 layers (output)", layers[-3:]),
    ]

    print(f"\n   U-Curve Summary:")
    for label, group in thirds:
        avg = sum(l["sensitivity_score"] for l in group) / len(group)
        bar = _sensitivity_bar(avg, max_score, width=15)
        print(f"     {label:<26} avg={avg:.3f} {bar}")


def print_allocation_table(allocation, profile):
    """Print the per-layer quantization allocation."""
    layers = allocation["layers"]
    num_layers = len(layers)

    print(f"\n   Allocation Summary:")
    print(f"   ├── Strategy: {allocation['strategy']}")
    print(f"   ├── Budget: {allocation['budget_gb']:.1f}GB")
    print(f"   ├── Model size: {_bold(f'{allocation['total_size_gb']:.2f}GB')}")
    print(f"   ├── Headroom: {allocation['headroom_gb']:.2f}GB (for KV cache + OS)")
    print(f"   ├── Avg bits/weight: {_bold(f'{allocation['avg_bits_per_weight']:.1f}')}")
    print(f"   ├── Compression: {allocation['compression_ratio']}x from BF16")
    print(f"   └── Quality cost: {allocation['total_quality_cost']:.4f}")

    # Distribution summary
    dist = allocation["level_distribution"]
    print(f"\n   Quantization Distribution:")
    for quant in QUANT_ORDER:
        if quant in dist:
            count = dist[quant]
            pct = count / num_layers * 100
            bar = "█" * int(pct / 2)
            color_icon = QUANT_OPTIONS[quant]["color"]
            bits = QUANT_OPTIONS[quant]["bits"]
            print(f"     {color_icon} {quant:<6} ({bits:>4.1f}b): {count:>3} layers ({pct:>5.1f}%) {bar}")

    # Comparison vs uniform Q4_K
    comp = allocation.get("comparison_vs_uniform_q4k", {})
    if comp:
        improvement = comp.get("adaptive_quality_improvement_pct", 0)
        size_diff = comp.get("adaptive_size_difference_gb", 0)
        print(f"\n   vs Uniform Q4_K:")
        if improvement > 0:
            print(f"     {_green(f'✓ {improvement:.1f}% better quality')} at {size_diff:+.2f}GB size difference")
        else:
            print(f"     {_yellow(f'{improvement:.1f}% quality difference')} at {size_diff:+.2f}GB size")

    # Compact layer view
    print(f"\n   Per-layer view (compact):")
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

    print(f"\n   Quality cost: lower = better (sensitivity-weighted quantization error)")
    print(f"   vs Q4_K: positive = adaptive preserves more quality at same avg bitrate")
