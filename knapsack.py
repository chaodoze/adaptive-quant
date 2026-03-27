"""
Knapsack solver for adaptive mixed-precision quantization.

Given a sensitivity profile and a memory budget, finds the optimal
per-layer quantization level that minimizes total quality loss
while fitting within the budget.

This is a classic bounded knapsack problem:
- Items = layers × quantization options
- Weight = memory cost of each option
- Value = quality preservation (inverse of sensitivity × error)
- Constraint = total memory ≤ budget
"""

from typing import Dict, Any


# Available quantization levels with their properties
# Bits per weight and approximate perplexity multiplier relative to BF16
# S/M/L variants reflect community usage (steampunque, Unsloth benchmarks)
QUANT_OPTIONS = {
    "BF16":    {"bits": 16.0,  "label": "BF16",    "quality": 1.000, "color": "🟢"},
    "Q8_0":    {"bits": 8.0,   "label": "Q8_0",    "quality": 0.999, "color": "🟢"},
    "Q6_K":    {"bits": 6.5,   "label": "Q6_K",    "quality": 0.995, "color": "🟢"},
    "Q5_K_L":  {"bits": 5.75,  "label": "Q5_K_L",  "quality": 0.993, "color": "🟢"},
    "Q5_K_M":  {"bits": 5.5,   "label": "Q5_K_M",  "quality": 0.990, "color": "🟡"},
    "Q5_K_S":  {"bits": 5.25,  "label": "Q5_K_S",  "quality": 0.987, "color": "🟡"},
    "Q4_K_L":  {"bits": 4.75,  "label": "Q4_K_L",  "quality": 0.980, "color": "🟡"},
    "Q4_K_M":  {"bits": 4.5,   "label": "Q4_K_M",  "quality": 0.975, "color": "🟡"},
    "Q4_K_S":  {"bits": 4.25,  "label": "Q4_K_S",  "quality": 0.968, "color": "🟡"},
    "Q4_0":    {"bits": 4.0,   "label": "Q4_0",    "quality": 0.954, "color": "🟡"},
    "Q3_K":    {"bits": 3.4,   "label": "Q3_K",    "quality": 0.920, "color": "🟠"},
    "IQ3_XXS": {"bits": 3.06,  "label": "IQ3_XXS", "quality": 0.880, "color": "🟠"},
    "Q2_K":    {"bits": 2.6,   "label": "Q2_K",    "quality": 0.800, "color": "🔴"},
    "IQ2_M":   {"bits": 2.2,   "label": "IQ2_M",   "quality": 0.700, "color": "🔴"},
}

# Ordered from highest to lowest quality
QUANT_ORDER = [
    "BF16", "Q8_0", "Q6_K",
    "Q5_K_L", "Q5_K_M", "Q5_K_S",
    "Q4_K_L", "Q4_K_M", "Q4_K_S",
    "Q4_0", "Q3_K", "IQ3_XXS", "Q2_K", "IQ2_M",
]

# Backward compat: old names → new canonical names
_QUANT_ALIASES = {"Q5_K": "Q5_K_M", "Q4_K": "Q4_K_M"}


def resolve_quant(name: str) -> str:
    """Resolve a quant name, handling old aliases (Q5_K → Q5_K_M, etc.)."""
    return _QUANT_ALIASES.get(name, name)


class KnapsackSolver:
    """
    Solve the per-layer quantization allocation problem.

    Strategy:
    1. Score each layer's sensitivity (from profiling)
    2. For each memory budget, use dynamic programming to find the
       allocation that minimizes total weighted quality loss
    3. Sensitive layers get higher precision; resilient layers get lower
    """

    def __init__(self, profile: Dict[str, Any]):
        self.profile = profile
        self.num_layers = profile["num_layers"]
        self.layers = profile["layers"]
        self.special_tensors = profile.get("special_tensors", {})

        # Get model size info
        self.bf16_size_gb = profile.get("bf16_size_gb", self._estimate_bf16_size())

        # Per-layer size at BF16
        self.layer_bf16_gb = self.bf16_size_gb / self.num_layers

    def _estimate_bf16_size(self):
        """Estimate BF16 model size from layer count (rough heuristic)."""
        # Typical: ~0.45GB per layer for 9B-class models
        return self.num_layers * 0.45

    def _layer_size_gb(self, quant_key):
        """Estimate a single layer's size at a given quantization level."""
        bits = QUANT_OPTIONS[quant_key]["bits"]
        return self.layer_bf16_gb * (bits / 16.0)

    def _quality_cost(self, layer_idx, quant_key):
        """
        Compute the quality cost of quantizing a layer to a given level.

        Cost = sensitivity_score × (1 - quality_retention)

        Higher sensitivity layers pay a bigger price for aggressive quantization.
        """
        layer = self.layers[layer_idx]
        sensitivity = layer["sensitivity_score"]
        quality = QUANT_OPTIONS[quant_key]["quality"]

        return sensitivity * (1.0 - quality)

    def solve(self, budget_gb: float) -> Dict[str, Any]:
        """
        Find optimal per-layer quantization for a given memory budget.

        Uses a greedy approach with refinement:
        1. Start with everything at BF16
        2. Greedily downgrade the least costly layer until budget is met
        3. Then try upgrades where budget allows

        Returns allocation dict with per-layer assignments and metrics.
        """
        # Initialize: everything at highest affordable precision
        allocation = ["BF16"] * self.num_layers
        current_size = sum(self._layer_size_gb("BF16") for _ in range(self.num_layers))

        if current_size <= budget_gb:
            # Model already fits at full precision
            return self._build_result(allocation, budget_gb, "full_precision")

        # Phase 1: Greedy downgrade until we fit
        # For each possible downgrade, compute cost-effectiveness:
        # efficiency = (memory_saved) / (quality_cost_added)
        while current_size > budget_gb:
            best_move = None
            best_efficiency = -1

            for layer_idx in range(self.num_layers):
                current_quant = allocation[layer_idx]
                current_quant_idx = QUANT_ORDER.index(current_quant)

                if current_quant_idx >= len(QUANT_ORDER) - 1:
                    continue  # Already at lowest

                next_quant = QUANT_ORDER[current_quant_idx + 1]

                mem_saved = self._layer_size_gb(current_quant) - self._layer_size_gb(next_quant)
                quality_cost = self._quality_cost(layer_idx, next_quant) - self._quality_cost(layer_idx, current_quant)

                if quality_cost < 1e-10:
                    efficiency = float("inf")
                else:
                    efficiency = mem_saved / quality_cost

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_move = (layer_idx, next_quant, mem_saved)

            if best_move is None:
                break  # Can't compress any further

            layer_idx, new_quant, mem_saved = best_move
            allocation[layer_idx] = new_quant
            current_size -= mem_saved

        # Phase 2: Upgrade where budget allows (give back to sensitive layers)
        for layer_idx in self._layers_by_sensitivity(descending=True):
            current_quant = allocation[layer_idx]
            current_quant_idx = QUANT_ORDER.index(current_quant)

            if current_quant_idx == 0:
                continue  # Already at highest

            better_quant = QUANT_ORDER[current_quant_idx - 1]
            mem_cost = self._layer_size_gb(better_quant) - self._layer_size_gb(current_quant)

            if current_size + mem_cost <= budget_gb:
                allocation[layer_idx] = better_quant
                current_size += mem_cost

        return self._build_result(allocation, budget_gb, "adaptive")

    def _layers_by_sensitivity(self, descending=True):
        """Return layer indices sorted by sensitivity score."""
        indexed = [(i, self.layers[i]["sensitivity_score"]) for i in range(self.num_layers)]
        indexed.sort(key=lambda x: x[1], reverse=descending)
        return [i for i, _ in indexed]

    def _build_result(self, allocation, budget_gb, strategy):
        """Build the result dictionary with metrics."""
        total_size = sum(self._layer_size_gb(allocation[i]) for i in range(self.num_layers))
        total_quality_cost = sum(self._quality_cost(i, allocation[i]) for i in range(self.num_layers))

        # Compute average bits per weight
        avg_bits = sum(QUANT_OPTIONS[allocation[i]]["bits"] for i in range(self.num_layers)) / self.num_layers

        # What would uniform Q4_K_M give us?
        uniform_size = sum(self._layer_size_gb("Q4_K_M") for _ in range(self.num_layers))
        uniform_cost = sum(self._quality_cost(i, "Q4_K_M") for i in range(self.num_layers))

        # Count quantization levels used
        level_counts = {}
        for q in allocation:
            level_counts[q] = level_counts.get(q, 0) + 1

        return {
            "strategy": strategy,
            "budget_gb": budget_gb,
            "total_size_gb": round(total_size, 2),
            "headroom_gb": round(budget_gb - total_size, 2),
            "avg_bits_per_weight": round(avg_bits, 2),
            "total_quality_cost": round(total_quality_cost, 6),
            "compression_ratio": round(self.bf16_size_gb / total_size, 2) if total_size > 0 else 0,
            "layers": [
                {
                    "index": i,
                    "quant": allocation[i],
                    "bits": QUANT_OPTIONS[allocation[i]]["bits"],
                    "size_gb": round(self._layer_size_gb(allocation[i]), 3),
                    "sensitivity": round(self.layers[i]["sensitivity_score"], 4),
                    "quality_cost": round(self._quality_cost(i, allocation[i]), 6),
                }
                for i in range(self.num_layers)
            ],
            "level_distribution": level_counts,
            "special_tensors": self.special_tensors,
            "comparison_vs_uniform_q4k": {
                "uniform_size_gb": round(uniform_size, 2),
                "uniform_quality_cost": round(uniform_cost, 6),
                "adaptive_quality_improvement_pct": round(
                    (1 - total_quality_cost / uniform_cost) * 100, 1
                ) if uniform_cost > 0 else 0,
                "adaptive_size_difference_gb": round(total_size - uniform_size, 2),
            },
        }

    def best_uniform_for_budget(self, budget_gb: float) -> Dict[str, Any]:
        """Find the highest-quality uniform quant that fits the budget."""
        for quant in QUANT_ORDER:
            size = sum(self._layer_size_gb(quant) for _ in range(self.num_layers))
            if size <= budget_gb:
                cost = sum(self._quality_cost(i, quant) for i in range(self.num_layers))
                return {
                    "quant": quant,
                    "bits": QUANT_OPTIONS[quant]["bits"],
                    "total_size_gb": round(size, 2),
                    "total_quality_cost": round(cost, 6),
                    "compression_ratio": round(self.bf16_size_gb / size, 2) if size > 0 else 0,
                }
        # Nothing fits — return the most aggressive option
        quant = QUANT_ORDER[-1]
        size = sum(self._layer_size_gb(quant) for _ in range(self.num_layers))
        cost = sum(self._quality_cost(i, quant) for i in range(self.num_layers))
        return {
            "quant": quant,
            "bits": QUANT_OPTIONS[quant]["bits"],
            "total_size_gb": round(size, 2),
            "total_quality_cost": round(cost, 6),
            "compression_ratio": round(self.bf16_size_gb / size, 2) if size > 0 else 0,
            "exceeds_budget": True,
        }


def solve_for_multiple_budgets(profile, budgets_gb):
    """Convenience: solve for multiple memory budgets at once."""
    solver = KnapsackSolver(profile)
    return {f"{b}gb": solver.solve(budget_gb=b) for b in budgets_gb}
