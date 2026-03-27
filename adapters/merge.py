"""
Merge multiple adapter outputs into a single sensitivity profile.

Normalizes each adapter's raw scores to [0, 1], then computes a
confidence-weighted average, then rescales to the range the
knapsack solver expects (~0.05 to 2.0).
"""


def _normalize(scores):
    """Min-max normalize a list of scores to [0, 1]."""
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-10:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _rescale(scores, target_min=0.05, target_max=2.0):
    """Rescale [0, 1] normalized scores to the target range."""
    return [target_min + s * (target_max - target_min) for s in scores]


def merge_adapter_outputs(adapter_outputs, num_layers):
    """
    Merge outputs from multiple adapters into a single score list + special_tensors.

    Args:
        adapter_outputs: list of dicts from adapter.extract()
        num_layers: expected layer count

    Returns:
        {
            "scores": [float, ...],  # merged sensitivity scores per layer
            "special_tensors": {...},  # merged special tensor overrides
            "sources": [{"name": ..., "confidence": ..., "metric": ...}, ...],
            "confidence": float,  # overall confidence
        }
    """
    valid_outputs = [o for o in adapter_outputs if o is not None]
    if not valid_outputs:
        raise ValueError("No valid adapter outputs to merge")

    # Normalize each adapter's scores independently
    normalized = []
    weights = []
    for output in valid_outputs:
        if len(output["scores"]) != num_layers:
            raise ValueError(
                f"Adapter '{output['adapter']}' returned {len(output['scores'])} scores "
                f"but expected {num_layers}"
            )
        normalized.append(_normalize(output["scores"]))
        weights.append(output["confidence"])

    # Weighted average across adapters
    total_weight = sum(weights)
    merged_normalized = []
    for i in range(num_layers):
        weighted_sum = sum(
            normalized[j][i] * weights[j]
            for j in range(len(valid_outputs))
        )
        merged_normalized.append(weighted_sum / total_weight)

    # Rescale to solver range
    merged_scores = _rescale(merged_normalized)

    # Merge special_tensors (later adapters override earlier ones)
    special_tensors = {}
    for output in valid_outputs:
        special_tensors.update(output.get("special_tensors", {}))

    # Sources provenance
    sources = [
        {
            "name": o["adapter"],
            "confidence": o["confidence"],
            "metric": o["metric"],
        }
        for o in valid_outputs
    ]

    # Overall confidence = max of individual confidences
    overall_confidence = max(o["confidence"] for o in valid_outputs)

    return {
        "scores": merged_scores,
        "special_tensors": special_tensors,
        "sources": sources,
        "confidence": overall_confidence,
    }
