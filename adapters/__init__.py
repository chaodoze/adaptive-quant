"""
Community data adapters for adaptive-quant.

Each adapter ingests one community data source and produces a partial profile:
a list of per-layer sensitivity scores plus metadata about the scoring methodology.

These are build-time tools for populating the catalog, not runtime dependencies.
"""


class BaseAdapter:
    """
    Base class for community data adapters.

    Subclasses implement extract() which returns a partial profile dict.
    """

    name: str = ""
    metric: str = ""        # e.g. "allocation_reverse", "kld", "rule_based"
    confidence: float = 0.5

    def extract(self, model_name, num_layers, **kwargs):
        """
        Extract sensitivity scores for a model.

        Returns:
            {
                "adapter": self.name,
                "metric": self.metric,
                "confidence": self.confidence,
                "num_layers": int,
                "scores": [float, ...],  # one per layer, raw scale
                "metadata": {...}
            }
        """
        raise NotImplementedError
