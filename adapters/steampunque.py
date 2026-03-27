"""
Adapter for steampunque's mixed-precision GGUF layer allocations.

Reverse-engineers sensitivity scores from known-good allocations:
a layer assigned higher precision must be more sensitive.

Source: huggingface.co/steampunque/Qwen3.5-35B-A3B-MP-GGUF
"""

from . import BaseAdapter

# steampunque's per-layer config for Qwen3.5-35B-A3B (Q4_K_H variant)
# Extracted from model card. Maps layer index to quant type assigned.
STEAMPUNQUE_35B_A3B = {
    # Layer allocations from the MP-GGUF model card
    # Edge layers get higher precision, middle layers get lower
    0: "Q5_K_S",
    1: "Q4_K_M",
    2: "Q4_K_S",
    3: "Q4_K_S",
    4: "Q4_K_S",
    5: "Q4_K_S",
    6: "Q4_K_S",
    7: "Q4_K_S",
    8: "Q4_K_S",
    9: "Q4_K_S",
    10: "Q4_K_S",
    11: "Q4_K_S",
    12: "Q4_K_S",
    13: "Q4_K_S",
    14: "Q4_K_S",
    15: "Q4_K_S",
    16: "Q4_K_S",
    17: "Q4_K_S",
    18: "Q4_K_S",
    19: "Q4_K_S",
    20: "Q4_K_S",
    21: "Q4_K_S",
    22: "Q4_K_S",
    23: "Q4_K_S",
    24: "Q4_K_S",
    25: "Q4_K_S",
    26: "Q4_K_S",
    27: "Q4_K_S",
    28: "Q4_K_M",
    29: "Q4_K_M",
    30: "Q4_K_M",
    31: "Q4_K_M",
    32: "Q4_K_L",
    33: "Q4_K_L",
    34: "Q5_K_S",
    35: "Q5_K_S",
    36: "Q5_K_M",
    37: "Q5_K_L",
    38: "Q6_K",
    39: "Q6_K",
}

# Bits per weight for each quant type (from knapsack.py QUANT_OPTIONS)
BITS_MAP = {
    "BF16": 16.0, "Q8_0": 8.0, "Q6_K": 6.5,
    "Q5_K_L": 5.75, "Q5_K_M": 5.5, "Q5_K_S": 5.25,
    "Q4_K_L": 4.75, "Q4_K_M": 4.5, "Q4_K_S": 4.25,
    "Q4_0": 4.0, "Q3_K": 3.4, "IQ3_XXS": 3.06,
    "Q2_K": 2.6, "IQ2_M": 2.2,
}

# Available allocation data
ALLOCATIONS = {
    "Qwen3.5-35B-A3B": STEAMPUNQUE_35B_A3B,
}


class SteampunqueAdapter(BaseAdapter):
    """
    Reverse-engineer sensitivity from steampunque's mixed-precision allocations.

    Algorithm: score(i) = bits(assigned_quant) / median_bits
    Higher precision assignment → higher inferred sensitivity.
    """

    name = "steampunque"
    metric = "allocation_reverse"
    confidence = 0.7

    def extract(self, model_name, num_layers, **kwargs):
        allocation = ALLOCATIONS.get(model_name)
        if allocation is None:
            return None

        bits_per_layer = [BITS_MAP[allocation[i]] for i in range(num_layers)]
        median_bits = sorted(bits_per_layer)[num_layers // 2]

        scores = [b / median_bits for b in bits_per_layer]

        return {
            "adapter": self.name,
            "metric": self.metric,
            "confidence": self.confidence,
            "num_layers": num_layers,
            "scores": scores,
            "metadata": {
                "source": "steampunque/Qwen3.5-35B-A3B-MP-GGUF",
                "median_bits": median_bits,
                "allocation": {str(k): v for k, v in allocation.items()},
            },
        }
