"""
Adapter for Kaitchup's shared-expert and attention sensitivity findings.

Source: kaitchup.substack.com/p/qwen35-quantization-similar-accuracy

Key findings:
- Shared expert tensors (ffn_*_shexp) should NOT be quantized — keep BF16
- Attention layers (especially attn_v, attn_o) need high precision
- Linear attention sensitivity shows up during long-sequence generation
- Quantizing shared expert causes significant accuracy drops despite
  representing only ~2% of model size
"""

from . import BaseAdapter


# Kaitchup's rules encoded as per-layer sensitivity modifiers
# These are additive boosts based on architectural features
RULES = {
    # Full attention layers get a sensitivity boost
    # (linear attention in DeltaNet layers is somewhat more resilient
    #  but still sensitive during long-context reasoning per Kaitchup)
    "full_attention_boost": 0.15,

    # Edge layers (first/last) always sensitive (universal finding)
    "edge_layer_boost_first": 0.25,
    "edge_layer_boost_last": 0.20,
    "near_edge_boost": 0.10,  # layers 1-2 and N-3 to N-2
}


class KaitchupAdapter(BaseAdapter):
    """
    Encode Kaitchup's MoE quantization rules as sensitivity modifiers.

    Produces:
    - Per-layer sensitivity boosts for attention-heavy and edge layers
    - special_tensors dict for shared experts → BF16
    """

    name = "kaitchup"
    metric = "rule_based"
    confidence = 0.8

    def extract(self, model_name, num_layers, **kwargs):
        architecture = kwargs.get("architecture", "dense")

        # Determine hybrid pattern (3:1 DeltaNet:Attention for Qwen3.5)
        attn_period = 4
        attn_positions = {3}  # Every 4th layer is full attention

        scores = []
        for i in range(num_layers):
            score = 0.5  # Base score for all layers

            # Edge layer boosts
            if i == 0:
                score += RULES["edge_layer_boost_first"]
            elif i == num_layers - 1:
                score += RULES["edge_layer_boost_last"]
            elif i <= 2 or i >= num_layers - 3:
                score += RULES["near_edge_boost"]

            # Full attention layers get a boost
            if architecture in ("moe_hybrid", "hybrid"):
                if i % attn_period in attn_positions:
                    score += RULES["full_attention_boost"]
            else:
                # Dense: every layer has full attention
                score += RULES["full_attention_boost"]

            scores.append(score)

        # Build special_tensors for MoE models
        special_tensors = {}
        if architecture in ("moe_hybrid", "moe"):
            # Kaitchup's key finding: don't quantize shared experts
            for i in range(num_layers):
                for tensor in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"]:
                    special_tensors[f"blk.{i}.{tensor}"] = "BF16"

        # Embeddings and output head at Q6_K (community consensus)
        special_tensors["token_embd.weight"] = "Q6_K"
        special_tensors["output.weight"] = "Q6_K"

        return {
            "adapter": self.name,
            "metric": self.metric,
            "confidence": self.confidence,
            "num_layers": num_layers,
            "scores": scores,
            "special_tensors": special_tensors,
            "metadata": {
                "source": "kaitchup.substack.com/p/qwen35-quantization-similar-accuracy",
                "rules": RULES,
            },
        }
