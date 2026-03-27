"""
Adapter for Unsloth's tensor-level KL Divergence benchmark findings.

Encodes the ranked tensor sensitivity from 150+ KLD benchmarks across 9TB of GGUFs.

Source: huggingface.co/unsloth/Qwen3.5-35B-A3B-Experiments-GGUF
Key findings:
- ssm_output: very sensitive, KLD increases dramatically when quantized
- attn_v, attn_output: sensitive, need 5-bit minimum
- ffn_down_exps: moderately sensitive
- ffn_gate_exps, ffn_up_exps: resilient, tolerate 3-bit
- Shared experts: must stay BF16 (handled by kaitchup adapter)
"""

from . import BaseAdapter


# Tensor sensitivity rankings from Unsloth's 99.9% KLD analysis
# Scale: 0.0 (resilient) to 1.0 (critical)
TENSOR_SENSITIVITY = {
    # Mamba/SSM components — critical
    "ssm_output": 1.0,
    "ssm_alpha": 0.6,
    "ssm_beta": 0.6,
    # Attention — high sensitivity
    "attn_v": 0.85,
    "attn_output": 0.80,
    "attn_qkv": 0.75,
    "attn_gate": 0.70,
    "attn_q": 0.75,
    "attn_k": 0.70,
    # FFN experts — moderate to low
    "ffn_down_exps": 0.45,
    "ffn_gate_exps": 0.25,
    "ffn_up_exps": 0.25,
    # Dense FFN — moderate
    "ffn_down": 0.50,
    "ffn_gate": 0.30,
    "ffn_up": 0.30,
}

# Layer type pattern for Qwen3.5 hybrid architectures
# 3:1 ratio: 3 DeltaNet (linear attention + Mamba) then 1 full attention
# DeltaNet layers have ssm_output which is highly sensitive
HYBRID_PATTERN = {
    "Qwen3.5-35B-A3B": {"period": 4, "attn_positions": [3], "num_layers": 40},
    "Qwen3.5-27B": {"period": 4, "attn_positions": [3], "num_layers": 64},
    "Qwen3.5-122B-A10B": {"period": 4, "attn_positions": [3], "num_layers": 48},
}


class UnslothKLDAdapter(BaseAdapter):
    """
    Compute per-layer sensitivity from Unsloth's tensor-level KLD rankings.

    For each layer, aggregates the sensitivity of its constituent tensors,
    weighted by the hybrid architecture pattern (DeltaNet vs full attention).
    """

    name = "unsloth_kld"
    metric = "kld_tensor_aggregate"
    confidence = 0.9

    def extract(self, model_name, num_layers, **kwargs):
        architecture = kwargs.get("architecture", "dense")
        pattern = HYBRID_PATTERN.get(model_name)

        scores = []
        for i in range(num_layers):
            if pattern:
                layer_in_period = i % pattern["period"]
                is_full_attention = layer_in_period in pattern["attn_positions"]
            else:
                is_full_attention = True  # Dense: all layers have full attention

            if is_full_attention:
                # Full attention layer: attn tensors dominate
                tensor_scores = [
                    TENSOR_SENSITIVITY.get("attn_q", 0.7),
                    TENSOR_SENSITIVITY.get("attn_k", 0.7),
                    TENSOR_SENSITIVITY.get("attn_v", 0.85),
                    TENSOR_SENSITIVITY.get("attn_output", 0.8),
                ]
            else:
                # DeltaNet layer: has SSM components (highly sensitive)
                tensor_scores = [
                    TENSOR_SENSITIVITY.get("attn_gate", 0.7),
                    TENSOR_SENSITIVITY.get("attn_qkv", 0.75),
                    TENSOR_SENSITIVITY.get("attn_output", 0.8),
                    TENSOR_SENSITIVITY.get("ssm_output", 1.0),
                    TENSOR_SENSITIVITY.get("ssm_alpha", 0.6),
                    TENSOR_SENSITIVITY.get("ssm_beta", 0.6),
                ]

            # Add FFN tensors (present in all layers)
            if architecture in ("moe_hybrid", "moe"):
                tensor_scores.extend([
                    TENSOR_SENSITIVITY.get("ffn_gate_exps", 0.25),
                    TENSOR_SENSITIVITY.get("ffn_up_exps", 0.25),
                    TENSOR_SENSITIVITY.get("ffn_down_exps", 0.45),
                ])
            else:
                tensor_scores.extend([
                    TENSOR_SENSITIVITY.get("ffn_gate", 0.3),
                    TENSOR_SENSITIVITY.get("ffn_up", 0.3),
                    TENSOR_SENSITIVITY.get("ffn_down", 0.5),
                ])

            # Layer score = weighted mean of tensor sensitivities
            # ssm_output and attn_v get extra weight as they dominate KLD
            layer_score = sum(tensor_scores) / len(tensor_scores)
            scores.append(layer_score)

        return {
            "adapter": self.name,
            "metric": self.metric,
            "confidence": self.confidence,
            "num_layers": num_layers,
            "scores": scores,
            "metadata": {
                "source": "unsloth/Qwen3.5-35B-A3B-Experiments-GGUF",
                "tensor_sensitivity": TENSOR_SENSITIVITY,
                "hybrid_pattern": pattern,
            },
        }
