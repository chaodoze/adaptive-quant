"""
Model metadata registry for the catalog.

Each entry provides metadata for listing/searching without loading full profiles.
"""

REGISTRY = [
    {
        "name": "Qwen3.5-35B-A3B",
        "file": "qwen3.5-35b-a3b.json",
        "family": "qwen3.5",
        "architecture": "moe_hybrid",
        "params_b": 35.0,
        "active_params_b": 3.0,
        "bf16_size_gb": 69.4,
        "num_layers": 40,
        "sources": ["steampunque", "unsloth_kld", "kaitchup"],
        "description": "MoE hybrid (DeltaNet+Attention). 256 experts, 8 routed + 1 shared.",
    },
    {
        "name": "Qwen3.5-27B",
        "file": "qwen3.5-27b.json",
        "family": "qwen3.5",
        "architecture": "hybrid",
        "params_b": 27.0,
        "active_params_b": None,
        "bf16_size_gb": 54.0,
        "num_layers": 64,
        "sources": ["unsloth_kld", "kaitchup"],
        "description": "Dense hybrid (DeltaNet+Attention). No MoE.",
    },
    {
        "name": "Qwen3.5-122B-A10B",
        "file": "qwen3.5-122b-a10b.json",
        "family": "qwen3.5",
        "architecture": "moe_hybrid",
        "params_b": 122.0,
        "active_params_b": 10.0,
        "bf16_size_gb": 240.0,
        "num_layers": 48,
        "sources": ["unsloth_kld", "kaitchup"],
        "description": "MoE hybrid (DeltaNet+Attention). 256 experts, 8 routed + 1 shared.",
    },
]
