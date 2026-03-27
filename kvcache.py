"""
KV cache memory modeling for TurboQuant (quantized KV cache).

Calculates runtime memory for the KV cache at different quantization levels,
and determines maximum usable context length given a total memory budget.

Only full attention layers contribute to KV cache growth. DeltaNet/Mamba
layers use fixed-size recurrent state (~32KB per head, negligible).
"""


# KV cache quantization options for llama.cpp --cache-type-k / --cache-type-v
CACHE_TYPES = {
    "f16":  {"bits": 16,   "bytes_per_element": 2.0,    "label": "FP16 (default)"},
    "q8_0": {"bits": 8,    "bytes_per_element": 1.0625, "label": "Q8_0 (near-lossless)"},
    "q4_1": {"bits": 5,    "bytes_per_element": 0.625,  "label": "Q4_1 (4-bit + offset)"},
    "q4_0": {"bits": 4.5,  "bytes_per_element": 0.5625, "label": "Q4_0 (4-bit symmetric)"},
}

CACHE_TYPE_ORDER = ["f16", "q8_0", "q4_1", "q4_0"]


def kv_bytes_per_token(profile):
    """
    Compute KV cache bytes per token at FP16 for a model.

    Uses the kv_architecture field from the catalog profile.
    Formula: 2 * num_kv_layers * num_kv_heads * head_dim * 2 (bytes for FP16)
    """
    kv_arch = profile.get("kv_architecture")
    if not kv_arch:
        return 0

    num_kv_layers = kv_arch["num_kv_layers"]
    num_kv_heads = kv_arch["num_kv_heads"]
    head_dim = kv_arch["head_dim"]

    # 2 for K and V, 2 bytes for FP16
    return 2 * num_kv_layers * num_kv_heads * head_dim * 2


def kv_cache_size_gb(context_length, cache_type, profile):
    """
    Compute KV cache size in GB for a given context length and cache type.
    """
    fp16_bytes_per_token = kv_bytes_per_token(profile)
    if fp16_bytes_per_token == 0:
        return 0

    cache_info = CACHE_TYPES[cache_type]
    # Scale from FP16 baseline
    compression = cache_info["bytes_per_element"] / 2.0
    total_bytes = fp16_bytes_per_token * context_length * compression
    return total_bytes / (1024 ** 3)


def max_context_length(total_memory_gb, model_weight_gb, cache_type, profile):
    """
    Compute the maximum context length that fits in available memory.

    total_memory_gb: Total memory available (e.g., 26GB after OS overhead)
    model_weight_gb: Memory used by model weights
    cache_type: KV cache quantization type
    """
    available_for_kv = total_memory_gb - model_weight_gb
    if available_for_kv <= 0:
        return 0

    fp16_bytes_per_token = kv_bytes_per_token(profile)
    if fp16_bytes_per_token == 0:
        return 0

    cache_info = CACHE_TYPES[cache_type]
    compression = cache_info["bytes_per_element"] / 2.0
    bytes_per_token = fp16_bytes_per_token * compression

    available_bytes = available_for_kv * (1024 ** 3)
    return int(available_bytes / bytes_per_token)


def kv_analysis(profile, model_weight_gb, total_memory_gb):
    """
    Full KV cache analysis across all cache types.

    Returns a list of dicts with context length and cache size for each type.
    """
    kv_arch = profile.get("kv_architecture")
    if not kv_arch:
        return []

    results = []
    for cache_type in CACHE_TYPE_ORDER:
        cache_info = CACHE_TYPES[cache_type]
        max_ctx = max_context_length(total_memory_gb, model_weight_gb, cache_type, profile)

        # Also compute cache size at some reference context lengths
        ref_contexts = [4096, 16384, 32768, 65536, 131072]
        cache_at_ref = {}
        for ctx in ref_contexts:
            size = kv_cache_size_gb(ctx, cache_type, profile)
            cache_at_ref[ctx] = round(size, 3)

        results.append({
            "cache_type": cache_type,
            "label": cache_info["label"],
            "bits": cache_info["bits"],
            "compression_vs_f16": round(2.0 / cache_info["bytes_per_element"], 2),
            "max_context_length": max_ctx,
            "max_context_k": f"{max_ctx / 1024:.1f}K" if max_ctx < 1_000_000 else f"{max_ctx / 1_000_000:.2f}M",
            "cache_at_16k_gb": cache_at_ref.get(16384, 0),
            "cache_at_ref": cache_at_ref,
        })

    return results
