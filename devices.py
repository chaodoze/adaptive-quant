"""
Device memory presets for common Apple hardware.

Memory budgets account for OS overhead, leaving the remainder for model weights + KV cache.
"""

DEVICE_PRESETS = {
    # M5 generation (March 2026)
    "m5-max-128gb": {
        "name": "MacBook Pro M5 Max 128GB",
        "memory_gb": 128,
        "os_overhead_gb": 12,
        "bandwidth_gbs": 614,
        "neural_engine_tops": 38,
    },
    "m5-pro-64gb": {
        "name": "MacBook Pro M5 Pro 64GB",
        "memory_gb": 64,
        "os_overhead_gb": 8,
        "bandwidth_gbs": 307,
        "neural_engine_tops": 38,
    },
    "m5-pro-36gb": {
        "name": "MacBook Pro M5 Pro 36GB",
        "memory_gb": 36,
        "os_overhead_gb": 7,
        "bandwidth_gbs": 307,
        "neural_engine_tops": 38,
    },
    "m5-air-24gb": {
        "name": "MacBook Air M5 24GB",
        "memory_gb": 24,
        "os_overhead_gb": 6,
        "bandwidth_gbs": 150,
        "neural_engine_tops": 38,
    },
    "m5-air-16gb": {
        "name": "MacBook Air M5 16GB",
        "memory_gb": 16,
        "os_overhead_gb": 5,
        "bandwidth_gbs": 150,
        "neural_engine_tops": 38,
    },

    # M4 generation (2024-2025)
    "m4-pro-48gb": {
        "name": "MacBook Pro M4 Pro 48GB",
        "memory_gb": 48,
        "os_overhead_gb": 8,
        "bandwidth_gbs": 273,
        "neural_engine_tops": 38,
    },
    "m4-air-16gb": {
        "name": "MacBook Air M4 16GB",
        "memory_gb": 16,
        "os_overhead_gb": 5,
        "bandwidth_gbs": 120,
        "neural_engine_tops": 38,
    },

    # Mobile
    "iphone-17-pro-8gb": {
        "name": "iPhone 17 Pro 8GB",
        "memory_gb": 8,
        "os_overhead_gb": 4,
        "bandwidth_gbs": 75,
        "neural_engine_tops": 35,
    },
    "ipad-pro-m5-16gb": {
        "name": "iPad Pro M5 16GB",
        "memory_gb": 16,
        "os_overhead_gb": 5,
        "bandwidth_gbs": 150,
        "neural_engine_tops": 38,
    },

    # M4 generation (2024-2025) - additional configs
    "m4-air-32gb": {
        "name": "MacBook Air M4 32GB",
        "memory_gb": 32,
        "os_overhead_gb": 6,
        "bandwidth_gbs": 120,
        "neural_engine_tops": 38,
    },
}
