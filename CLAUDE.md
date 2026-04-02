# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Adaptive mixed-precision quantization tool for GGUF models (llama.cpp ecosystem). Instead of uniformly quantizing every layer to the same precision, it ingests community sensitivity data (steampunque, Unsloth KLD benchmarks, Kaitchup) and uses a knapsack solver to assign higher precision to sensitive layers and lower precision to resilient ones — fitting within a target memory budget.

## Running

```bash
# Run from parent directory (adaptive-quant dir needs adaptive_quant symlink)
cd /path/to/parent && ln -sf adaptive-quant adaptive_quant

# Main workflow: community-sourced profiles
python -m adaptive_quant catalog list
python -m adaptive_quant benchmark Qwen3.5-35B-A3B --budget 26
python -m adaptive_quant optimize Qwen3.5-35B-A3B --device m4-air-32gb --dry-run
python -m adaptive_quant optimize Qwen3.5-35B-A3B --budget 26 --script

# Original workflow: DIY profiling (needs llama.cpp)
python -m adaptive_quant profile <model.gguf>
python -m adaptive_quant generate <profile.json> --budget 24 --dry-run

# Demo with simulated data
python -m adaptive_quant demo
```

No `setup.py` or `pyproject.toml` yet — run directly as a module. No external dependencies beyond stdlib (numpy import was removed).

Real profiling/generation requires `llama-perplexity` and `llama-quantize` from llama.cpp (`brew install llama.cpp`).

## Linting

```bash
ruff check .
ruff check --fix .
```

## Architecture

Two paths to the same solver: **Community Catalog** or **DIY Profiling** → **Knapsack Solve** → **Generate GGUF**

### Core Pipeline

1. **catalog/** — Pre-built sensitivity profiles from community data. `catalog/__init__.py` provides `list_models()`, `get_profile(name)`. Profiles live as JSON in `catalog/profiles/`. Registry in `catalog/_registry.py`.

2. **adapters/** — Build-time tools that ingest community data sources and produce normalized sensitivity scores. Not runtime dependencies.
   - `steampunque.py` — Reverse-engineers scores from known-good layer allocations (`score = bits_assigned / median_bits`)
   - `unsloth_kld.py` — Tensor-level KLD sensitivity rankings (ssm_output critical, attention high, ffn_exps low)
   - `kaitchup.py` — MoE rules: shared experts → BF16, attention layers boosted, edge layers boosted
   - `merge.py` — Min-max normalizes each adapter to [0,1], confidence-weighted average, rescales to [0.05, 2.0]

3. **knapsack.py** — `KnapsackSolver` takes a sensitivity profile + memory budget. Greedy downgrade phase picks most efficient moves (memory-saved / quality-cost), then upgrade phase gives back precision to sensitive layers. 14 quant levels (BF16 through IQ2_M including S/M/L variants). `best_uniform_for_budget()` finds comparison baseline.

4. **generator.py** — `GGUFGenerator` builds `llama-quantize` commands with per-tensor `--tensor-type` overrides. Architecture-aware: selects dense, hybrid (DeltaNet+Attention), or MoE tensor lists. Handles `special_tensors` (shared experts at BF16, embeddings at Q6_K).

5. **profiler.py** — `SensitivityProfiler` (real, needs llama.cpp) and `SimulatedProfiler` (synthetic U-curve). Original DIY path.

### Supporting Modules
- **devices.py** — Device presets mapping slugs to memory/bandwidth specs
- **visualizer.py** — Terminal rendering: heatmaps, allocation tables, catalog listing, benchmarks
- **cli.py** — Subcommands: `catalog`, `optimize`, `benchmark`, `profile`, `generate`, `demo`

## Key Design Decisions

- **Catalog is static JSON, not runtime HuggingFace fetching.** Profiles are curated once by running adapters, then committed. Fast, offline, auditable.
- **Adapters are build-time, not runtime.** They parse messy upstream formats. Run during development to populate catalog.
- **14 quant levels with S/M/L variants** (Q4_K_S/M/L, Q5_K_S/M/L). `resolve_quant()` maps old names (Q4_K → Q4_K_M).
- **`special_tensors`** bypass the solver — shared experts at BF16 per Kaitchup, embeddings/output at Q6_K per community consensus.
- **Hybrid architecture support** — Qwen3.5 uses 3:1 DeltaNet:Attention pattern. ssm_output tensors are marked critical (Unsloth finding).

## Cataloged Models

| Model | Layers | Architecture | Sources |
|-------|--------|-------------|---------|
| Qwen3.5-35B-A3B | 40 | MoE hybrid | steampunque + Unsloth + Kaitchup |
| Qwen3.5-27B | 64 | Dense hybrid | Unsloth + Kaitchup |
| Qwen3.5-122B-A10B | 48 | MoE hybrid | Unsloth + Kaitchup |

## Hardware Learnings (M4 MacBook Air 32GB)

### GPU Memory Reality
- Metal reports `recommendedMaxWorkingSetSize = 22.9GB`, but actual usable is lower due to compute buffer overhead
- A 21GB model (Q5_K_S) OOMs even at `-ngl 20` — the unified memory pool is shared with OS (~6-8GB)
- **Practical GPU limit: ~15-18GB model weights** to leave room for Metal buffers, KV cache, and OS
- CPU-only (`-ngl 0`) works but is extremely slow for 20GB+ models

### Quantization Workflow
- BF16 source GGUFs from `unsloth/<model>-GGUF` (e.g., `unsloth/Qwen3.5-27B-GGUF`)
- `llama-quantize` flags go **before** positional args (source, output, type)
- Use `--tensor-type-file` for many overrides instead of inline `--tensor-type` flags
- Use `--token-embedding-type` and `--output-tensor-type` for special tensors (not `--tensor-type`)
- S/M/L variants (Q5_K_S, Q5_K_M, Q5_K_L) are **base presets** — per-tensor overrides only accept `q5_k`, `q4_k`, etc.
- Quantization takes ~5-6 minutes for a 35B model on M4 Air

### Model Sizing for This Hardware
- **35B-A3B at 22GB (Q5_K_S)**: OOMs on GPU. Too big for M4 Air 32GB.
- **35B-A3B at 15GB (Q3_K)**: Fits GPU but quality cost is high (2.27). MoE helps since only 3B active.
- **27B at 15GB (Q4_K_M)**: Best balance — dense model, all params active, quality cost 1.60. Leaves ~8GB for GPU overhead + KV cache.
- **27B at 18GB (Q5_K_S)**: Higher quality (cost 0.67) but tighter GPU fit.
- **Recommended: Qwen3.5-27B at Q4_K_M (~15GB)** for practical daily use on this hardware.

### TurboQuant KV Cache
- Run with `--cache-type-k q4_0 --cache-type-v q4_0` for 3.6x longer context vs FP16 cache
- Only full attention layers (every 4th in Qwen3.5 hybrid) use KV cache; DeltaNet layers use fixed-size state
