"""
Generate mixed-precision GGUF files using llama-quantize.

Takes a knapsack allocation and produces a single GGUF with
per-layer quantization levels.
"""

import os
import subprocess


class GGUFGenerator:
    """
    Generate mixed-precision GGUF files.

    Uses llama-quantize's --tensor-type flag to apply different
    quantization levels to different layers.
    """

    # Mapping from our quant names to llama.cpp tensor-type strings
    QUANT_MAP = {
        "BF16":    "bf16",
        "Q8_0":    "q8_0",
        "Q6_K":    "q6_k",
        "Q5_K_L":  "q5_k",     # llama.cpp uses q5_k for all S/M/L
        "Q5_K_M":  "q5_k",
        "Q5_K_S":  "q5_k",
        "Q4_K_L":  "q4_k",
        "Q4_K_M":  "q4_k",
        "Q4_K_S":  "q4_k",
        "Q4_0":    "q4_0",
        "Q3_K":    "q3_k",
        "IQ3_XXS": "iq3_xxs",
        "Q2_K":    "q2_k",
        "IQ2_M":   "iq2_m",
        # Legacy aliases
        "Q5_K":    "q5_k",
        "Q4_K":    "q4_k",
    }

    # Architecture-specific tensor name patterns per layer
    DENSE_TENSORS = [
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]

    HYBRID_TENSORS = [
        "attn_gate.weight",
        "attn_qkv.weight",
        "attn_output.weight",
        "ssm_alpha.weight",
        "ssm_beta.weight",
        "ssm_output.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]

    MOE_TENSORS = [
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    ]

    SHARED_EXPERT_TENSORS = [
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ]

    def __init__(self, llama_quantize="llama-quantize"):
        self.llama_quantize = llama_quantize

    def _get_layer_tensors(self, architecture):
        """Select the right tensor list based on model architecture."""
        if architecture in ("moe_hybrid", "moe"):
            return self.HYBRID_TENSORS + self.MOE_TENSORS
        elif architecture == "hybrid":
            return self.HYBRID_TENSORS
        else:
            return self.DENSE_TENSORS

    def _build_overrides(self, allocation, base_quant, architecture):
        """Build --tensor-type override arguments."""
        layer_tensors = self._get_layer_tensors(architecture)
        overrides = []

        # Per-layer overrides
        for layer in allocation["layers"]:
            if layer["quant"] != base_quant:
                quant_str = self.QUANT_MAP[layer["quant"]]
                layer_idx = layer["index"]
                for tensor_name in layer_tensors:
                    overrides.append(f"blk.{layer_idx}.{tensor_name}={quant_str}")

        # Special tensor overrides (embeddings, output head, shared experts)
        special = allocation.get("special_tensors", {})
        for tensor_pattern, quant_name in special.items():
            quant_str = self.QUANT_MAP.get(quant_name, quant_name.lower())
            if "*" in tensor_pattern:
                # Wildcard patterns need to be expanded per layer
                num_layers = len(allocation["layers"])
                for i in range(num_layers):
                    expanded = tensor_pattern.replace("*.", f"blk.{i}.")
                    overrides.append(f"{expanded}={quant_str}")
            else:
                overrides.append(f"{tensor_pattern}={quant_str}")

        return overrides

    def generate(self, source_model, allocation, output_path, architecture="dense"):
        """
        Generate a mixed-precision GGUF file.

        Args:
            source_model: Path to BF16/F16 source GGUF
            allocation: Result from KnapsackSolver.solve()
            output_path: Where to write the output GGUF
            architecture: Model architecture type
        """
        if not os.path.exists(source_model):
            raise FileNotFoundError(f"Source model not found: {source_model}")

        # Determine the most common quantization level (used as the base)
        level_dist = allocation["level_distribution"]
        base_quant = max(level_dist, key=level_dist.get)
        base_quant_str = self.QUANT_MAP[base_quant]

        print("\n   🔧 Generating adaptive GGUF")
        print(f"   Base quantization: {base_quant} ({level_dist[base_quant]}/{len(allocation['layers'])} layers)")
        print(f"   Architecture: {architecture}")
        print(f"   Source: {source_model}")
        print(f"   Output: {output_path}")

        # Build the llama-quantize command
        cmd = [
            self.llama_quantize,
            source_model,
            output_path,
            base_quant_str,
        ]

        overrides = self._build_overrides(allocation, base_quant, architecture)
        for override in overrides:
            cmd.extend(["--tensor-type", override])

        override_layers = sum(1 for layer in allocation["layers"] if layer["quant"] != base_quant)
        print(f"   Overrides: {override_layers} layers differ from base, {len(overrides)} tensor overrides total")
        print("   Running llama-quantize...")

        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"llama-quantize failed (exit {result.returncode}):\n"
                f"STDOUT: {result.stdout[-500:]}\n"
                f"STDERR: {result.stderr[-500:]}"
            )

        # Verify output
        if os.path.exists(output_path):
            size_gb = os.path.getsize(output_path) / (1024 ** 3)
            print(f"   Output size: {size_gb:.2f}GB")
            print(f"   Compression: {allocation.get('compression_ratio', 0):.1f}x from BF16")
        else:
            raise RuntimeError(f"Output file not created: {output_path}")

    def generate_script(self, source_model, allocation, output_path="adaptive.gguf",
                        architecture="dense"):
        """
        Generate a shell script that can be run later.

        Useful when you want to profile on one machine and quantize on another.
        """
        level_dist = allocation["level_distribution"]
        base_quant = max(level_dist, key=level_dist.get)
        base_quant_str = self.QUANT_MAP[base_quant]

        lines = [
            "#!/bin/bash",
            "# Generated by adaptive-quant",
            f"# Model: {source_model}",
            f"# Budget: {allocation['budget_gb']}GB",
            f"# Avg bits: {allocation['avg_bits_per_weight']}",
            f"# Compression: {allocation.get('compression_ratio', 0):.1f}x",
            f"# Architecture: {architecture}",
            "",
            "llama-quantize \\",
            f"  {source_model} \\",
            f"  {output_path} \\",
            f"  {base_quant_str} \\",
        ]

        overrides = self._build_overrides(allocation, base_quant, architecture)
        for override in overrides:
            lines.append(f"  --tensor-type {override} \\")

        # Remove trailing backslash from last line
        lines[-1] = lines[-1].rstrip(" \\")

        return "\n".join(lines)
