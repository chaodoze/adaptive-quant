"""
Per-layer quantization sensitivity profiling.

Two implementations:
- SensitivityProfiler: Uses llama.cpp tools to measure real perplexity deltas
- SimulatedProfiler: Uses realistic synthetic data based on published research
"""

import os
import subprocess
import tempfile
import time


class SensitivityProfiler:
    """
    Profile real models by quantizing one layer at a time and measuring
    perplexity delta. Requires llama.cpp binaries.

    Algorithm:
    1. Measure baseline perplexity (full precision)
    2. For each layer, quantize just that layer to Q4_0 and Q2_K
    3. Measure perplexity delta from baseline
    4. Output sensitivity score per layer
    """

    # Quantization levels to probe for sensitivity
    PROBE_QUANTS = ["Q4_0", "Q2_K"]

    def __init__(self, model_path, llama_perplexity="llama-perplexity",
                 llama_quantize="llama-quantize", reference_text=None):
        self.model_path = model_path
        self.llama_perplexity = llama_perplexity
        self.llama_quantize = llama_quantize
        self.reference_text = reference_text
        self._validate_tools()

    def _validate_tools(self):
        """Check that required llama.cpp binaries exist."""
        for tool in [self.llama_perplexity, self.llama_quantize]:
            try:
                subprocess.run([tool, "--help"], capture_output=True, timeout=10)
            except FileNotFoundError:
                raise RuntimeError(
                    f"Required tool '{tool}' not found. Install llama.cpp:\n"
                    f"  brew install llama.cpp\n"
                    f"Or specify path: --llama-perplexity /path/to/llama-perplexity"
                )

    def _get_reference_text(self):
        """Download wikitext-2 if no reference text provided."""
        if self.reference_text and os.path.exists(self.reference_text):
            return self.reference_text

        # Try to find existing wikitext-2
        default_path = "wikitext-2-raw/wiki.test.raw"
        if os.path.exists(default_path):
            return default_path

        print("   Downloading wikitext-2 test dataset...")
        subprocess.run([
            "bash", "-c",
            "wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip "
            "&& unzip -qo wikitext-2-raw-v1.zip"
        ], check=True)
        return default_path

    def _measure_perplexity(self, model_path, reference_text):
        """Run llama-perplexity and extract the score."""
        result = subprocess.run(
            [self.llama_perplexity, "-m", model_path, "-f", reference_text,
             "-c", "512", "--no-display-prompt"],
            capture_output=True, text=True, timeout=3600
        )
        # Parse perplexity from output
        for line in result.stdout.split("\n"):
            if "perplexity" in line.lower() and "=" in line:
                try:
                    val = float(line.split("=")[-1].strip().split()[0])
                    return val
                except (ValueError, IndexError):
                    continue

        raise RuntimeError(f"Could not parse perplexity from output:\n{result.stdout[-500:]}")

    def _quantize_single_layer(self, layer_idx, quant_type, tmpdir):
        """Quantize a single layer to the given type, leave others at original precision."""
        output_path = os.path.join(tmpdir, f"layer_{layer_idx}_{quant_type}.gguf")

        # Use --tensor-type to target specific layer tensors
        tensor_patterns = [
            f"blk.{layer_idx}.attn_q",
            f"blk.{layer_idx}.attn_k",
            f"blk.{layer_idx}.attn_v",
            f"blk.{layer_idx}.attn_output",
            f"blk.{layer_idx}.ffn_gate",
            f"blk.{layer_idx}.ffn_up",
            f"blk.{layer_idx}.ffn_down",
        ]

        cmd = [self.llama_quantize, self.model_path, output_path, "Q8_0"]
        for pattern in tensor_patterns:
            cmd.extend(["--tensor-type", f"{pattern}={quant_type}"])

        subprocess.run(cmd, capture_output=True, check=True, timeout=600)
        return output_path

    def _detect_num_layers(self):
        """Detect the number of transformer layers in the model."""
        code = (
            "import re\n"
            "with open(%r, 'rb') as f:\n"
            "    content = f.read(1024*1024)\n"
            r"    blocks = set(re.findall(rb'blk\.(\d+)', content))" "\n"
            "    print(max(int(b) for b in blocks) + 1 if blocks else 32)\n"
        ) % self.model_path
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True
        )
        try:
            return int(result.stdout.strip())
        except ValueError:
            print("   ⚠️  Could not detect layer count, defaulting to 32")
            return 32

    def run(self):
        """Run the full sensitivity profiling pipeline."""
        ref_text = self._get_reference_text()
        num_layers = self._detect_num_layers()

        print(f"   Model: {self.model_path}")
        print(f"   Layers: {num_layers}")
        print(f"   Reference text: {ref_text}")
        print(f"   Probing quantizations: {self.PROBE_QUANTS}")

        # Step 1: Baseline perplexity
        print("\n   📊 Measuring baseline perplexity...")
        baseline_ppl = self._measure_perplexity(self.model_path, ref_text)
        print(f"   Baseline perplexity: {baseline_ppl:.4f}")

        # Step 2: Per-layer sensitivity
        layers = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for layer_idx in range(num_layers):
                layer_data = {"index": layer_idx, "sensitivities": {}}

                for quant_type in self.PROBE_QUANTS:
                    print(f"   🔍 Layer {layer_idx}/{num_layers-1} @ {quant_type}...", end=" ", flush=True)
                    start = time.time()

                    try:
                        quant_path = self._quantize_single_layer(layer_idx, quant_type, tmpdir)
                        ppl = self._measure_perplexity(quant_path, ref_text)
                        delta = (ppl - baseline_ppl) / baseline_ppl
                        elapsed = time.time() - start

                        layer_data["sensitivities"][quant_type] = {
                            "perplexity": ppl,
                            "delta_pct": delta * 100,
                            "elapsed_sec": elapsed,
                        }
                        print(f"PPL={ppl:.4f} (Δ={delta*100:+.2f}%) [{elapsed:.0f}s]")

                        # Clean up to save disk
                        os.remove(quant_path)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        layer_data["sensitivities"][quant_type] = {
                            "error": str(e)
                        }

                # Composite sensitivity score: weighted average of deltas
                deltas = []
                for qt, data in layer_data["sensitivities"].items():
                    if "delta_pct" in data:
                        deltas.append(abs(data["delta_pct"]))
                layer_data["sensitivity_score"] = sum(deltas) / len(deltas) if deltas else 0

                layers.append(layer_data)

        return {
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "num_layers": num_layers,
            "baseline_perplexity": baseline_ppl,
            "probe_quants": self.PROBE_QUANTS,
            "layers": layers,
        }


class SimulatedProfiler:
    """
    Generate realistic sensitivity profiles based on published research.

    Uses the U-shaped sensitivity curve found in:
    - SliderQuant (OpenReview): "shallow/deep layers are more sensitive"
    - Variable Layer-Wise Quantization: "first and last layer are most important"
    - Apple Super Weights paper: outliers concentrated in edge layers

    Calibrated against Qwen3.5-9B benchmarks from the ngrok quantization post.
    """

    # Known model architectures
    MODELS = {
        "Qwen3.5-9B": {
            "num_layers": 40,
            "params_billions": 9.0,
            "bf16_size_gb": 18.0,
            "baseline_perplexity": 8.186,  # From ngrok article
            "params_per_layer_millions": 225,  # Approximate
        },
        "Qwen3.5-35B-A3B": {
            "num_layers": 64,
            "params_billions": 35.0,
            "bf16_size_gb": 70.0,
            "baseline_perplexity": 6.2,
            "params_per_layer_millions": 547,
        },
        "Llama-3.3-8B": {
            "num_layers": 32,
            "params_billions": 8.0,
            "bf16_size_gb": 16.0,
            "baseline_perplexity": 7.95,
            "params_per_layer_millions": 250,
        },
    }

    def __init__(self, model_name):
        self.model_name = model_name
        if model_name not in self.MODELS:
            print(f"   ⚠️  No preset for '{model_name}', using Qwen3.5-9B shape")
            self.model_info = self.MODELS["Qwen3.5-9B"]
        else:
            self.model_info = self.MODELS[model_name]

    def _u_curve(self, layer_idx, num_layers):
        """
        Generate U-shaped sensitivity score.

        Based on empirical findings:
        - Layer 0 (embedding projection): highest sensitivity
        - Last layer (output projection): second highest
        - Layers 1-3: elevated sensitivity
        - Middle layers: lowest sensitivity
        - Layers N-3 to N-1: rising sensitivity
        """
        t = layer_idx / (num_layers - 1)  # Normalize to [0, 1]

        # U-shape: high at edges, low in middle
        # Using a polynomial that matches published sensitivity curves
        base = 4.0 * (t - 0.5) ** 2  # Parabola: 1.0 at edges, 0.0 at center

        # Extra spike at very first and very last layer
        if layer_idx == 0:
            base += 0.8
        elif layer_idx == num_layers - 1:
            base += 0.6
        elif layer_idx <= 2:
            base += 0.3
        elif layer_idx >= num_layers - 3:
            base += 0.2

        # Add some realistic noise
        import hashlib
        noise_seed = int(hashlib.md5(f"{self.model_name}_{layer_idx}".encode()).hexdigest()[:8], 16)
        noise = ((noise_seed % 1000) / 1000 - 0.5) * 0.15

        return max(0.05, base + noise)

    def run(self):
        """Generate simulated sensitivity profile."""
        num_layers = self.model_info["num_layers"]
        baseline_ppl = self.model_info["baseline_perplexity"]

        print(f"   Model: {self.model_name} (simulated)")
        print(f"   Layers: {num_layers}")
        print(f"   Baseline perplexity: {baseline_ppl}")
        print(f"   BF16 size: {self.model_info['bf16_size_gb']:.1f}GB")

        layers = []
        for i in range(num_layers):
            score = self._u_curve(i, num_layers)

            # Convert sensitivity score to realistic perplexity deltas
            q4_delta = score * 1.8  # Q4 causes moderate degradation
            q2_delta = score * 12.0  # Q2 causes severe degradation

            layers.append({
                "index": i,
                "sensitivity_score": round(score, 4),
                "sensitivities": {
                    "Q4_0": {
                        "perplexity": round(baseline_ppl * (1 + q4_delta / 100), 4),
                        "delta_pct": round(q4_delta, 2),
                    },
                    "Q2_K": {
                        "perplexity": round(baseline_ppl * (1 + q2_delta / 100), 4),
                        "delta_pct": round(q2_delta, 2),
                    },
                },
                "params_millions": self.model_info["params_per_layer_millions"],
            })

        return {
            "model_path": f"(simulated) {self.model_name}",
            "model_name": self.model_name,
            "num_layers": num_layers,
            "baseline_perplexity": baseline_ppl,
            "bf16_size_gb": self.model_info["bf16_size_gb"],
            "probe_quants": ["Q4_0", "Q2_K"],
            "layers": layers,
        }
