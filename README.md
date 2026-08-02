# Compression Sequencing Study

**Hardware-Aware Optimization Sequencing in Model Compression: Evaluating Accuracy-Efficiency Trade-offs on Consumer-Grade Hardware**

This repository contains the code, logs, and results for a research project investigating whether the *order* in which pruning and quantization are applied to a neural network (MobileNetV2) affects real-world deployment performance — latency, model size, and accuracy — on consumer-grade hardware.


## Project Structure

- **`docs/`** — Development logs and result write-ups from Laptop A (Intel i7 + NVIDIA RTX 4070 Laptop GPU).
- **`results/`** — Result write-ups from Laptop B (Intel i5-1240P + Intel Iris Xe integrated graphics).
- **`notebooks/`** — Jupyter notebooks used on both machines, organized by phase:
  - `phase1` (root-level notebooks) — Baseline FP32, masking-based pruning, quantization, and both compression orders.
  - `phase2/` — Structural (channel-removal) pruning combined with quantization, including the Quantize→Prune repair pipeline.
  - `phase3/` — Accuracy evaluation on the Imagenette2-160 validation set.
  - `phase4/` — Post-pruning fine-tuning recovery experiments.

## Summary of Findings

1. Masking-based pruning does not physically shrink the network and provides no deployment benefit.
2. Structural pruning achieves genuine compression, but pruning an already-quantized model causes a framework-level dimensional mismatch that requires manual repair.
3. All structurally pruned pipelines, regardless of order, collapse to near-random classification accuracy without a recovery step.
4. A short (3-epoch) fine-tuning stage after pruning recovers accuracy to near-baseline levels on two independent hardware platforms.

