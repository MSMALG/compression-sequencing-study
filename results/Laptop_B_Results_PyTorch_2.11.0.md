# Laptop B Results — Intel i5-1240P + Intel Iris Xe
## PyTorch 2.11.0 (CPU Build)

## Hardware & Software

| Component | Details |
|---|---|
| CPU | Intel Core i5-1240P 1.70 GHz |
| GPU | Intel Iris Xe Graphics (128 MB) |
| RAM | 16.0 GB (15.7 GB usable) |
| OS | Windows 11 |
| Python | 3.11 |
| PyTorch | 2.11.0 (CPU build) |
| Torchvision | 0.26.0 (CPU build) |
| Torchaudio | 2.11.0 (CPU build) |
| CUDA | Not available |

---

## Experiment 1 — Baseline FP32

| Device | Latency (seconds) |
|---|---|
| CPU | 0.087339 |

**Observation:**
This establishes the baseline inference latency for Laptop B before any compression is applied. CUDA is not available on this hardware configuration.

---

## Experiment 2 — Structured Pruning (20%)

| Device | Latency (seconds) |
|---|---|
| CPU | 0.083956 |

**Model Size Comparison:**

| Model | Size (MB) |
|---|---|
| Baseline FP32 | 13.60 |
| Pruned (20%) | 13.60 |

**Change vs Baseline:**

| Metric | Baseline | Pruned | Change |
|---|---|---|---|
| CPU Latency | 0.087339 s | 0.083956 s | −3.9% |
| Model Size | 13.60 MB | 13.60 MB | 0% |

**Observation:**
A marginal latency reduction was observed on CPU following pruning. No measurable change in model storage size was recorded. The pruning implementation introduced weight sparsity without physically altering the network architecture.

---

## Experiment 3 — Quantization Only (Q-Only)

| Device | Latency (seconds) |
|---|---|
| CPU | 0.091211 |

**Model Size Comparison:**

| Model | Size (MB) |
|---|---|
| Baseline FP32 | 13.60 |
| Quantized (INT8) | 9.94 |

**Change vs Baseline:**

| Metric | Baseline | Quantized | Change |
|---|---|---|---|
| CPU Latency | 0.087339 s | 0.091211 s | +4.4% |
| Model Size | 13.60 MB | 9.94 MB | −26.9% |

**Observation:**
Quantization successfully reduced model size by approximately 27%. However, CPU inference latency increased slightly, consistent with findings from Laptop A.

---

## Experiment 4 — Pruning → Quantization (P → Q)

| Metric | Value |
|---|---|
| CPU Latency | 0.088384 s |
| Model Size | 9.94 MB |

**Comparison with Q-Only:**

| Experiment | CPU Latency |
|---|---|
| Q-Only | 0.091211 s |
| P → Q | 0.088384 s |

**Observation:**
The P → Q pipeline produced a marginal latency improvement compared to quantization only. The difference is small and would require further trials to confirm statistical significance.

---

## Experiment 5 — Quantization → Pruning (Q → P)

| Metric | Value |
|---|---|
| CPU Latency | 0.094388 s |
| Model Size | 9.94 MB |

**Comparison with P → Q:**

| Experiment | CPU Latency |
|---|---|
| P → Q | 0.088384 s |
| Q → P | 0.094388 s |

**Observation:**
The Q → P pipeline produced higher latency than P → Q on Laptop B, suggesting that applying pruning after quantization may introduce additional overhead in this hardware and software configuration.

---

## Full Summary — Laptop B (PyTorch 2.11.0)

| Experiment | CPU Latency (s) | Model Size (MB) |
|---|---|---|
| Baseline FP32 | 0.087339 | 13.60 |
| Structured Pruning (20%) | 0.083956 | 13.60 |
| Dynamic Quantization | 0.091211 | 9.94 |
| P → Q | 0.088384 | 9.94 |
| Q → P | 0.094388 | 9.94 |
