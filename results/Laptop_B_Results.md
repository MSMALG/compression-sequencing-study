# Laptop B Results — Intel i5-1240P + Intel Iris Xe

## Hardware & Software

| Component | Details |
|---|---|
| CPU | Intel Core i5-1240P 1.70 GHz |
| GPU | Intel Iris Xe Graphics (128 MB) |
| RAM | 16.0 GB (15.7 GB usable) |
| OS | Windows 11 |
| Python | 3.11 |
| PyTorch | 2.12.1 (CPU build) |
| GPU Framework | Intel OpenVINO (Experiment 1 only) |
| CUDA | Not available |

---

## Experiment 1 — Baseline FP32

| Device | Latency (seconds) |
|---|---|
| CPU | 0.038689 |
| GPU — Intel Iris Xe (OpenVINO) | 0.008896 |

**Observation:**
The Intel Iris Xe GPU achieved approximately 4.3× lower inference latency than the CPU.
GPU benchmarking was performed using Intel OpenVINO runtime, as CUDA is not available on this hardware configuration.

---

## Experiment 2 — Structured Pruning (20%)

| Device | Latency (seconds) |
|---|---|
| CPU | 0.037095 |

**Model Size Comparison:**

| Model | Size (MB) |
|---|---|
| Baseline FP32 | 13.60 |
| Pruned (20%) | 13.60 |

**Change vs Baseline:**

| Metric | Baseline | Pruned | Change |
|---|---|---|---|
| CPU Latency | 0.038689 s | 0.037095 s | −4.1% |
| Model Size | 13.60 MB | 13.60 MB | 0% |

**Observation:**
A marginal latency reduction was observed on CPU following pruning. No measurable change in model storage size was recorded. As with Laptop A, the pruning implementation introduced weight sparsity without physically altering the network architecture.

---

## Experiment 3 — Quantization Only (Q-Only)

| Device | Latency (seconds) |
|---|---|
| CPU | 0.038947 |

**Model Size Comparison:**

| Model | Size (MB) |
|---|---|
| Baseline FP32 | 13.60 |
| Quantized (INT8) | 9.94 |

**Change vs Baseline:**

| Metric | Baseline | Quantized | Change |
|---|---|---|---|
| CPU Latency | 0.038689 s | 0.038947 s | +0.7% |
| Model Size | 13.60 MB | 9.94 MB | −26.9% |

**Observation:**
Quantization successfully reduced model size by approximately 27%. CPU latency remained nearly unchanged, consistent with the findings from Laptop A.

---

## Experiment 4 — Pruning → Quantization (P → Q)

| Metric | Value |
|---|---|
| CPU Latency | 0.038294 s |
| Model Size | 9.94 MB |

**Comparison with Q-Only:**

| Experiment | CPU Latency |
|---|---|
| Q-Only | 0.038947 s |
| P → Q | 0.038294 s |

**Observation:**
A marginal latency improvement was observed compared to quantization only. The difference is small and would require further trials to confirm statistical significance.

---

## Experiment 5 — Quantization → Pruning (Q → P)

| Metric | Value |
|---|---|
| CPU Latency | 0.040075 s |
| Model Size | 9.94 MB |

**Comparison with P → Q:**

| Experiment | CPU Latency |
|---|---|
| P → Q | 0.038294 s |
| Q → P | 0.040075 s |

**Observation:**
Unlike Laptop A where Q → P showed a slight latency advantage, on Laptop B the P → Q pipeline produced marginally lower latency. This difference in ordering behavior across hardware platforms may be worth investigating further.

---

## Full Summary — Laptop B

| Experiment | CPU Latency (s) | GPU Latency (s) | Model Size (MB) |
|---|---|---|---|
| Baseline FP32 | 0.038689 | 0.008896 (OpenVINO) | 13.60 |
| Structured Pruning (20%) | 0.037095 | N/A | 13.60 |
| Dynamic Quantization | 0.038947 | N/A | 9.94 |
| P → Q | 0.038294 | N/A | 9.94 |
| Q → P | 0.040075 | N/A | 9.94 |

---

## Cross-Platform Comparison — CPU Latency

| Experiment | Laptop A (Intel i7 + RTX 4070) | Laptop B (Intel i5-1240P + Iris Xe) |
|---|---|---|
| Baseline FP32 | 0.053954 s | 0.038689 s |
| Structured Pruning (20%) | 0.066158 s | 0.037095 s |
| Dynamic Quantization | 0.066799 s | 0.038947 s |
| P → Q | 0.067500 s | 0.038294 s |
| Q → P | 0.065440 s | 0.040075 s |

**Note:**
Laptop B consistently achieved lower CPU inference latency across all experiments despite comparable hardware specifications. A likely contributing factor is the difference in PyTorch builds: Laptop A used PyTorch 2.11.0 compiled with CUDA support, while Laptop B used PyTorch 2.12.1 compiled for CPU only. CPU-only builds may carry less framework overhead during CPU inference, contributing to the observed latency difference.
