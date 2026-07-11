# Phase 2 — Structural Pruning Experiments

---

# Experiment 1 — Structural Pruning (P-Only)

## Objective

Evaluate whether physically removing channels from the network improves deployment efficiency compared to masking-based pruning performed in Phase 1.

Unlike the previous implementation, structural pruning reconstructs the neural network by permanently removing low-importance channels rather than replacing their weights with zeros.

---

## Method

**Library:** Torch-Pruning

**Importance Metric:** L2 Magnitude (`MagnitudeImportance`, p=2)

**Pruning Ratio:** 20%

**Model:** MobileNetV2 (Pretrained)

---

## Results

| Metric      |          Value |
| ----------- | -------------: |
| Parameters  |      2,460,140 |
| Model Size  |        9.59 MB |
| CPU Latency | **0.037731 s** |

### Comparison with Phase 1 (Masking Pruning)

| Metric      | Phase 1 (Masking) | Phase 2 (Structural) |
| ----------- | ----------------: | -------------------: |
| Parameters  |     ~3.50 Million |         2.46 Million |
| Model Size  |          13.60 MB |              9.59 MB |
| CPU Latency |        0.066158 s |       **0.037731 s** |

### Comparison with Baseline FP32

| Metric      |   Baseline |     Structural |
| ----------- | ---------: | -------------: |
| Model Size  |   13.60 MB |        9.59 MB |
| CPU Latency | 0.053954 s | **0.037731 s** |

---

## Observations

* Structural pruning physically reduced the size of the neural network.
* Approximately one million parameters were removed.
* Model storage decreased from **13.60 MB** to **9.59 MB**.
* CPU inference latency improved from **0.053954 s** to **0.037731 s**.
* Structural pruning substantially outperformed masking-based pruning in terms of deployment efficiency.

---

## Preliminary Conclusion

Unlike masking-based pruning, structural pruning achieved genuine model compression while improving inference latency relative to the original FP32 model.

These findings suggest that deployment efficiency depends not only on the compression technique itself but also on how the compression is implemented.

---

# Experiment 2 — Structural Pruning → Quantization (P → Q)

## Objective

Evaluate whether applying dynamic quantization after structural pruning provides additional deployment benefits beyond structural pruning alone.

This experiment investigates whether reducing numerical precision after physically shrinking the network can further improve storage efficiency while maintaining inference performance.

---

## Method

**Pipeline:**

Structural Pruning → Dynamic Quantization

**Pruning Library:** Torch-Pruning

**Quantization Method:** PyTorch Dynamic Quantization (INT8)

**Model:** Structurally Pruned MobileNetV2

---

## Results

| Metric      |          Value |
| ----------- | -------------: |
| Model Size  |        6.67 MB |
| CPU Latency | **0.038161 s** |

### Comparison with Structural Pruning Only

| Metric      |   Structural P | Structural P → Q |
| ----------- | -------------: | ---------------: |
| Model Size  |        9.59 MB |          6.67 MB |
| CPU Latency | **0.037731 s** |   **0.038161 s** |

### Comparison with Baseline FP32

| Metric      |   Baseline | Structural P → Q |
| ----------- | ---------: | ---------------: |
| Model Size  |   13.60 MB |          6.67 MB |
| CPU Latency | 0.053954 s |   **0.038161 s** |

---

## Observations

* Dynamic quantization further reduced model storage after structural pruning.
* Total model size decreased by approximately **51%** relative to the original FP32 baseline.
* CPU latency remained close to the structurally pruned model while remaining below the FP32 baseline.
* The combination of structural pruning and quantization produced the highest compression ratio observed in the study.
* Unlike Phase 1, compression gains translated into genuine deployment benefits.

---

## Preliminary Conclusion

Structural Pruning followed by Quantization (**P → Q**) produced the strongest compression outcome observed in the study, achieving substantial model size reduction while maintaining efficient CPU inference.

The results suggest that physically reducing network structure before reducing numerical precision is an effective deployment strategy for consumer hardware.

---

# Experiment 3 — Structural Quantization → Pruning (Q → P)

## Objective

Evaluate whether reversing the compression order influences deployment behavior when structural pruning is used.

This experiment directly investigates the primary research question regarding compression sequencing.

---

## Method

**Pipeline:**

Dynamic Quantization → Structural Pruning

**Quantization Method:** PyTorch Dynamic Quantization (INT8)

**Pruning Library:** Torch-Pruning

**Model:** MobileNetV2 (Pretrained)

---

## Initial Outcome

Structural pruning successfully reduced the size of the quantized network.

However, inference failed immediately after pruning.

Investigation revealed that structural pruning reduced the backbone output dimensionality from **1280 channels** to **1024 channels**, while the `DynamicQuantizedLinear` classifier retained its original **1280-input** configuration.

This created a dimensional mismatch that prevented inference execution.

---

## Framework Compatibility Finding

The default combination of PyTorch Dynamic Quantization and Torch-Pruning did not automatically reconstruct the quantized classifier after structural channel removal.

As a result, the native **Q → P** pipeline was not directly deployable.

A manual classifier reconstruction step was required to restore inference functionality.

---

## Results (After Classifier Reconstruction)

| Metric      |          Value |
| ----------- | -------------: |
| Parameters  |      1,435,140 |
| Model Size  |        9.60 MB |
| CPU Latency | **0.034752 s** |

### Comparison with Structural P → Q

| Metric      |          P → Q | Q → P (Repaired) |
| ----------- | -------------: | ---------------: |
| Model Size  |        6.67 MB |          9.60 MB |
| CPU Latency | **0.038161 s** |   **0.034752 s** |

---

## Observations

* Structural pruning successfully removed channels from the quantized model.
* The native **Q → P** pipeline could not be executed without additional intervention.
* Manual classifier reconstruction restored inference functionality.
* Even after repair, the resulting model remained larger than the **P → Q** pipeline.
* CPU inference latency remained close to the original FP32 baseline.
* Compression order influenced both deployment performance and implementation feasibility.

---

## Preliminary Conclusion

Unlike Structural **P → Q**, the Structural **Q → P** pipeline encountered framework-level compatibility issues between Dynamic Quantization and structural channel removal.

Although inference could be restored through manual reconstruction, the resulting model remained larger than the **P → Q** pipeline despite achieving comparable inference performance.

These findings suggest that compression order affects not only performance metrics but also practical deployability within existing deep learning toolchains.
