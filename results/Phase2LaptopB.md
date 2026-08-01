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

| Metric | Value |
|---------|------:|
| Parameters | 2,460,140 |
| Model Size | 9.59 MB |
| CPU Latency | **0.036048 s** |

### Comparison with Phase 1 (Masking Pruning)

| Metric | Phase 1 (Masking) | Phase 2 (Structural) |
|---------|------------------:|---------------------:|
| Parameters | ~3.50 Million | 2.46 Million |
| Model Size | 13.60 MB | 9.59 MB |
| CPU Latency | 0.066158 s | **0.036048 s** |

### Comparison with Baseline FP32

| Metric | Baseline | Structural |
|---------|---------:|-----------:|
| Model Size | 13.60 MB | 9.59 MB |
| CPU Latency | 0.053954 s | **0.036048 s** |

---

## Observations

- Structural pruning physically reduced the size of the neural network.
- Approximately one million parameters were removed.
- Model storage decreased from **13.60 MB** to **9.59 MB**.
- CPU inference latency was measured at **0.036048 s**.
- Structural pruning substantially outperformed masking-based pruning in terms of deployment efficiency.

---

## Preliminary Conclusion

Unlike masking-based pruning, structural pruning achieved genuine model compression while maintaining efficient CPU inference.

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

| Metric | Value |
|---------|------:|
| Model Size | 6.67 MB |
| CPU Latency | **0.034869 s** |

### Comparison with Structural Pruning Only

| Metric | Structural P | Structural P → Q |
|---------|-------------:|-----------------:|
| Model Size | 9.59 MB | 6.67 MB |
| CPU Latency | **0.036048 s** | **0.034869 s** |

### Comparison with Baseline FP32

| Metric | Baseline | Structural P → Q |
|---------|---------:|-----------------:|
| Model Size | 13.60 MB | 6.67 MB |
| CPU Latency | 0.053954 s | **0.034869 s** |

---

## Observations

- Dynamic quantization further reduced model storage after structural pruning.
- Total model size decreased by approximately **51%** relative to the original FP32 baseline.
- CPU latency remained close to the structurally pruned model.
- The combination of structural pruning and quantization produced the highest compression ratio observed in this phase.
- Unlike Phase 1, compression gains translated into genuine deployment benefits.

---

## Preliminary Conclusion

Structural Pruning followed by Quantization (**P → Q**) produced the strongest compression outcome observed in this phase, achieving substantial model compression while maintaining efficient CPU inference.

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

## Repair Process

A repair procedure was performed to preserve both the learned classifier weights and dynamic quantization.

The original quantized classifier weights were traced back to the corresponding floating-point representation, and the surviving **1024** input channels identified by the pruning process were retained. The classifier weights were then sliced to match the reduced backbone output dimensionality before reconstructing the `DynamicQuantizedLinear` layer with **1024** input features.

This repair preserved the trained classifier weights while maintaining dynamic quantization, allowing inference to execute successfully without replacing the quantized classifier with an unquantized linear layer.

---

## Results (After Repair)

| Metric | Value |
|---------|------:|
| Parameters | 1,435,140 |
| Model Size | 6.66 MB |
| CPU Latency | **0.038106 s** |

### Comparison with Structural P → Q

| Metric | P → Q | Q → P (Repaired) |
|---------|------:|-----------------:|
| Model Size | 6.67 MB | 6.66 MB |
| CPU Latency | **0.034869 s** | **0.038106 s** |

---

## Observations

- Structural pruning successfully removed channels from the quantized model.
- The native **Q → P** pipeline could not be executed without additional intervention.
- The repair preserved both the trained classifier weights and dynamic quantization.
- After repair, inference executed successfully using a `DynamicQuantizedLinear` classifier with **1024** input features.
- The repaired model maintained a compact storage size of **6.66 MB**.
- Compression order influenced both implementation complexity and deployment behavior.

---

## Preliminary Conclusion

Unlike Structural **P → Q**, the Structural **Q → P** pipeline required additional repair to resolve the dimensional mismatch introduced by structural pruning.

After repairing the classifier while preserving dynamic quantization, inference was successfully restored, allowing a fair evaluation of the **Q → P** pipeline.

These findings suggest that compression order affects not only performance metrics but also the practical implementation required to deploy compressed neural networks.
