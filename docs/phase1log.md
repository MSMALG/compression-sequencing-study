# Experiment 1 — Baseline Benchmark

## Objective
Establish a baseline benchmark before applying model compression techniques such as pruning and quantization.

## Model
The experiments currently use MobileNetV2, a lightweight convolutional neural network primarily designed for image classification and edge-device deployment.

Although MobileNetV2 is an image classification model, the research itself is not limited to image classification tasks. The study focuses on general model compression and deployment behavior, specifically how pruning and quantization sequencing affects latency, efficiency, and hardware performance. These principles can extend to other neural network architectures and AI domains.

MobileNetV2 was selected because:
- it is lightweight and efficient,
- widely used in edge AI research,
- suitable for consumer-grade hardware,
- and practical for rapid experimentation and benchmarking.

## Experimental Goal
Measure inference latency on CPU and GPU before any compression is applied.

## Environment
Hardware:
- Intel Core i7 CPU
- NVIDIA RTX 4070 Laptop GPU

Software:
- Python 3.11
- PyTorch 2.11.0 + CUDA 12.6
- JupyterLab

## CUDA
CUDA is NVIDIA’s parallel computing platform that allows PyTorch to execute tensor operations on the GPU instead of the CPU, significantly accelerating neural network inference and computation.

## Benchmark Setup
- Model: MobileNetV2 (pretrained)
- Input tensor shape: 1×3×224×224
- Benchmark type: Average inference latency over multiple runs
- Devices tested:
  - CPU
  - GPU (CUDA)

## Results

| Device | Latency (seconds) |
|--------|------------------|
| CPU | 0.053954 |
| GPU | 0.018986 |

## Observation
The GPU achieved approximately 2.8× lower inference latency than the CPU during baseline inference.

These measurements establish the reference point for future compression experiments.

---

# Experiment 2 — Structured Pruning Benchmark (20%)

## Objective

Evaluate the impact of structured pruning on inference latency and model size before introducing quantization.

## Model

* MobileNetV2 (pretrained)
* Structured pruning applied to convolutional layers

## Experimental Goal

Determine whether introducing structured sparsity improves deployment efficiency on consumer-grade hardware.

## Environment

### Hardware

* Intel Core i7 CPU
* NVIDIA RTX 4070 Laptop GPU

### Software

* Python 3.11
* PyTorch 2.11.0 + CUDA 12.6
* JupyterLab

## Pruning Configuration

Structured pruning was applied to all convolutional layers using PyTorch's pruning utilities.

Parameters:

* Pruning method: Structured L2-norm pruning
* Pruning amount: 20%
* Dimension pruned: Output channels (dim=0)

After pruning, the pruning masks were removed using `prune.remove()` to make the modifications permanent within the model weights.

## Benchmark Setup

* Model: MobileNetV2 (20% structured pruning)
* Input tensor shape: 1×3×224×224
* Benchmark type: Average inference latency over 100 runs
* Devices tested:

  * CPU
  * GPU (CUDA)

## Results

| Device | Latency (seconds) |
| ------ | ----------------: |
| CPU    |          0.066158 |
| GPU    |          0.021111 |

### Model Size Comparison

| Model         | Size (MB) |
| ------------- | --------: |
| Baseline FP32 |     13.60 |
| Pruned (20%)  |     13.60 |

## Observations

Compared with the baseline model:

| Device | Baseline (s) | Pruned (s) | Change |
| ------ | -----------: | ---------: | -----: |
| CPU    |     0.053954 |   0.066158 | +22.6% |
| GPU    |     0.018986 |   0.021111 | +11.2% |

The pruned model exhibited higher inference latency on both CPU and GPU despite the introduction of sparsity.

Additionally, no measurable reduction in model storage size was observed. Both the baseline and pruned models occupied approximately 13.60 MB.

These results indicate that the applied pruning strategy introduced sparsity within the model weights but did not physically reduce the underlying network architecture. Consequently, the theoretical reduction in parameter utilization did not translate into practical runtime acceleration or storage savings.

## Preliminary Conclusion

This experiment highlights an important distinction between parameter sparsity and deployment efficiency. Introducing structured sparsity alone may not guarantee improvements in latency or model size on consumer hardware when the architecture itself remains unchanged.

The findings support the motivation of this research, namely that theoretical compression gains do not necessarily produce real-world deployment benefits.

---

# Experiment 3 — Quantization Only Benchmark (Q-Only)

## Objective

Evaluate the impact of post-training quantization on model size and inference latency before combining quantization with pruning.

## Model

* MobileNetV2 (pretrained)
* Dynamic INT8 quantization applied to supported linear layers

## Experimental Goal

Determine whether reducing numerical precision improves deployment efficiency on consumer-grade hardware.

## Environment

### Hardware

* Intel Core i7 CPU
* NVIDIA RTX 4070 Laptop GPU

### Software

* Python 3.11
* PyTorch 2.11.0 + CUDA 12.6
* JupyterLab

## Quantization Configuration

Dynamic post-training quantization was applied using PyTorch's built-in quantization utilities.

Parameters:

* Quantization method: Dynamic Quantization
* Target data type: INT8 (`torch.qint8`)
* Quantized layers: `torch.nn.Linear`

The quantized model was generated without additional training or fine-tuning.

## Benchmark Setup

* Model: MobileNetV2 (Dynamic INT8 Quantization)
* Input tensor shape: 1×3×224×224
* Benchmark type: Average inference latency over 100 runs
* Device tested:

  * CPU

GPU benchmarking was not performed because dynamic quantization is primarily intended for CPU execution and is not generally supported for CUDA inference in this configuration.

## Results

### Inference Latency

| Device | Latency (seconds) |
| ------ | ----------------: |
| CPU    |          0.066799 |

### Model Size Comparison

| Model         | Size (MB) |
| ------------- | --------: |
| Baseline FP32 |     13.60 |
| Quantized     |      9.94 |

## Observations

Compared with the baseline model:

| Metric      |   Baseline |  Quantized | Change |
| ----------- | ---------: | ---------: | -----: |
| CPU Latency | 0.053954 s | 0.066799 s | +23.8% |
| Model Size  |   13.60 MB |    9.94 MB | -26.9% |

The quantized model achieved a noticeable reduction in storage size, decreasing from 13.60 MB to 9.94 MB.

However, CPU inference latency increased from 0.053954 seconds to 0.066799 seconds. Therefore, while quantization improved storage efficiency, it did not improve runtime performance in this experimental configuration.

A likely explanation is that MobileNetV2 is primarily composed of convolutional layers, while the applied dynamic quantization method targets linear layers. As a result, only a relatively small portion of the network benefited from quantization.

## Preliminary Conclusion

The results demonstrate that compression benefits depend on the chosen metric. Dynamic quantization successfully reduced model size but did not reduce inference latency.

This finding further supports the central motivation of the research: improvements in theoretical compression metrics do not necessarily translate into improved deployment performance on consumer hardware.

---

# Current Experimental Summary

| Experiment               | CPU Latency (s) | GPU Latency (s) | Model Size (MB) |
| ------------------------ | --------------: | --------------: | --------------: |
| Baseline FP32            |        0.053954 |        0.018986 |           13.60 |
| Structured Pruning (20%) |        0.066158 |        0.021111 |           13.60 |
| Dynamic Quantization     |        0.066799 |             N/A |            9.94 |

# Experiment 4 — Pruning → Quantization (P → Q)

## Objective

Evaluate the effect of applying quantization after structured pruning and determine whether compression order influences deployment performance.

## Compression Pipeline

```text
MobileNetV2
    ↓
20% Structured Pruning
    ↓
INT8 Dynamic Quantization
    ↓
Benchmarking
```

## Results

| Metric      |      Value |
| ----------- | ---------: |
| CPU Latency | 0.067500 s |
| Model Size  |    9.94 MB |

## Observation

Applying quantization after pruning produced a model size identical to the quantization-only experiment.

Compared with Quantization Only:

| Experiment | CPU Latency |
| ---------- | ----------: |
| Q-Only     |  0.066799 s |
| P → Q      |  0.067500 s |

The additional pruning stage did not produce measurable deployment benefits in this implementation.

---

# Experiment 5 — Quantization → Pruning (Q → P)

## Objective

Evaluate the reverse compression sequence and compare its behavior against the P → Q pipeline.

## Compression Pipeline

```text
MobileNetV2
    ↓
INT8 Dynamic Quantization
    ↓
20% Structured Pruning
    ↓
Benchmarking
```

## Results

| Metric      |      Value |
| ----------- | ---------: |
| CPU Latency | 0.065440 s |
| Model Size  |    9.94 MB |

## Observation

The Q → P pipeline achieved the same final model size as the P → Q pipeline but produced slightly lower CPU latency.

Comparison:

| Experiment | CPU Latency |
| ---------- | ----------: |
| P → Q      |  0.067500 s |
| Q → P      |  0.065440 s |

A small latency advantage was observed for Q → P, although additional trials would be required to determine whether the difference is statistically significant.

---

# Experimental Summary

| Experiment               | CPU Latency (s) | GPU Latency (s) | Model Size (MB) |
| ------------------------ | --------------: | --------------: | --------------: |
| Baseline FP32            |        0.053954 |        0.018986 |           13.60 |
| Structured Pruning (20%) |        0.066158 |        0.021111 |           13.60 |
| Dynamic Quantization     |        0.066799 |             N/A |            9.94 |
| P → Q                    |        0.067500 |             N/A |            9.94 |
| Q → P                    |        0.065440 |             N/A |            9.94 |

# Preliminary Findings

The first experimental cycle suggests that compression gains do not necessarily translate into deployment gains.

Although quantization reduced storage requirements, none of the tested compression pipelines improved inference latency relative to the baseline model.

These observations provide early evidence that compression efficiency and deployment efficiency may represent distinct optimization objectives on consumer-grade hardware.

Further investigation using true channel-removal pruning is planned in a second experimental phase.
