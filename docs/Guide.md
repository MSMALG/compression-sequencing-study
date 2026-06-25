# Compression Sequencing Study

## Goal

This project investigates whether the order of model compression techniques affects deployment performance on consumer hardware.

The experiments use MobileNetV2 and evaluate:

1. Baseline FP32
2. Structured Pruning
3. Quantization
4. Pruning → Quantization (P→Q)
5. Quantization → Pruning (Q→P)

Metrics:

* CPU latency
* GPU latency (if available)
* Model size

---

# Setup

## Install Anaconda

Download and install Anaconda.

Open Anaconda Prompt.

Create the environment:

```bash
conda create -n compression-thesis python=3.11
```

Type:

```text
y
```

Activate:

```bash
conda activate compression-thesis
```

Install packages:

```bash
pip install torch torchvision pandas jupyterlab
```

Launch JupyterLab:

```bash
jupyter lab
```

When selecting a kernel choose:

```text
Python [conda env:compression-thesis]
```

---

# Hardware Information

Before running experiments record:

* CPU model
* GPU model
* RAM size
* PyTorch version

Run:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

---

# Running Experiments

Run the notebooks in numerical order:

1. Experiment_1_Baseline.ipynb
2. Experiment_2_Pruning.ipynb
3. Experiment_3_Quantization.ipynb
4. Experiment_4_Pruning_Then_Quantization.ipynb
5. Experiment_5_Quantization_Then_Pruning.ipynb

Do not modify the code.

Record all outputs.

---

# Results Template

| Experiment   | CPU Latency | Model Size |
| ------------ | ----------- | ---------- |
| Baseline     |             |            |
| Pruning      |             |            |
| Quantization |             |            |
| P→Q          |             |            |
| Q→P          |             |            |

Also provide:

* CPU model
* GPU model
* RAM
* PyTorch version

---

# Deliverables

After finishing all experiments:

1. Upload results to the repository.
2. Send completed results table.
3. Include any errors or observations encountered during execution.
