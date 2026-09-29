## Results

| Architecture | Baseline Accuracy (%) | Repaired, No Fine-Tune (%) | Accuracy Drop |
|---|---:|---:|---:|
| MobileNetV2 | — | 8.05 | — |
| ResNet-18 | 97.63 | 38.47 | ~59 points |
| VGG16 | 97.30 | 33.35 | ~64 points |
| EfficientNet-B0 | 99.41 | 8.97 | ~90 points |

All repaired models produced the expected output shape of `torch.Size([1, 1000])` and ran successfully using the generalized repair function.
