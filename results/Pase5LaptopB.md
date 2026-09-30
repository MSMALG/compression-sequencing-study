## Results

| Architecture | Baseline Accuracy (%) | Repaired, No Fine-Tune (%) | Accuracy Drop |
|---|---:|---:|---:|
| MobileNetV2 | 97.35 | 8.05 | ~89 points |
| ResNet-18 | 97.63 | 38.47 | ~59 points |
| VGG16 | 97.30 | 33.35 | ~64 points |
| EfficientNet-B0 | 99.41 | 8.97 | ~90 points |

All repaired models produced the expected output shape of `torch.Size([1, 1000])` and ran successfully using the generalized repair function.

### MobileNetV2

The classifier expected 1280 input features, but 1024 channels actually arrived. The repaired model produced the expected output shape of `torch.Size([1, 1000])`. The resulting accuracy was **8.05%**, compared with **97.35%** for the baseline model.

### ResNet-18

The fully connected layer expected 512 input features, but 409 channels actually arrived. The repaired model produced the expected output shape of `torch.Size([1, 1000])`. The resulting accuracy was **38.47%**, compared with **97.63%** for the baseline model.

### VGG16

The first classifier layer expected 25088 input features, but 20041 channels actually arrived. The last convolutional layer was identified as `features.28`. A total of 409 channels were kept, corresponding to 20041 expanded flattened slots. The repaired model produced the expected output shape of `torch.Size([1, 1000])`. The resulting accuracy was **33.35%**, compared with **97.30%** for the baseline model.

### EfficientNet-B0

The classifier expected 1280 input features, but 1024 channels actually arrived. The last convolutional layer was identified as `features.8.0`. The repaired model produced the expected output shape of `torch.Size([1, 1000])`. The resulting accuracy was **8.97%**, compared with **99.41%** for the baseline model.
