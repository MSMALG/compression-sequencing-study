# Phase 3 — Accuracy Evaluation

---

# Objective

Evaluate Top-1 accuracy retention (10-class Imagenette subset) for:

- Baseline FP32
- Structural Pruning (P)
- Structural Pruning → Quantization (P → Q)
- Structural Quantization → Pruning (Q → P, repaired)

---

# Method

**Dataset:** Imagenette2-160 validation set (3,925 images, 10 classes)

**Note:** MobileNetV2 outputs 1000 ImageNet classes. Predictions were restricted to the 10 logits corresponding to Imagenette's classes using an explicit WordNet ID (WNID) to ImageNet index mapping before evaluation. Reported accuracy is therefore measured on the Imagenette subset rather than the full ImageNet-1000 classification task.

---

# Results

| Experiment | Accuracy (%) |
|------------|-------------:|
| Baseline FP32 | **97.35** |
| Structural Pruning (P) | **8.38** |
| P → Q | **8.18** |
| Q → P (Repaired) | **8.08** |

---

# Observations

- The baseline model achieved **97.35%** accuracy, confirming that the evaluation pipeline and class-index mapping were implemented correctly.
- All compressed models experienced a substantial reduction in accuracy, achieving approximately **8%** Top-1 accuracy on the Imagenette validation set.
- Accuracy differences among **Structural Pruning**, **P → Q**, and **Q → P (Repaired)** were minimal, indicating that compression order had little influence on classification performance under the current experimental settings.
- The significant accuracy degradation is likely due to applying one-shot **20% structural pruning** without any post-pruning fine-tuning or recovery stage.
- These results indicate that while structural pruning successfully reduces model complexity, additional retraining is necessary to preserve predictive performance.

---

# Reframing Phase 2 Conclusions

Phase 2 demonstrated that the compression pipelines produced meaningful reductions in model size and efficient CPU inference. However, the accuracy evaluation presented in this phase shows that these deployment benefits came at the cost of a substantial loss in classification performance.

Consequently, deployment metrics such as model size and inference latency should be considered together with predictive accuracy when evaluating compression strategies.

---

# Preliminary Conclusion

Under the current experimental setup, structural pruning substantially reduced classification accuracy regardless of whether quantization was applied before or after pruning. The small differences observed between **P → Q** and **Q → P (Repaired)** suggest that compression order had minimal impact on accuracy, while the absence of a post-pruning fine-tuning stage was the dominant factor affecting model performance.

These findings indicate that fine-tuning or recovery after structural pruning is necessary before either compression pipeline can be considered suitable for deployment.
