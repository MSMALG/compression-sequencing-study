# Phase 4 — Fine-Tuning Recovery

## Objective

Evaluate whether the significant accuracy loss introduced by structural pruning can be recovered through post-pruning fine-tuning, and whether the recovered performance is maintained after dynamic quantization.

---

## Method

**Model:** Structurally pruned MobileNetV2 (20% pruning ratio, same pruning configuration as previous phases)

### Fine-tuning Configuration

- **Dataset:** Imagenette2-160 training split (9,469 images)
- **Optimizer:** SGD
- **Learning Rate:** 0.001
- **Momentum:** 0.9
- **Loss Function:** Cross-Entropy Loss
- **Training Epochs:** 3
- **Evaluation:** Accuracy computed using only the 10 Imagenette-relevant ImageNet logits, consistent with the evaluation protocol used in previous phases.

---

## Results

| Stage | Accuracy (%) |
|-------|-------------:|
| Pruned (P), no fine-tune | **8.38** |
| Pruned (P), fine-tuned | **90.83** |
| Fine-tuned P → Q (quantized) | **90.80** |

Training loss steadily decreased throughout fine-tuning:

| Epoch | Average Loss |
|------:|-------------:|
| 1 | 0.7895 |
| 2 | 0.3953 |
| 3 | 0.2876 |

The consistent reduction in training loss indicates successful optimization and stable convergence during the three fine-tuning epochs.

---

## Observations

- Fine-tuning successfully restored model performance after structural pruning, increasing validation accuracy from **8.38%** to **90.83%**.
- Dynamic quantization applied after fine-tuning preserved the recovered performance, with the quantized model achieving **90.80%** accuracy.
- The difference between the fine-tuned model and its quantized counterpart was minimal (90.83% → 90.80%), demonstrating that dynamic quantization introduced virtually no additional degradation after recovery.
- Only three epochs of fine-tuning were required to recover the majority of the lost accuracy, indicating that the recovery process is computationally inexpensive relative to the performance improvement obtained.
- This experiment evaluates the **Prune → Fine-Tune → Quantize (P → FT → Q)** pipeline. Investigating fine-tuning after the **Quantize → Prune (Q → P)** pipeline remains future work.

---

## Preliminary Conclusion

The substantial accuracy degradation caused by structural pruning can be effectively mitigated through a short fine-tuning stage. After recovery, the model retains its performance even after dynamic quantization, indicating that quantization contributes negligible additional accuracy loss when applied to a properly fine-tuned pruned model. These results suggest that incorporating a recovery stage is more important to overall deployment performance than the compression sequence itself.
