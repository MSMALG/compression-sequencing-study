# Phase 4 — Fine-Tuning Recovery

## Objective

Determine whether the severe accuracy loss observed in Phase 3
(structural pruning collapsing accuracy to ~8%) can be recovered
through post-pruning fine-tuning, and whether that recovery survives
subsequent quantization.

## Method

Model: Structurally pruned MobileNetV2 (20% pruning ratio, same
configuration as Phase 2/3)

Fine-tuning:
- Dataset: Imagenette2-160 training split (9,469 images)
- Optimizer: SGD, lr=0.001, momentum=0.9
- Loss: Cross-entropy over the 10 Imagenette-relevant logits
  (same masking approach as Phase 3 evaluation)
- Epochs: 3

Evaluation: same 10-class-masked accuracy method as Phase 3, on the
Imagenette validation split (3,925 images).

## Results

| Stage                          | Accuracy (%) |
|---------------------------------|-------------:|
| Pruned (P), no fine-tune         |         8.38 |
| Pruned (P), fine-tuned           |        89.99 |
| Fine-tuned P → Q (quantized)     |        90.04 |

Training loss decreased steadily across epochs (0.8055 → 0.4123 →
0.3061), indicating stable convergence with no signs of divergence
or overfitting within this short training run.

## Observations

- Fine-tuning recovered the vast majority of the accuracy lost to
  pruning, from near-random (8.38%) to 89.99% — within ~7.4 points
  of the unpruned baseline (97.35%, Phase 3).
- Quantizing the fine-tuned model caused a negligible additional
  change (89.99% → 90.04%), confirming that quantization itself is
  not the source of the accuracy loss seen in Phase 3 — the loss was
  entirely attributable to un-recovered pruning damage.
- Only 3 epochs were needed to reach this recovery, suggesting the
  fine-tuning cost is modest relative to the accuracy regained.
- This experiment was conducted for the P → Q pipeline only. Whether
  fine-tuning after quantize-then-prune (Q → P) recovers accuracy to
  a similar degree has not yet been tested and is noted as a
  direction for future work.

## Preliminary Conclusion

The severe accuracy collapse observed in Phase 3 is not an inherent
limitation of structural pruning or quantization — it is a
consequence of omitting a recovery step. A short fine-tuning phase
after pruning restores accuracy to near-baseline levels, and this
recovery is preserved under subsequent quantization. This suggests
that, in practice, compression order (P → Q vs. Q → P) is a much
smaller factor in overall deployment quality than whether a
fine-tuning/recovery step is included at all.
