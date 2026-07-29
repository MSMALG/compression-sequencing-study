# Phase 3 — Accuracy Evaluation

## Objective

Evaluate Top-1 accuracy retention (10-class Imagenette subset) for:
1. Baseline FP32
2. Structural Pruning (P)
3. Structural Pruning → Quantization (P → Q)
4. Structural Quantization → Pruning (Q → P, repaired)

## Method

Dataset: Imagenette2-160 validation set (3,925 images, 10 classes)
Note: MobileNetV2 outputs 1000 ImageNet classes; predictions were
restricted to the 10 logits corresponding to Imagenette's classes via
an explicit wnid-to-ImageNet-index mapping before scoring. Reported
accuracy is therefore relative to this 10-class subset, not full
ImageNet-1000 top-1.

## Results

| Experiment              | Accuracy (%) |
|--------------------------|-------------:|
| Baseline FP32            |        97.35 |
| Structural Pruning (P)   |         8.38 |
| P → Q                    |         8.18 |
| Q → P (Repaired)         |         8.08 |

## Observations

- Baseline accuracy (97.35%) confirms the evaluation pipeline and label
  mapping are correct.
- All three compressed variants collapse to near-random accuracy
  (~8%, vs. ~10% chance level for a 10-class problem), regardless of
  whether quantization is applied before or after pruning.
- The spread between P, P→Q, and Q→P is under 0.4 percentage points —
  far smaller than the differences observed in latency across the same
  three pipelines in Phase 2. This suggests compression order has
  negligible effect on accuracy, in contrast to its measurable effect
  on latency and model size.
- The accuracy collapse is attributed to one-shot 20% structural
  pruning with no fine-tuning/recovery step afterward. This matches
  expected behavior from the pruning literature: aggressive one-shot
  channel pruning without retraining typically destroys learned
  representations.

## Reframing Phase 2 Conclusions

Phase 2 concluded that P → Q produced the "strongest deployment
outcome" based on model size and latency alone. Phase 3 shows this
conclusion is incomplete: while P → Q does minimize size and latency,
it also reduces classification accuracy to near-random. A compressed
model that is small and fast but non-functional does not represent a
genuine deployment improvement.

## Preliminary Conclusion

Structural pruning order (P → Q vs. Q → P) has a measurable effect on
deployment metrics (size, latency) but negligible effect on accuracy —
both orderings destroy accuracy roughly equally when no fine-tuning
step is included. This indicates that, for this pruning ratio and
method, a post-pruning fine-tuning/recovery step is necessary before
either pipeline could be considered deployment-ready.
