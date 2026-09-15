# Phase 5 — Generalized Repair and Cross-Architecture Testing

## Objective

Phase 2 repaired the Quantize→Prune dimensional mismatch for MobileNetV2
using a hand-written, architecture-specific fix that also depended on
having a separate pristine (unquantized) copy of the model available.
Phase 5 asks two questions:

1. Can this repair be rewritten as a general-purpose tool that works
   without a pristine reference model?
2. Does that tool generalize to architectures other than MobileNetV2 —
   and if pruning damage varies across architectures, what explains
   that variation?

---

## Method

### The Generalized Repair Function

A new function, `repair_quantized_linear_after_pruning`, dequantizes an
already-broken quantized layer's own weights directly, slices them to
the surviving input channels, and re-quantizes — removing the
dependency on a separate pristine model that Phase 2's fix required.

```python
def repair_quantized_linear_after_pruning(quantized_linear, kept_input_idxs):
    float_weight = quantized_linear.weight().dequantize()
    float_bias = quantized_linear.bias()

    kept_idxs_tensor = torch.tensor(kept_input_idxs, dtype=torch.long)
    sliced_weight = float_weight[:, kept_idxs_tensor]

    repaired_float_linear = nn.Linear(len(kept_input_idxs), float_weight.shape[0])
    with torch.no_grad():
        repaired_float_linear.weight.copy_(sliced_weight)
        repaired_float_linear.bias.copy_(float_bias)

    repaired_float_linear.qconfig = torch.quantization.default_dynamic_qconfig
    return torch.ao.nn.quantized.dynamic.Linear.from_float(repaired_float_linear)
```

A second function, `get_expected_input_dim`, uses a forward hook to
automatically detect how many channels are actually arriving at a
given layer, replacing Phase 2's approach of discovering the mismatch
by letting the model crash:

```python
def get_expected_input_dim(model, classifier_module, input_tensor):
    captured = {}
    def hook(module, input):
        captured['dim'] = input[0].shape[1]
    handle = classifier_module.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_tensor)
    except RuntimeError:
        pass
    handle.remove()
    return captured['dim']
```

### Test Pipeline

For each architecture: quantize (dynamic INT8, `nn.Linear` layers) →
structurally prune 20% (Torch-Pruning, `MagnitudeImportance`) →
detect the resulting mismatch → repair → evaluate accuracy
(Imagenette2-160 validation set, 10-class-masked, no fine-tuning).

### Architectures Tested

- **MobileNetV2** — inverted residual blocks, depthwise-separable convolutions.
- **ResNet-18** — standard convolutions, residual (skip-connection) blocks.
- **VGG16** — standard convolutions, no skip connections; required
  additional handling since its final conv output is flattened
  (each pruned channel corresponds to a 7×7 = 49-slot block in the
  flattened input, not a single slot) before reaching its first
  linear layer.
- **EfficientNet-B0** — depthwise-separable convolutions (MBConv
  blocks) with inverted residuals, added specifically to test whether
  MobileNetV2's behavior was a general property of depthwise-separable
  architectures or an isolated case.

---

## Results

| Architecture | Convolution Style | Skip Connections | Baseline Accuracy (%) | Repaired, No Fine-Tune (%) | Accuracy Drop |
|---|---|---|---:|---:|---:|
| MobileNetV2 | Depthwise-separable | Yes | 97.35 | 8.05 | ~89 points |
| EfficientNet-B0 | Depthwise-separable | Yes | 99.41 | 8.97 | ~90 points |
| ResNet-18 | Standard | Yes | 97.63 | 38.47 | ~59 points |
| VGG16 | Standard | No | 97.30 | 33.35 | ~64 points |

All repaired models produced correct output shapes and ran without
error using the same repair function, with no architecture-specific
changes to its core logic.

---

## What We Discovered

### 1. The repair generalizes across architectures

The same repair function fixed the Quantize→Prune dimensional mismatch
on all four architectures without modification. VGG16 required
additional logic to locate its correct target layer by exact name
(rather than by position in the pruning history, which failed — see
below) and to expand pruned channel indices across its flattened
feature map, but the underlying weight-repair logic itself was
unchanged throughout.

**A note on a false start:** an early version of the VGG16 detection
logic searched the pruning history for "the last conv-layer pruning
event by list position" rather than by exact layer name. This
incorrectly matched an earlier convolutional layer in VGG16's 13-layer
stack, not its true final layer, causing a channel-count mismatch that
was only caught because an explicit assertion failed rather than
silently producing a wrong repair. Switching to exact-name matching
(finding the true last `Conv2d` module by walking the network
structure directly) resolved this. This mirrors the lesson from
Phase 2: silent failures are far more dangerous than loud ones, and
this project's practice of asserting expected invariants at each step
caught the issue before it could produce misleading results.

### 2. Pruning-induced accuracy collapse is not explained by skip connections

An initial hypothesis, formed after testing only MobileNetV2 and
ResNet-18, proposed that ResNet-18's smaller accuracy drop was due to
its skip connections providing a path for information to bypass pruned
layers. Testing VGG16 — which has no skip connections at all — directly
contradicted this: VGG16's accuracy drop (~64 points) closely resembled
ResNet-18's (~59 points) and was far smaller than MobileNetV2's
(~89 points). The presence or absence of skip connections does not
determine how severely a network's accuracy collapses under naive
structural pruning.

### 3. Pruning-induced accuracy collapse is explained by convolution type

Adding EfficientNet-B0 as a second depthwise-separable architecture
confirmed a clear pattern: both depthwise-separable networks
(MobileNetV2, EfficientNet-B0) collapsed to near-random accuracy under
one-shot 20% pruning (~89–90 point drops), while both standard-
convolution networks (ResNet-18, VGG16) retained substantially more
accuracy (~59–64 point drops), regardless of whether they had skip
connections. This is a consistent, replicated pattern across two
independent examples on each side, not a property of a single model.

A plausible mechanism: depthwise-separable convolutions process each
channel largely in isolation before a lightweight pointwise layer mixes
information across channels, leaving each channel carrying more
unique, non-redundant information. Pruning a channel in this design
likely removes information with no substitute elsewhere in the network.
Standard convolutions mix channel information more densely throughout,
providing more redundant capacity to fall back on when a channel is
removed.

---

## Conclusion of Phase 5

The generalized repair function, developed to remove Phase 2's
dependency on a pristine reference model, was successfully validated
across four architecturally distinct networks with no changes to its
core repair logic, supporting its use as a general-purpose tool for
resolving Quantize→Prune dimensional mismatches rather than a one-off,
model-specific fix.

Separately, testing across these four architectures revealed that
pruning-induced accuracy collapse varies substantially and
systematically by architecture. An initial hypothesis attributing this
variation to skip connections was tested and rejected. A revised
hypothesis — that networks built primarily from depthwise-separable
convolutions are substantially more vulnerable to one-shot structural
pruning than networks using standard convolutions — is supported by
two independent architectures on each side of the comparison.

This finding meaningfully extends the thesis's core conclusion from
Phases 3–4. It is not only true that a post-pruning recovery step
matters more than compression order (Phase 4); it now also appears that
**how much recovery is needed, and how urgently, depends heavily on the
network's underlying convolution design** — a consideration absent from
the original thesis question and worth flagging explicitly for anyone
deploying depthwise-separable models (a category that includes many of
the most popular mobile/edge architectures) with pruning.

---

## Limitations of Phase 5

- **Two examples per category.** The depthwise-separable-vs-standard
  pattern is supported by two architectures on each side; a broader
  architecture sweep would strengthen this considerably.
- **Single pruning ratio.** All Phase 5 experiments use the same 20%
  ratio as prior phases; whether this pattern holds at other ratios is
  untested.
- **No fine-tuning tested here.** Phase 4 showed fine-tuning recovers
  MobileNetV2's accuracy substantially; whether EfficientNet-B0,
  ResNet-18, and VGG16 all recover similarly well (and whether
  depthwise-separable networks require more fine-tuning epochs to
  reach comparable recovery) has not been tested.
- **VGG16's flatten-step handling is architecture-aware, not fully
  automatic.** The core repair function is general-purpose, but
  correctly locating the target layer and channel mapping for VGG16
  required manual reasoning about its flatten step — a fully automatic
  tool would need to detect this structural pattern itself.

## Future Work (Phase 6)

- Test a true MobileNetV1 (pure depthwise-separable, no inverted
  residuals) to further isolate whether depthwise-separable convolution
  itself, independent of MobileNetV2/EfficientNet's specific block
  design, is the driving factor.
- Extend Phase 4's fine-tuning recovery experiment across all four
  architectures to test whether depthwise-separable networks require
  more recovery effort (more epochs, different learning rate) to reach
  comparable post-fine-tune accuracy.
- Generalize the VGG16-style flatten-step handling into the core
  detection logic, rather than requiring architecture-specific
  reasoning.
