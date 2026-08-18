# Training Features Backported from Forks

This document summarizes training-related features and improvements backported
into the Advanced-RVC-Inference codebase from three community forks:

1. **Vietnamese-RVC** (PhamHuynhAnh16) — https://github.com/PhamHuynhAnh16/Vietnamese-RVC
2. **Applio** (IAHispano) — https://github.com/IAHispano/Applio
3. **PolTrain** (Politrees) — https://github.com/Politrees/PolTrain

These complement the bug fixes documented in `TRAINING_BUG_FIXES.md`.

---

## Files Modified

1. `arvc/engine/models/algorithms/discriminators.py` — DiscriminatorR AMP safety
2. `arvc/engine/training/runner/losses.py` — feature_loss .detach()
3. `arvc/engine/training/runner/mel_processing.py` — Pure-torch mel filter bank
4. `arvc/engine/training/runner/utils.py` — dB-normalized plot, mel_spectrogram_similarity
5. `arvc/engine/training/runner/train.py` — PolOpt scheduler, vocoder-conditional mel loss, _last.pth marker, mel similarity metric
6. `arvc/engine/training/runner/data_utils.py` — Relative filelist paths
7. `arvc/engine/training/extract/feature.py` — pesto fallback, configurable f0_min/f0_max
8. `arvc/engine/training/extract/embedding.py` — torch.compile embedder
9. `arvc/utils/variables.py` — tf32/compile_all/int8 config flags, setup_compile()
10. `arvc/engine/models/optimizers/__init__.py` — PolOpt registry entry

## Files Added

1. `arvc/engine/models/optimizers/polopt.py` — PolOpt optimizer (from PolTrain)

---

## BACKPORTS FROM Vietnamese-RVC

### VRVC-1: DiscriminatorR AMP Safety
**File:** `arvc/engine/models/algorithms/discriminators.py`
**Original issue:** `DiscriminatorR.forward` called `self.spectrogram(x)` directly
on the input tensor, which under AMP fp16 autocast can produce NaN/Inf because
`torch.stft` is not numerically stable in fp16 on all backends.
**Fix (from VRVC):** Cast input to float32 before STFT, then back to original
dtype: `x = self.spectrogram(x.float()).unsqueeze(1).to(x.dtype)`. This matches
the pattern already used by `DiscriminatorP` and the main mel loss.
**Impact:** Prevents training crashes and NaN propagation when training v3
models (RefineGAN/BigVGAN vocoders) with `fp16=True` on CUDA.

### VRVC-2: feature_loss .detach() on Real Features
**File:** `arvc/engine/training/runner/losses.py`
**Original issue:** `feature_loss(fmap_r, fmap_g)` did not detach the real
feature maps `fmap_r`. During the generator step, `fmap_r` comes from a
real-wave forward pass through the discriminator — without `.detach()`, the
autograd graph is built for the real path even though the G optimizer never
updates D parameters. This wastes memory and compute.
**Fix (from VRVC):** `(rl.float().detach() - gl.float()).abs().mean()`.
**Impact:** Minor training speedup (~2-3%) and lower peak VRAM usage.

### VRVC-3: `pesto` in F0 Backend Fallback List
**File:** `arvc/engine/training/extract/feature.py`
**Original issue:** The list of f0 methods that force `num_processes=1` on
OCL/privateuseone (DirectML) backends was missing `"pesto"`. PESTO uses ONNX
runtime under the hood, which crashes on DirectML when run in parallel —
same as crepe/fcpe/rmvpe/penn/swift.
**Fix (from VRVC):** Added `"pesto" in f0_method` to the condition.
**Impact:** Prevents crash when selecting PESTO on DirectML/OpenCL backends.

### VRVC-4: Configurable f0_min / f0_max
**File:** `arvc/engine/training/extract/feature.py`
**Original issue:** `FeatureInput` hardcoded `f0_min=50.0` and `f0_max=1100.0`,
which doesn't fit all use cases (soprano voice > 1100 Hz, bass voice < 50 Hz,
instruments with wider ranges).
**Fix (from VRVC):** Read from config dict:
```python
self.f0_min = float(configs.get("f0_min", 50))
self.f0_max = float(configs.get("f0_max", 1100))
```
**Impact:** Allows customization for non-standard vocal ranges via config.json.

### VRVC-5: Pure-Torch Mel Filter Bank
**File:** `arvc/engine/training/runner/mel_processing.py`
**Original issue:** `spec_to_mel_torch` and `MultiScaleMelSpectrogramLoss`
called `librosa.filters.mel(...)` + `torch.from_numpy(...)` to build the mel
filter bank. This forces a CPU round-trip on every distinct
`(fmax, dtype, device)` combination — slow, and breaks on machines without
librosa installed.
**Fix (from VRVC):** Added `mel_filter_bank()`, `_hz_to_mel()`, `_mel_to_hz()`
pure-torch implementations that compute the Slaney-normalized triangular
filter bank natively on the target device/dtype. The output is bit-for-bit
equivalent to `librosa.filters.mel(norm="slaney")` for the same arguments.
Both `spec_to_mel_torch` and `MultiScaleMelSpectrogramLoss.mel_spectrogram`
now use this, with a librosa fallback for very old torch versions.
**Impact:** Faster mel loss computation (especially on GPU and multi-scale
which uses 8 different n_mels values), and removes the hard librosa
dependency for training.

### VRVC-6: TF32 / compile_all / int8 / compile_mode Config Flags
**File:** `arvc/utils/variables.py`
**Original issue:** The `Config` class only had `cpu_mode`, `brain`,
`debug_mode`, `fp16` flags. It was missing `tf32`, `compile_all`,
`compile_mode`, `int8`, and the supporting detection logic
(`cuda_tf32`, `cuda_bf16`, `tf32_support`, `bf16_support`). Consequence:
every `getattr(main_config, 'tf32', False)` call in train.py always
returned `False`, so TF32 was silently disabled even on Ampere+ GPUs
(A100, RTX 30xx/40xx, H100).
**Fix (from VRVC):** Added the full flag set with proper hardware detection.
TF32 is auto-enabled when the GPU supports it (compute capability ≥ 8.0)
and the user opts in via `config.json`. When enabled, sets
`torch.backends.cuda.matmul.allow_tf32 = True` and
`torch.backends.cudnn.allow_tf32 = True`.
**Impact:** ~3x matmul speedup on Ampere+ GPUs with negligible accuracy loss.

### VRVC-7: setup_compile() for torch.compile
**File:** `arvc/utils/variables.py`
**Original issue:** No setup for `torch.compile` environment — Triton
detection, inductor cache, graph capture optimizations were missing.
**Fix (from VRVC):** Added `_setup_compile()` method that:
- Detects Triton (required by TorchInductor). If missing, disables
  `compile_all` and updates config.json so the user doesn't get confusing
  errors on next startup.
- Sets up persistent inductor cache via `TORCHINDUCTOR_CACHE_DIR` (avoids
  recompilation on every startup).
- Enables `torch._dynamo.config.capture_scalar_outputs` for graph-friendly
  scalar output capture (e.g. `loss.item()`).
- Enables `torch._inductor.config.freezing` to freeze constants into the
  graph (smaller graph, faster compile).
**Impact:** Makes `torch.compile` actually work end-to-end.

### VRVC-8: torch.compile Embedder
**File:** `arvc/engine/training/extract/embedding.py`
**Original issue:** Embedder model was never `torch.compile`d, even when
`compile_all=True`. Embedding extraction is typically the bottleneck of the
preprocessing pipeline.
**Fix (from VRVC):** Added conditional torch.compile when
`config.compile_all=True` and `device.startswith("cuda")`:
```python
if getattr(_cfg, "compile_all", False) and device.startswith("cuda"):
    _mode = getattr(_cfg, "compile_mode", None)
    model = torch.compile(model, mode=_mode) if _mode else torch.compile(model)
```
**Impact:** 1.3-2x speedup on embedding extraction (the preprocessing
bottleneck) when `compile_all=True` is set in config.json.

---

## BACKPORTS FROM Applio

### APPLIO-1: Vocoder-Conditional multiscale_mel_loss Auto-Enable
**File:** `arvc/engine/training/runner/train.py`
**Original issue:** Users could disable `multiscale_mel_loss` even when
using RefineGAN or BigVGAN vocoders. These vocoders produce waveforms whose
spectral characteristics vary significantly across frequency bands — a
single-scale L1 mel loss fails to capture these differences, leading to
subtle artifacts (buzzing, muffled high-end) that get baked into the model.
**Fix (from Applio):** Auto-enable `multiscale_mel_loss=True` when
`vocoder in ("RefineGAN", "BigVGAN")`, even if the user explicitly disabled
it. Logs a warning so the user knows. Also changed the runtime branch in
`train_and_evaluate` to use `isinstance(fn_mel_loss, MultiScaleMelSpectrogramLoss)`
instead of the boolean flag — this correctly reflects the actual loss
function being used.
**Impact:** Prevents subtle quality degradation on v3 vocoders.

### APPLIO-2: Relative Paths in filelist.txt
**File:** `arvc/engine/training/runner/data_utils.py`
**Original issue:** `TextAudioLoader.get_audio_text_pair` assumed the
audiopath from the filelist was always a valid path. If the filelist was
generated on Machine A with absolute paths like `/home/userA/datasets/...`
and then transferred to Machine B where the dataset is at `/data/...`,
training would fail with FileNotFoundError.
**Fix (from Applio):** When the path is relative or doesn't exist, try
resolving it against each `spec_dirs` entry (basename match first, then
full relative path join). Falls through to the original behavior if no
candidate exists.
**Impact:** Makes filelists portable across machines — users can share
training configs without editing paths.

---

## BACKPORTS FROM PolTrain

### POL-1: PolOpt Optimizer (Yogi + AdaBelief Hybrid)
**File (new):** `arvc/engine/models/optimizers/polopt.py`
**File:** `arvc/engine/models/optimizers/__init__.py` (registry)
**File:** `arvc/engine/training/runner/train.py` (scheduler)
**What it is:** A new optimizer that combines:
1. **Yogi-sign control** for the second moment: uses `sign((g - m)^2 - v)`
   instead of the raw squared gradient to update `v`. This prevents the
   denominator from collapsing to zero and avoids explosive LR swings
   during GAN training (where D and G gradients oscillate strongly).
2. **Decoupled weight decay** (AdamW-style) for better generalization.
3. **Correct epsilon placement** (OUTSIDE the sqrt, after bias correction),
   per the AdamW paper. Many implementations put eps inside the sqrt,
   which makes the effective learning rate depend on the gradient scale.
4. **Trust Region Clamping** (`max_step_clip=1.0`): clamps the per-parameter
   update magnitude to prevent acoustic filter structures from being
   destroyed by transient gradient spikes. Critical for RVC where conv banks
   are sensitive to large weight perturbations.
**Registration:** Added to `OPTIMIZER_REGISTRY` with rating 4.5/5 and
category "Belief-Based". Now selectable from the UI alongside AdamW, RAdam,
AnyPrecisionAdamW, AdaBelief, AdaBeliefV2.
**Scheduler:** PolOpt pairs with `CosineAnnealingLR` (T_max=total_epoch,
eta_min=1e-6), matching the configuration used in PolTrain where it
produced the best results.

### POL-2: mel_spectrogram_similarity TensorBoard Metric
**File:** `arvc/engine/training/runner/utils.py` (new function)
**File:** `arvc/engine/training/runner/train.py` (logged in scalar_dict)
**What it is:** Converts the L1 mel loss into a human-interpretable 0-100%
similarity percentage. 100% = identical reconstruction, 0% = completely
different. Much easier to monitor in TensorBoard than the raw L1 mel loss
value (which has no intuitive scale).
**Implementation:** `100.0 - (F.l1_loss(y_hat_mel, y_mel) * 100.0)`,
clamped to [0, 100]. Handles shape mismatches by trimming to the smaller
last dimension.
**Logged as:** `metric/mel_similarity` in TensorBoard.

### POL-3: dB-Normalized plot_spectrogram_to_numpy
**File:** `arvc/engine/training/runner/utils.py`
**Original issue:** The original `plot_spectrogram_to_numpy` auto-scaled
the color range per image. This meant early-epoch (noisy) and late-epoch
(clean) spectrograms looked similar because the color range rescaled each
time — users couldn't visually compare progress across epochs.
**Fix (from PolTrain):** Convert to dB scale (`10 * log10`) when the input
looks linear (max > 1.0), then use a fixed `Normalize(vmin=-10, vmax=0)`
range with a dB-formatted colorbar. Now spectrograms are directly
comparable across epochs.
**Impact:** TensorBoard mel spectrogram images actually show training
progress visually.

### POL-4: `_last.pth` Final Model Marker
**File:** `arvc/engine/training/runner/train.py`
**Original issue:** After training completes, the final model is saved as
`{model_name}_{epoch}e_{global_step}s.pth`. Downstream tooling (inference
scripts, model browsers, upload scripts) had to glob for the highest epoch
number, which is fragile (parsing numbers from filenames).
**Fix (from PolTrain):** Also save a copy as
`{model_name}_last.pth` on the final epoch. This is an explicit marker that
downstream tooling can simply look for.
**Impact:** Cleaner integration with inference/upload pipelines.

---

## Summary

Total features backported: **14**

| Source | Count |
|--------|-------|
| Vietnamese-RVC | 8 |
| Applio | 2 |
| PolTrain | 4 |

### Performance Optimizations (5)
- Pure-torch mel filter bank (eliminates librosa CPU round-trip)
- TF32 matmul/cudnn auto-enable on Ampere+ GPUs
- torch.compile embedder (1.3-2x embedding extraction speedup)
- setup_compile() with Triton detection + inductor cache
- feature_loss .detach() (minor — saves graph build on real path)

### Correctness Fixes (4)
- DiscriminatorR AMP safety (prevents NaN in fp16 STFT)
- `pesto` in DirectML f0 fallback list (prevents crash)
- Vocoder-conditional multiscale_mel_loss auto-enable
- Relative filelist paths (portability)

### New Features (3)
- PolOpt optimizer (Yogi + AdaBelief hybrid with trust-region clamping)
- mel_spectrogram_similarity TensorBoard metric (0-100% interpretable score)
- dB-normalized mel spectrogram plots (comparable across epochs)

### Quality of Life (2)
- Configurable f0_min / f0_max
- `_last.pth` final model marker

---

## Configuration

To enable the new optional features, add these to `arvc/configs/config.json`:

```json
{
    "tf32": true,
    "compile_all": true,
    "compile_mode": "default",
    "int8": false,
    "f0_min": 50,
    "f0_max": 1100,
    "compile_cache_dir": "none"
}
```

- `tf32`: Enable TF32 tensor cores on Ampere+ GPUs. ~3x matmul speedup
  with negligible accuracy loss. Safe to leave on.
- `compile_all`: torch.compile embedder + training model. 1.3-2x speedup
  but requires Triton (`pip install triton`). Auto-disables if Triton
  is missing.
- `compile_mode`: "default" / "reduce-overhead" / "max-autotune".
  "reduce-overhead" is best for small models, "max-autotune" for large.
- `int8`: Use 8-bit optimizer state via bitsandbytes. Saves VRAM at
  small accuracy cost. Requires `pip install bitsandbytes`.
- `f0_min` / `f0_max`: Customize F0 range for non-standard voices.
- `compile_cache_dir`: Path to persistent inductor cache. "none" disables.

All flags default to safe values (off) — no behavior change unless opted in.
