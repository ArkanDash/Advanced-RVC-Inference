# Training Bug Fixes Summary

This document summarizes all training-related bugs identified and fixed in the
Advanced-RVC-Inference codebase.

> **Note (v2.2.2):** File paths below reflect the package reorganization
> (see [`docs/PACKAGE_STRUCTURE.md`](docs/PACKAGE_STRUCTURE.md) for the
> migration cheat-sheet). The most common change is
> `arvc/services/training.py` → `arvc/services/training/training.py`.

## Files Modified

1. `arvc/engine/training/runner/train.py` — Main training script (10 fixes)
2. `arvc/engine/training/runner/data_utils.py` — Dataset & sampler (2 fixes)
3. `arvc/engine/training/runner/mel_processing.py` — Mel loss (1 fix)
4. `arvc/engine/training/runner/utils.py` — Checkpoint/logging utils (2 fixes)
5. `arvc/engine/training/runner/extract_model.py` — Model export (1 fix)
6. `arvc/engine/training/extract/embedding.py` — Embedding extraction (2 fixes)
7. `arvc/engine/training/extract/preparing_files.py` — Config/filelist generation (1 fix)
8. `arvc/engine/training/extract/rms.py` — RMS energy extraction (1 fix)
9. `arvc/engine/training/create_dataset.py` — Dataset creation (2 fixes)
10. `arvc/engine/models/utils.py` — Feature extraction (1 fix)
11. `arvc/services/training/training.py` — Training service layer (2 fixes)

---

## CRITICAL Bug Fixes

### FIX 1: Gradient Accumulation Completely Broken
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `optim_g.zero_grad()` was called EVERY iteration, clearing accumulated
gradients and completely defeating gradient accumulation. When `grad_accum_steps > 1`,
only the LAST batch's gradients were used for the update. Additionally,
`scaler.unscale_(optim_g)` was called every iteration, which raises
`RuntimeError: unscale_() has already been called for this optimizer` on the
second call.
**Fix:** Restructured the G step to only call `zero_grad()` AFTER `step()`, and
only call `unscale_()`/`step()`/`zero_grad()` on the accumulation step. Backward
is called every iteration to accumulate gradients.

### FIX 2: Energy Loss Tensor Shape Mismatch
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `commons.slice_segments()` was called with `dim=2` on a 3D tensor
`[batch, 1, seq_len]`. The function slices `x[:, :segment_size]` for `dim=2`,
which on the size-1 axis returns the full unsliced tensor. This meant
`energy_slice` was NOT sliced to the segment, and the L1 loss compared a single
scalar prediction against every frame — semantically meaningless.
**Fix:** Changed `dim=2` to `dim=3`, which slices the last axis (seq_len) of the
3D tensor, matching how `wave` is sliced elsewhere.

### FIX 3: `configs` Not Imported in embedding.py
**File:** `arvc/engine/training/extract/embedding.py`
**Bug:** Line 168 referenced `configs.get("logs_path", ...)` but only `config`
(not `configs`) was imported. This raised `NameError: name 'configs' is not
defined` when the fallback mute path was triggered.
**Fix:** Added `configs` to the import statement.

### FIX 4: MultiScaleMelSpectrogramLoss Missing `num_scales` Property
**File:** `arvc/engine/training/runner/mel_processing.py`
**Bug:** `train.py` uses `getattr(fn_mel_loss, 'num_scales', 3)` to normalize
the mel loss. The class has 8 scales but no `num_scales` attribute, so the
default of 3 was always returned. This made the mel loss ~2.67x larger than
intended (8/3), skewing the loss balance.
**Fix:** Added a `num_scales` property that returns `len(self.stft_params)`.

### FIX 5: Resume from Checkpoint Crashes on Empty Deques
**File:** `arvc/engine/training/runner/train.py`
**Bug:** On checkpoint resume, `global_step` may already be a multiple of 50,
triggering the TensorBoard logging code on the first batch. But the deques are
empty (not persisted in checkpoint), causing `ZeroDivisionError` (sum([])/0)
and `RuntimeError` (torch.stack([])).
**Fix:** Added empty deque checks before computing averages.

### FIX 6: `scaler.update()` Not Called Frequently Enough
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `scaler.update()` was only called on accumulation steps (when G
stepped). But `scaler.step(optim_d)` was called every iteration for the D
step. If D gradients had NaN/Inf, the scale factor was never reduced until the
next G step, causing repeated skipped updates.
**Fix:** Added `scaler.update()` at the end of every iteration.

### FIX 7: Embedder Layer Mixing Passes Projected Features as Source
**File:** `arvc/engine/models/utils.py`
**Bug:** After the first `extract_features()` call, `feats` was overwritten
with projected features (output of `final_proj`). The mix branch then passed
these projected features as `source` to the second `extract_features()` call,
which expects raw audio input. This produced garbage features when
`embedders_mix=True`.
**Fix:** Saved the original raw-audio input before overwriting `feats`, and
used the original input for the mix branch.

---

## HIGH-Severity Bug Fixes

### FIX 8: SVC DataLoader Has No DistributedSampler
**File:** `arvc/engine/training/runner/train.py`
**Bug:** SVC architecture had `batch_sampler=None` and `shuffle=False`, meaning
all GPU ranks processed the same data (wasting multi-GPU) and data was loaded
in fixed order every epoch.
**Fix:** Added `DistributedSampler` for SVC architecture with proper
`num_replicas`/`rank`/`shuffle` settings.

### FIX 9: DataLoader with batch_sampler Cannot Specify batch_size/shuffle
**File:** `arvc/engine/training/runner/train.py`
**Bug:** Original code passed `batch_size` and `shuffle` alongside
`batch_sampler`, which is invalid in PyTorch (causes ValueError in some
versions).
**Fix:** Conditionally pass arguments using `loader_kwargs` dict — only pass
`batch_size`/`shuffle`/`sampler` for SVC, or `batch_sampler` for RVC.

### FIX 10: DDP for Single-Process Training Adds Overhead
**File:** `arvc/engine/training/runner/train.py`
**Bug:** DDP was applied even for single-process training (n_gpus == 1), adding
gradient synchronization overhead without benefit. Also, `dist.init_process_group`
was always called even when unnecessary.
**Fix:** Skip DDP wrapping when `n_gpus <= 1`. Made DDP init more robust with
try/except.

### FIX 11: `len(train_loader) < 3` Check Can Cause DDP Deadlock
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `sys.exit(1)` inside a DDP child process without calling
`dist.destroy_process_group()` causes other DDP processes to hang waiting for
the exited process.
**Fix:** Added `dist.destroy_process_group()` before `sys.exit(1)`.

### FIX 12: Overtraining Detector History Not Restored to Global Scope
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `continue_overtrain_detector()` assigned to `loss_disc_history` etc.
without `global` declaration, creating LOCAL variables instead of updating the
module-level globals. The loaded history was never actually used.
**Fix:** Added `global` declaration for all four history variables.

### FIX 13: Reference Inference Uses Batched Tensors
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `next(iter(train_loader))` returns a BATCH (batch_size items).
`net_g.infer()` expects a SINGLE example (shape [1, T, D]). When batch_size >
1, the infer call received batched input, producing incorrect output.
**Fix:** Select only the first item from the batch with `[:1]` slicing.

### FIX 14: torch.compile Applied After DDP Wrapping
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `torch.compile` was applied to the DDP-wrapped model. Per PyTorch
docs, the recommended order is `DDP(torch.compile(model))`, not
`torch.compile(DDP(model))`. Compiling the DDP wrapper can cause gradient sync
issues.
**Fix:** Reordered to apply `torch.compile` BEFORE DDP wrapping.

### FIX 15: `torch.xpu.manual_seed` / `opencl.pytorch_ocl.manual_seed_all` May Not Exist
**File:** `arvc/engine/training/runner/train.py`
**Bug:** These APIs may not exist in all PyTorch installations, causing
`AttributeError`.
**Fix:** Use the universal `torch.manual_seed()` which seeds all devices.

### FIX 16: Non-Detached Tensors in TensorBoard Logging
**File:** `arvc/engine/training/runner/utils.py`
**Bug:** `writer.add_scalar(k, v, global_step)` with a tensor `v` that has
`requires_grad=True` retains the computation graph, causing GPU memory leak over
many epochs.
**Fix:** Added `.detach()` for all tensor values in `summarize()`.

---

## MEDIUM-Severity Bug Fixes

### FIX 17: `np.fromstring` Deprecated
**File:** `arvc/engine/training/runner/utils.py`
**Bug:** `np.fromstring` is deprecated since NumPy 1.14 and removed in NumPy 2.0+.
**Fix:** Replaced with `np.frombuffer` with a final fallback to `buffer_rgba`.

### FIX 18: DistributedBucketSampler._bisect Drops Boundary Items
**File:** `arvc/engine/training/runner/data_utils.py`
**Bug:** The condition `self.boundaries[mid] < x` used strict less-than. Items
whose length exactly equals a boundary value were silently dropped (returned -1).
**Fix:** Changed to `self.boundaries[mid] <= x <= self.boundaries[mid + 1]`.

### FIX 19: Wrong Length Estimate in TextAudioLoader
**File:** `arvc/engine/training/runner/data_utils.py`
**Bug:** `os.path.getsize(audiopath) // (3 * self.hop_length)` used magic number
`3` which doesn't match any common audio format (16-bit PCM = 2 bytes/sample,
32-bit float = 4 bytes/sample).
**Fix:** Use `soundfile.info()` for accurate frame count, with fallback to
`getsize // (2 * hop_length)`.

### FIX 20: `loss_mel = torch.clamp(loss_mel, max=100.0)` Blocks Gradient Flow
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `torch.clamp` with `max=100.0` has ZERO gradient when loss > 100,
causing training to get stuck if mel loss is very high at the start.
**Fix:** Replaced with soft clamp: `100.0 * torch.tanh(loss_mel / 100.0)`.

### FIX 21: `epoch % save_every_epoch == False` Compares Int with Bool
**File:** `arvc/engine/training/runner/train.py`
**Bug:** Comparing integer with boolean is a code smell (works in Python but
confuses linters).
**Fix:** Changed to `== 0`.

### FIX 22: cleanup Deletes Checkpoints Without try/except
**File:** `arvc/engine/training/runner/train.py`
**Bug:** `os.remove(file_path)` raises `FileNotFoundError` if the file is already
gone, or `PermissionError` on Windows when files are locked.
**Fix:** Wrapped all `os.remove`/`os.rmdir` calls in try/except.

### FIX 23: `model_hash` Is Effectively Random
**File:** `arvc/engine/training/runner/extract_model.py`
**Bug:** `str(ckpt)` returns "OrderedDict([...])" (no actual weight values), and
`datetime.now()` has microsecond precision. The hash was effectively random —
calling `extract_model` twice on the same checkpoint produced different hashes.
**Fix:** Hash the actual tensor data via `torch.save` into a buffer for a
deterministic, content-based hash.

### FIX 24: Relative Mute Output Path
**File:** `arvc/engine/training/extract/embedding.py`
**Bug:** `mute_out_path` used a relative path (`os.path.join("..", "assets", ...)`)
resolved against CWD, not the package root. When run from a different CWD, the
mute file was saved to the wrong location.
**Fix:** Use an absolute path based on the file's location.

### FIX 25: `cut_preprocess` Not Converted to str()
**File:** `arvc/services/training/training.py`
**Bug:** `subprocess.Popen` requires all args to be strings, but `cut_preprocess`
was not converted.
**Fix:** Added `str(cut_preprocess)`.

### FIX 26: `if_done` Busy-Waits with `time.sleep(0.5)`
**File:** `arvc/services/training/training.py`
**Bug:** Busy-wait loop polling `p.poll()` every 0.5s wastes CPU cycles.
**Fix:** Replaced with `p.wait()` which blocks efficiently.

### FIX 27: Relative Paths in create_dataset.py
**File:** `arvc/engine/training/create_dataset.py`
**Bug:** `dataset_temp` and `DATASET_DIR` used relative paths resolved against CWD.
**Fix:** Use absolute paths based on the package root.

### FIX 28: Bare `except:` in create_dataset.py
**File:** `arvc/engine/training/create_dataset.py`
**Bug:** Bare `except:` catches `KeyboardInterrupt`, `SystemExit`, `MemoryError`.
**Fix:** Changed to `except Exception:`.

### FIX 29: Bare `except:` in rms.py
**File:** `arvc/engine/training/extract/rms.py`
**Bug:** Same as above.
**Fix:** Changed to `except Exception:`.

### FIX 30: `generate_config` Uses `os.getcwd()` for Config Path
**File:** `arvc/engine/training/extract/preparing_files.py`
**Bug:** Config template path used `os.getcwd()` which is the current working
directory, not the package root. When run from a different CWD, the config
template was not found.
**Fix:** Use the package root based on the file's location, with fallback to
`configs.get("configs_path")`.

---

## Summary

Total bugs fixed: **30**

| Severity | Count |
|----------|-------|
| CRITICAL | 7 |
| HIGH | 9 |
| MEDIUM | 14 |

All fixes maintain backward compatibility and follow the existing code style.
Each fix includes a detailed comment explaining the original issue and the
rationale for the fix.
