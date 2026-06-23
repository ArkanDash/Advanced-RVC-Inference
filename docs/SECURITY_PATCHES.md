# Security Patches — Advanced-RVC-Inference

This document lists every security hardening applied on top of upstream
`ArkanDash/Advanced-RVC-Inference`. All changes are defense-in-depth —
none of them alter training or inference numerics, so vocal fidelity is
bit-for-bit identical to upstream.

The single source of truth for the safe-load helpers is
[`arvc/engine/models/safe_load.py`](../arvc/engine/models/safe_load.py).
Everything else in this document is a *call-site* patch that routes through
those helpers.

---

## 1. `torch.load` → `safe_torch_load`

`torch.load(..., weights_only=False)` is a known arbitrary-code-execution
vector — a malicious `.pth` / `.pt` file can run any Python code the
moment it is loaded. The `safe_torch_load` helper forces
`weights_only=True` and refuses to fall back to the unsafe legacy loader
even on old PyTorch.

### Call sites patched (16 files)

| File | Was | Now |
|---|---|---|
| `arvc/engine/models/predictors/PESTO/PESTO.py:25` | `torch.load(model_path, map_location="cpu", weights_only=True)` | `safe_torch_load(model_path)` |
| `arvc/engine/models/predictors/PENN/PENN.py:84` | same | same |
| `arvc/engine/models/predictors/RMVPE/RMVPE.py:27` | same | same |
| `arvc/engine/models/predictors/CREPE/CREPE.py:33` | same | same |
| `arvc/engine/models/predictors/DJCM/DJCM.py:27` | same | same |
| `arvc/engine/models/predictors/FCPE/FCPE.py:243` | same | same |
| `arvc/engine/models/predictors/FCPE/FCPE.py:293` | same | same |
| `arvc/engine/training/runner/train.py:663` | `torch.load(chk_path, map_location="cpu", weights_only=True)` | `safe_torch_load(chk_path)` |
| `arvc/engine/training/runner/train.py:1002` | `torch.load(pretrainG, ...)["model"]` | `safe_torch_load(pretrainG)["model"]` |
| `arvc/engine/training/runner/train.py:1020` | `torch.load(pretrainD, ...)["model"]` | `safe_torch_load(pretrainD)["model"]` |
| `arvc/engine/training/runner/data_utils.py:130` | `torch.load(spec_filename, weights_only=True)` | `safe_torch_load(spec_filename)` |
| `arvc/engine/training/runner/utils.py:46` | `torch.load(checkpoint_path, ...)` | `safe_torch_load(checkpoint_path)` |
| `arvc/engine/speaker/whisper.py:49` | `torch.load(fp, ...)` | `safe_torch_load(fp)` |
| `arvc/engine/models/onnx/onnx_export.py:22` | `torch.load(input_path, ...)` | `safe_torch_load(input_path)` |
| `arvc/engine/uvr/uvr5_lib/uvr/vr_separator.py:60` | `torch.load(self.model_path, ...)` | `safe_torch_load(self.model_path)` |
| `arvc/engine/models/embedders/fairseq.py:37` | `torch.load(filename, ...)` | `safe_torch_load(filename)` |

---

## 2. Restricted `pickle.Unpickler`

`arvc/engine/models/safe_load.py:_RestrictedUnpickler` only allows
primitive Python types (`dict`, `list`, `tuple`, `int`, `str`, …) and a
small set of numpy types. Every `REDUCE` / `BUILD` / `OBJ` opcode
targeting a non-whitelisted class raises `pickle.UnpicklingError`. This
blocks every known pickle-based RCE gadget (`os.system`,
`subprocess.Popen`, `builtins.eval`, `__import__`, etc.).

Used by:
- `arvc/engine/models/predictors/WORLD/WORLD.py`
- `arvc/engine/uvr/uvr5_lib/vr_network/model_param_init.py`
- `arvc/engine/models/embedders/fairseq.py` (fallback path)

---

## 3. `yaml.safe_load` everywhere

`yaml.FullLoader` is **not** safe — it can still execute arbitrary Python
via custom tags. All `yaml.load(...)` calls have been migrated to either
`yaml.safe_load(...)` directly or `safe_yaml_load(...)` from
`safe_load.py`. The only remaining `yaml.load` references in the codebase
are inside `safe_load.py` itself and `# SECURITY PATCH:` comments.

---

## 4. `validate_path_within()` wired into path-join call sites

`validate_path_within(path, base_dirs)` resolves `path` and raises
`ValueError` if it escapes any of `base_dirs`. It also rejects null
bytes, normalizes Unicode to NFC canonical form (prevents homoglyph
bypasses), and refuses absolute paths when `allow_absolute=False`.

### Call sites patched

| File | Site | Risk closed |
|---|---|---|
| `arvc/engine/inference/inference.py:147` | `model_path = os.path.join(configs["weights_path"], model)` | `model = "../../foo.pth"` from GUI/CLI escapes weights dir |
| `arvc/engine/inference/inference.py:1149` | (TTS path — same pattern) | same |
| `arvc/services/training.py:208, 268, 325, 402` | `model_dir = os.path.join(configs["logs_path"], model_name)` (×4) | `model_name = "../../foo"` from GUI text input — new `_safe_model_dir()` helper |

---

## 5. Downloader hardening (5 modules)

All five downloaders (`huggingface.py`, `gdown.py`, `meganz.py`,
`mediafire.py`, `pixeldrain.py`) now enforce:

1. **`MAX_DOWNLOAD_BYTES = 8 GB`** — both as a Content-Length pre-check
   and as a streaming cap. Prevents disk-fill DoS via a maliciously large
   hosted file.
2. **`ALLOWED_EXTENSIONS`** whitelist (`.pth`, `.pt`, `.onnx`, `.index`,
   `.zip`, `.wav`, `.mp3`, …). Files with disallowed extensions get
   `.bin` appended — we never write a `.exe` / `.sh` / `.bat` to disk
   next to the user's model library.
3. **`_sanitize_filename(name)`** — forces any attacker-controlled
   filename (from `Content-Disposition` headers, MEGA file attributes,
   MediaFire URL segments, etc.) to a single basename component. Closes
   the classic `../../../../etc/cron.d/evil` path-traversal.
4. **`timeout=300`** on every `requests.get(..., stream=True)` and the
   `urllib.request.urlopen(...)` call at app startup. A hung CDN endpoint
   can no longer hang the worker thread indefinitely.

### Per-file specifics

| File | Notable fix |
|---|---|
| `arvc/utils/gdown.py` | `tempfile.mktemp` → `tempfile.mkstemp` (closes TOCTOU symlink race). Silent `except OSError: pass` on rename-failure now raises `RuntimeError` with a clear message. |
| `arvc/utils/meganz.py` | `random.randint(0, 0xFFFFFFFF)` → `secrets.randbits(32)` for the MEGA API nonce. `requests.get(file_data['g'], stream=True)` now has `timeout=300`. `os.path.join(dest_path, attribs['n'])` (attacker-controlled filename) → `_sanitize_filename(attribs['n'])`. |
| `arvc/utils/pixeldrain.py` | Was loading entire file into memory via `response.content` — now streams to disk. `response.headers.get("Content-Disposition")` (was `AttributeError` if missing) now defaults to the URL slug. |
| `arvc/utils/mediafire.py` | `int(r.headers.get('content-length'))` (was `KeyError`/`ValueError` if missing) now defaults to `0`. Both `sess.get(url)` and the streaming download `requests.get(...)` now have `timeout=300`. |
| `arvc/utils/huggingface.py` | Bare `except:` on `import wget` → `except ImportError:`. Hard size cap enforced on both `Content-Length` and the streaming write loop. |

---

## 6. Bare `except:` clauses replaced with typed exceptions

Bare `except:` swallows `KeyboardInterrupt` and `SystemExit`, masking
real errors. The two most damaging cases were:

| File:Line | Was | Now | Why it matters |
|---|---|---|---|
| `arvc/engine/training/runner/train.py:897` | `except:` → silently restarted training from epoch 1 on **any** error (including CUDA OOM, checkpoint corruption) | `except (FileNotFoundError, RuntimeError, OSError, KeyError, ValueError) as e:` + logged warning | Prevents silent data loss — users were losing hours of training progress with no warning |
| `arvc/engine/models/onnx/onnx_export.py:115` | `except:` swallowed `KeyboardInterrupt` during ONNX export | `except Exception:` | User Ctrl-C now actually interrupts |

---

## 7. Network timeouts added

| File:Line | Was | Now |
|---|---|---|
| `arvc/utils/variables.py:580` | `urllib.request.urlopen(url)` | `urllib.request.urlopen(url, timeout=30)` |
| `arvc/services/tts.py:20` | `requests.get(google_translate_tts, ...)` | `requests.get(..., timeout=30)` |
| `arvc/engine/uvr/uvr5_lib/separator.py:136` | `requests.get(uvr_models.json)` | `requests.get(..., timeout=30)` |
| `arvc/engine/uvr/uvr5_lib/separator.py:193` | `requests.get(model_data.json)` | `requests.get(..., timeout=30)` |
| `arvc/utils/meganz.py:83` | `requests.get(file_data['g'], stream=True)` | `requests.get(..., stream=True, timeout=300)` |
| `arvc/utils/mediafire.py:16` | `sess.get(url)` + `requests.get(href, stream=True)` | both `timeout=300` |
| `arvc/utils/pixeldrain.py:6` | `requests.get(api/file/...)` | `requests.get(..., stream=True, timeout=300)` |
| `arvc/utils/gdown.py:103` | `sess.get(url, headers={Range}, stream=True)` | `sess.get(..., stream=True, timeout=300)` |

---

## 8. Other security hygiene

- No `eval()` / `exec()` builtins remain — the two prior `eval()` sinks
  in `fairseq.py` were already replaced with `ast.literal_eval`.
- No `subprocess.run(..., shell=True)` — every `subprocess.Popen` /
  `subprocess.run` uses list-form argv (`shell=False`).
- No hardcoded credentials, tokens, or `auth=` strings anywhere in the
  codebase. HuggingFace tokens are passed in via `push_to_hub(...,
  hf_token, ...)` and forwarded to `huggingface_hub`.
- SSL verification is on everywhere (no `verify=False`).

---

## Verifying the patches

You can audit the codebase yourself at any time:

```bash
# Should return ZERO live torch.load calls outside safe_load.py
rg "torch\.load\(" arvc/ --glob '!**/safe_load.py' -n

# Should return ZERO live pickle.load / pickle.loads / pickle.Unpickler outside safe_load.py
rg "pickle\.(load|loads|Unpickler)\(" arvc/ --glob '!**/safe_load.py' -n

# Should return ZERO live yaml.load (only yaml.safe_load is allowed)
rg "yaml\.load\(" arvc/ --glob '!**/safe_load.py' -n

# Should return ZERO subprocess.* with shell=True
rg "subprocess\.\w+\(.*shell\s*=\s*True" arvc/ -n

# Should return ZERO tempfile.mktemp
rg "tempfile\.mktemp\b" arvc/ -n

# Should return ZERO eval() / exec() builtins (false positives are nn.Module.eval())
rg "\b(eval|exec)\s*\(" arvc/ -n | grep -v "\.eval()"
```

The Colab notebook also auto-verifies the safe_load module is present
after `git clone` (see the "Install Dependencies" cell).
