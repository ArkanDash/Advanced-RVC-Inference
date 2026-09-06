"""
System-level services for Advanced RVC Inference.

This subpackage contains service-layer modules that handle process lifecycle,
system utilities, and model file management.

Modules:
- ``process``      — process management, archive helpers, file movement
- ``restart``      — application restart, language/theme switching
- ``model_utils``  — model inspection (model_info), fusion, ONNX export
- ``utils``        — generic helpers (stop_pid, google_translate, etc.)

For backward compatibility, every public symbol from each module is also
re-exported at the subpackage level.
"""

from . import model_utils, process, restart, utils
from .model_utils import *    # noqa: F401, F403
from .process import *        # noqa: F401, F403
from .restart import *        # noqa: F401, F403
from .utils import *          # noqa: F401, F403


# ── strtobool shim ─────────────────────────────────────────────────────────
# Multiple training modules do `from arvc.utils import strtobool` (see
# arvc/rvc/training/runner/train.py, arvc/rvc/training/extract/extract.py,
# arvc/rvc/training/preprocess/preprocess.py, arvc/uvr/separate_music.py).
#
# `distutils.util.strtobool` was the original source — but `distutils` was
# removed in Python 3.12 (PEP 632). Rather than forcing every caller to
# change their import line, we expose a shim here that behaves identically
# to the legacy `distutils.util.strtobool`:
#   - Returns 1 for truthy strings ("y", "yes", "t", "true", "on", "1")
#   - Returns 0 for falsy strings ("n", "no", "f", "false", "off", "0")
#   - Accepts bool / int / float directly
#   - Raises ValueError for unknown values
#
# Note: callers always wrap it with `bool(strtobool(x))`, so returning int
# is fine and matches the historical contract.
def strtobool(val):
    """Convert a string representation of truth to 1 (true) or 0 (false).

    This is a re-implementation of `distutils.util.strtobool` (removed in
    Python 3.12) so that training/preprocess/extract CLI scripts that do
    `from arvc.utils import strtobool` continue to work.

    Args:
        val: String, bool, int, or float to interpret.

    Returns:
        1 if val represents truth, 0 otherwise.

    Raises:
        ValueError: if `val` is a string that doesn't match any known
            truthy/falsy token.
    """
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 1 if val else 0
    if val is None:
        return 0
    val_str = str(val).strip().lower()
    if val_str in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val_str in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {val!r}")


__all__ = ["model_utils", "process", "restart", "utils", "strtobool"]
