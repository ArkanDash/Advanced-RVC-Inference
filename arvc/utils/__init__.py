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

__all__ = ["model_utils", "process", "restart", "utils"]
