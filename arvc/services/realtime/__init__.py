"""
Realtime inference services for Advanced RVC Inference.

This subpackage contains service-layer modules for realtime voice conversion
(server + client).

Modules:
- ``realtime``        — realtime conversion server (audio capture, VAD,
                       pipeline, output playback)
- ``realtime_client`` — client for connecting to a remote realtime server

For backward compatibility, every public symbol from each module is also
re-exported at the subpackage level.
"""

from . import realtime, realtime_client
from .realtime import *           # noqa: F401, F403
from .realtime_client import *    # noqa: F401, F403

__all__ = ["realtime", "realtime_client"]
