"""
Training-related services for Advanced RVC Inference.

This subpackage contains service-layer modules that orchestrate end-to-end
training tasks (dataset preparation, model training, reference creation).

Modules:
- ``training`` — top-level training orchestration: dataset creation,
                 reference creation, training launch

For backward compatibility, every public symbol from ``training`` is also
re-exported at the subpackage level, so both of these work:

    from arvc.rvc.training.training import create_dataset
    from arvc.rvc.training import create_dataset
"""

from . import training
from .training import *      # noqa: F401, F403

__all__ = ["training"]
