"""AtlasWM: stable end-to-end JEPA world models.

Public API:
    AtlasWM        — full end-to-end model
    AtlasReg       — the anti-collapse regularizer
    AtlasRegConfig — its configuration
    ViTEncoder     — standalone encoder
    Predictor      — standalone predictor
    CEMPlanner     — latent-space planner
"""

__version__ = "0.1.0"

from atlaswm.model import AtlasWM
from atlaswm.regularizer import AtlasReg, AtlasRegConfig
from atlaswm.encoder import ViTEncoder
from atlaswm.predictor import Predictor
from atlaswm.planning import CEMPlanner

__all__ = [
    "AtlasWM",
    "AtlasReg",
    "AtlasRegConfig",
    "ViTEncoder",
    "Predictor",
    "CEMPlanner",
    "__version__",
]
