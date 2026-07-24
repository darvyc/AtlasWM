"""AtlasWM: structured characteristic-function regularization for JEPA world models."""

__version__ = "1.0.0"

from atlaswm.encoder import ViTEncoder
from atlaswm.model import AtlasWM
from atlaswm.planning import CEMPlanner
from atlaswm.predictor import Predictor
from atlaswm.regularizer import AtlasReg, AtlasRegConfig
from atlaswm.statistics import (
    empirical_cf_discrepancy,
    gaussian_bhep_discrepancy,
    gaussian_bhep_null_floor,
    henze_zirkler_beta,
    spherical_projection_even_moment_coefficient,
    student_t_unit_variance_scale,
)

__all__ = [
    "AtlasWM",
    "AtlasReg",
    "AtlasRegConfig",
    "ViTEncoder",
    "Predictor",
    "CEMPlanner",
    "empirical_cf_discrepancy",
    "gaussian_bhep_discrepancy",
    "gaussian_bhep_null_floor",
    "henze_zirkler_beta",
    "spherical_projection_even_moment_coefficient",
    "student_t_unit_variance_scale",
    "__version__",
]
