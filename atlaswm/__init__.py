"""AtlasWM public API."""

__version__ = "2.0.0"

from atlaswm.baselines import (
    CovarianceRegularizer,
    FullGaussianMMDRegularizer,
    ZeroRegularizer,
)
from atlaswm.data import (
    ToyEnvConfig,
    ToyTrajectoryDataset,
    ToyVisualEnv,
    TrajectoryArrayDataset,
    TrajectoryManifestDataset,
    TrajectoryNPZDataset,
    split_by_trajectory,
)
from atlaswm.diagnostics import covariance_error, effective_rank, latent_diagnostics
from atlaswm.encoder import ViTEncoder
from atlaswm.evaluation import (
    evaluate_loader,
    evaluate_prediction_batch,
    evaluate_toy_control,
    evaluate_state_probe_loader,
    held_out_linear_probe,
)
from atlaswm.model import AtlasWM
from atlaswm.planning import CEMPlanner, PlanResult
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
from atlaswm.training import Trainer, TrainerConfig, TrainState

__all__ = [
    "AtlasWM",
    "AtlasReg",
    "AtlasRegConfig",
    "ViTEncoder",
    "Predictor",
    "CEMPlanner",
    "PlanResult",
    "ToyEnvConfig",
    "ToyVisualEnv",
    "ToyTrajectoryDataset",
    "TrajectoryArrayDataset",
    "TrajectoryManifestDataset",
    "TrajectoryNPZDataset",
    "split_by_trajectory",
    "CovarianceRegularizer",
    "FullGaussianMMDRegularizer",
    "ZeroRegularizer",
    "Trainer",
    "TrainerConfig",
    "TrainState",
    "effective_rank",
    "covariance_error",
    "latent_diagnostics",
    "evaluate_loader",
    "evaluate_prediction_batch",
    "evaluate_toy_control",
    "evaluate_state_probe_loader",
    "held_out_linear_probe",
    "empirical_cf_discrepancy",
    "gaussian_bhep_discrepancy",
    "gaussian_bhep_null_floor",
    "henze_zirkler_beta",
    "spherical_projection_even_moment_coefficient",
    "student_t_unit_variance_scale",
    "__version__",
]
