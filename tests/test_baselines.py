import torch

from atlaswm.baselines import (
    CovarianceRegularizer,
    FullGaussianMMDRegularizer,
    ZeroRegularizer,
)


def test_matched_baselines_return_scalar_gradients():
    latent = torch.randn(16, 4, requires_grad=True)
    values = [
        ZeroRegularizer()(latent),
        CovarianceRegularizer()(latent),
        FullGaussianMMDRegularizer(beta=0.7, pair_chunk_size=5)(latent),
    ]
    for value in values:
        assert value.ndim == 0 and torch.isfinite(value)
    sum(values).backward()
    assert torch.isfinite(latent.grad).all()
