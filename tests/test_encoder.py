import torch

from atlaswm.encoder import ViTEncoder


def test_encoder_has_no_cross_sample_or_future_frame_coupling():
    torch.manual_seed(0)
    encoder = ViTEncoder(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        n_heads=2,
        dropout=0.0,
    )
    encoder.train()
    anchor = torch.randn(1, 3, 16, 16)
    companion_a = torch.randn(1, 3, 16, 16)
    companion_b = torch.randn(1, 3, 16, 16) * 100
    first = encoder(torch.cat((anchor, companion_a), dim=0))[0]
    second = encoder(torch.cat((anchor, companion_b), dim=0))[0]
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)

    trajectory_a = torch.stack((anchor[0], companion_a[0])).unsqueeze(0)
    trajectory_b = torch.stack((anchor[0], companion_b[0])).unsqueeze(0)
    first_frame_a = encoder(trajectory_a)[0, 0]
    first_frame_b = encoder(trajectory_b)[0, 0]
    assert torch.allclose(first_frame_a, first_frame_b, atol=1e-6, rtol=1e-6)


def test_encoder_validates_image_shape():
    encoder = ViTEncoder(img_size=16, patch_size=4, embed_dim=8, depth=0, n_heads=1)
    try:
        encoder(torch.randn(2, 3, 20, 20))
    except ValueError as exc:
        assert "expected image size" in str(exc)
    else:
        raise AssertionError("invalid image size was accepted")
