import importlib
import math

import pytest
import torch
import torch.nn.functional as F


def _distribution_api():
    module = importlib.import_module("prob_models.reparameterized_distribution")
    return (
        module.MaskedStatPool,
        module.ReparameterizedDistributionHead,
        module.antithetic_standard_normal,
    )


def test_masked_stat_pool_ignores_padding_and_handles_single_valid_token():
    MaskedStatPool, _, _ = _distribution_api()
    pool = MaskedStatPool(dim=3, hidden_dim=4)
    tokens = torch.tensor([[[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]])
    mask = torch.tensor([[1, 0]])

    center, dispersion, entropy = pool(tokens, mask)

    torch.testing.assert_close(center, tokens[:, 0])
    torch.testing.assert_close(dispersion, torch.zeros_like(dispersion))
    torch.testing.assert_close(entropy, torch.zeros_like(entropy))


def test_masked_stat_pool_nonfinite_padding_cannot_contaminate_statistics():
    MaskedStatPool, _, _ = _distribution_api()
    pool = MaskedStatPool(dim=3, hidden_dim=4)
    finite = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]]])
    nonfinite = finite.clone()
    nonfinite[:, 2] = torch.tensor([float("nan"), float("inf"), float("-inf")])
    mask = torch.tensor([[1, 1, 0]])

    expected = pool(finite, mask)
    actual = pool(nonfinite, mask)

    for actual_statistic, expected_statistic in zip(actual, expected):
        torch.testing.assert_close(actual_statistic, expected_statistic)


def test_masked_stat_pool_entropy_is_normalized_over_valid_positions():
    MaskedStatPool, _, _ = _distribution_api()
    pool = MaskedStatPool(dim=3, hidden_dim=4)
    with torch.no_grad():
        pool.score.weight.zero_()
    tokens = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    _, _, entropy = pool(tokens, mask)

    torch.testing.assert_close(entropy, torch.ones(2, 1))


def test_masked_stat_pool_rejects_all_invalid_rows():
    MaskedStatPool, _, _ = _distribution_api()
    pool = MaskedStatPool(dim=3, hidden_dim=4)

    with pytest.raises(ValueError, match="at least one valid position"):
        pool(torch.randn(2, 4, 3), torch.zeros(2, 4, dtype=torch.long))


@pytest.mark.parametrize(
    ("tokens", "mask", "message"),
    [
        (torch.randn(2, 3), torch.ones(2, 3), "tokens must have shape"),
        (torch.randn(2, 4, 3), torch.ones(2, 3), "mask must have shape"),
        (torch.randn(2, 4, 5), torch.ones(2, 4), "token dimension"),
    ],
)
def test_masked_stat_pool_validates_tensor_shapes(tokens, mask, message):
    MaskedStatPool, _, _ = _distribution_api()
    pool = MaskedStatPool(dim=3, hidden_dim=4)

    with pytest.raises(ValueError, match=message):
        pool(tokens, mask)


@pytest.mark.parametrize("sample_count", [2, 4, 8])
def test_antithetic_noise_contains_exact_interleaved_pairs(sample_count):
    _, _, antithetic_standard_normal = _distribution_api()
    noise = antithetic_standard_normal(
        2,
        sample_count,
        5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(7),
    )

    assert noise.shape == (2, sample_count, 5)
    paired = noise.reshape(2, sample_count // 2, 2, 5)
    torch.testing.assert_close(paired[:, :, 0], -paired[:, :, 1])


def test_antithetic_noise_is_reproducible_with_local_generator():
    _, _, antithetic_standard_normal = _distribution_api()
    first = antithetic_standard_normal(
        2,
        8,
        5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(11),
    )
    second = antithetic_standard_normal(
        2,
        8,
        5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(11),
    )

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("sample_count", [0, 1, 3, -2])
def test_antithetic_noise_rejects_non_positive_or_odd_sample_counts(sample_count):
    _, _, antithetic_standard_normal = _distribution_api()

    with pytest.raises(ValueError, match="positive even integer"):
        antithetic_standard_normal(
            2,
            sample_count,
            5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_distribution_head_outputs_bounded_logvar_and_unit_samples():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    tokens = torch.randn(3, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])

    output = head(tokens, mask, sample_count=4)

    assert output.center.shape == (3, 8)
    assert output.dispersion.shape == (3, 8)
    assert output.attention_entropy.shape == (3, 1)
    assert output.mean.shape == (3, 8)
    assert output.logvar.shape == (3, 8)
    assert output.samples.shape == (3, 4, 8)
    assert output.anchor_kl.ndim == 0
    assert output.logvar.dtype == torch.float32
    assert torch.all(output.logvar >= -8.0)
    assert torch.all(output.logvar <= 2.0)
    torch.testing.assert_close(
        output.samples.norm(dim=-1),
        torch.ones(3, 4),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.isfinite(output.anchor_kl)


def test_distribution_head_disables_cpu_autocast_for_fp32_probability_path():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    tokens = torch.randn(3, 5, 8)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = head(tokens, torch.ones(3, 5), sample_count=4)

    fp32_outputs = (
        output.center,
        output.dispersion,
        output.attention_entropy,
        output.mean,
        output.logvar,
        output.samples,
        output.anchor_kl,
    )
    assert all(tensor.dtype == torch.float32 for tensor in fp32_outputs)


@pytest.mark.parametrize("component_name", ["pool", "mean_head", "logvar_head"])
def test_distribution_head_rejects_non_fp32_parameters_with_clear_error(component_name):
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    getattr(head, component_name).half()

    with pytest.raises(RuntimeError, match="parameters must remain in FP32"):
        head(
            torch.randn(3, 5, 8, dtype=torch.float16),
            torch.ones(3, 5),
            sample_count=4,
        )


def test_distribution_head_starts_at_deterministic_center_and_anchor_prior():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    prior_std = 0.1
    head = ReparameterizedDistributionHead(dim=6, hidden_dim=12, prior_std=prior_std)

    output = head(torch.randn(2, 4, 6), torch.ones(2, 4), sample_count=4)

    torch.testing.assert_close(output.mean, output.center)
    torch.testing.assert_close(
        output.logvar,
        torch.full_like(output.logvar, math.log(prior_std**2)),
    )
    torch.testing.assert_close(
        output.anchor_kl,
        torch.zeros_like(output.anchor_kl),
        atol=1e-6,
        rtol=0,
    )


def test_distribution_head_anchor_kl_matches_closed_form_mean_reduction():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    prior_std = 0.5
    head = ReparameterizedDistributionHead(dim=2, hidden_dim=4, prior_std=prior_std)
    mean_shift = torch.tensor([0.25, -0.5])
    configured_logvar = torch.tensor([-1.0, -2.0])
    with torch.no_grad():
        head.mean_head[-1].bias.copy_(mean_shift)
        head.logvar_head[-1].bias.copy_(configured_logvar)

    output = head(torch.randn(3, 4, 2), torch.ones(3, 4), sample_count=4)
    prior_variance = prior_std**2
    expected = (
        0.5
        * (
            (configured_logvar.exp() + mean_shift.square()) / prior_variance
            - 1.0
            + math.log(prior_variance)
            - configured_logvar
        ).mean()
    )

    torch.testing.assert_close(output.anchor_kl, expected)


def test_distribution_head_fixed_noise_is_deterministic():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    tokens = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)
    noise = torch.randn(3, 4, 8)

    first = head(tokens, mask, sample_count=4, noise=noise)
    second = head(tokens, mask, sample_count=4, noise=noise.clone())

    torch.testing.assert_close(first.mean, second.mean)
    torch.testing.assert_close(first.logvar, second.logvar)
    torch.testing.assert_close(first.samples, second.samples)


def test_distribution_head_mean_only_returns_normalized_mean_and_ignores_noise():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    tokens = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)

    output = head(
        tokens,
        mask,
        sample_count=1,
        noise=torch.randn(3, 1, 8),
        mean_only=True,
    )

    assert output.samples.shape == (3, 1, 8)
    torch.testing.assert_close(output.samples[:, 0], F.normalize(output.mean, dim=-1))


def test_distribution_head_reparameterization_reaches_heads_and_tokens():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    torch.manual_seed(17)
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    tokens = torch.randn(3, 5, 8, requires_grad=True)
    output = head(tokens, torch.ones(3, 5), sample_count=4)
    weights = torch.linspace(-1.0, 1.0, output.samples.numel(), dtype=output.samples.dtype).reshape_as(output.samples)

    (output.samples * weights).sum().backward()

    gradients = (
        tokens.grad,
        head.mean_head[-1].weight.grad,
        head.logvar_head[-1].weight.grad,
    )
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_detached_samples_block_matching_gradient_but_anchor_kl_still_trains():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    with torch.no_grad():
        head.mean_head[-1].bias.fill_(0.2)
        head.logvar_head[-1].bias.add_(0.5)
    tokens = torch.randn(3, 5, 8, requires_grad=True)

    output = head(
        tokens,
        torch.ones(3, 5),
        sample_count=4,
        detach_samples=True,
    )

    assert not output.samples.requires_grad
    output.anchor_kl.backward()
    gradients = (
        tokens.grad,
        head.mean_head[-1].bias.grad,
        head.logvar_head[-1].bias.grad,
    )
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_distribution_head_validates_sampling_contract():
    _, ReparameterizedDistributionHead, _ = _distribution_api()
    head = ReparameterizedDistributionHead(dim=8, hidden_dim=16, prior_std=0.1)
    tokens = torch.randn(3, 5, 8)
    mask = torch.ones(3, 5)

    with pytest.raises(ValueError, match="positive even integer"):
        head(tokens, mask, sample_count=3)
    with pytest.raises(ValueError, match="mean-only sampling requires sample_count=1"):
        head(tokens, mask, sample_count=2, mean_only=True)
    with pytest.raises(ValueError, match="sample_count must be an integer"):
        head(tokens, mask, sample_count=True, mean_only=True)
    with pytest.raises(ValueError, match="noise must have shape"):
        head(tokens, mask, sample_count=4, noise=torch.randn(3, 2, 8))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": 8, "hidden_dim": 0},
        {"dim": 8, "prior_std": 0.0},
        {"dim": 8, "logvar_min": 2.0, "logvar_max": -8.0},
        {"dim": 8, "eps": 0.0},
    ],
)
def test_distribution_head_validates_constructor_arguments(kwargs):
    _, ReparameterizedDistributionHead, _ = _distribution_api()

    with pytest.raises(ValueError):
        ReparameterizedDistributionHead(**kwargs)
