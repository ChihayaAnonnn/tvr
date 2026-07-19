import copy
from dataclasses import fields

import pytest
import torch

from modules.stochastic_prototype_ranking import RSPRCore, RSPROutput


def _core(*, eval_seed: int = 31) -> RSPRCore:
    return RSPRCore(
        dim=4,
        sample_count=4,
        eval_sample_count=6,
        match_temperature=0.3,
        rank_temperature=0.4,
        hard_negative_count=1,
        prior_std=0.2,
        hard_max=False,
        eval_seed=eval_seed,
        hidden_dim=8,
    )


def _inputs(*, requires_grad: bool = False):
    torch.manual_seed(9)
    text_tokens = torch.randn(3, 3, 4, requires_grad=requires_grad)
    video_tokens = torch.randn(3, 4, 4, requires_grad=requires_grad)
    text_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    video_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]])
    return text_tokens, text_mask, video_tokens, video_mask


def _noise(batch_size: int, sample_count: int, dim: int, offset: float) -> torch.Tensor:
    return torch.linspace(
        -1.3 + offset,
        1.1 + offset,
        steps=batch_size * sample_count * dim,
    ).reshape(batch_size, sample_count, dim)


def _assert_nonzero_finite_gradients(core: RSPRCore, text_tokens: torch.Tensor, video_tokens: torch.Tensor) -> None:
    gradients = (
        core.text_distribution.mean_head[-1].weight.grad,
        core.text_distribution.logvar_head[-1].weight.grad,
        core.video_distribution.mean_head[-1].weight.grad,
        core.video_distribution.logvar_head[-1].weight.grad,
        text_tokens.grad,
        video_tokens.grad,
    )
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_rspro_output_exposes_only_core_contract_fields():
    assert tuple(field.name for field in fields(RSPROutput)) == (
        "text_distribution",
        "video_distribution",
        "probabilistic_logits",
        "pair_uncertainty",
        "stochastic_pair_scores",
        "anchor_kl",
    )


def test_core_uses_independent_distribution_parameters_and_averages_anchor_kl():
    core = _core()
    text_parameter_ids = {id(parameter) for parameter in core.text_distribution.parameters()}
    video_parameter_ids = {id(parameter) for parameter in core.video_distribution.parameters()}
    assert text_parameter_ids.isdisjoint(video_parameter_ids)

    output = core(*_inputs(), sample_count=4)

    torch.testing.assert_close(
        output.anchor_kl,
        0.5 * (output.text_distribution.anchor_kl + output.video_distribution.anchor_kl),
    )
    assert output.probabilistic_logits.shape == (3, 3)
    assert output.pair_uncertainty.shape == (3, 3)
    assert output.stochastic_pair_scores.shape == (3, 3, 4)


def test_probabilistic_logits_backpropagate_to_both_modalities_and_tokens():
    core = _core()
    text_tokens, text_mask, video_tokens, video_mask = _inputs(requires_grad=True)
    output = core(
        text_tokens,
        text_mask,
        video_tokens,
        video_mask,
        sample_count=4,
        text_noise=_noise(3, 4, 4, 0.0),
        video_noise=_noise(3, 4, 4, 0.35),
    )

    output.probabilistic_logits.mean().backward()

    _assert_nonzero_finite_gradients(core, text_tokens, video_tokens)


def test_stochastic_rank_loss_backpropagates_to_both_modalities_and_tokens():
    core = _core()
    text_tokens, text_mask, video_tokens, video_mask = _inputs(requires_grad=True)
    output = core(
        text_tokens,
        text_mask,
        video_tokens,
        video_mask,
        sample_count=4,
        text_noise=_noise(3, 4, 4, -0.15),
        video_noise=_noise(3, 4, 4, 0.5),
    )

    rank_output = core.rank_loss(
        output.stochastic_pair_scores,
        torch.tensor([0, 1, 2]),
        output.probabilistic_logits,
    )
    rank_output.loss.backward()

    _assert_nonzero_finite_gradients(core, text_tokens, video_tokens)


def test_detach_samples_blocks_matching_and_rank_gradients_but_anchor_kl_remains_trainable():
    core = _core()
    with torch.no_grad():
        core.text_distribution.mean_head[-1].bias.fill_(0.15)
        core.video_distribution.mean_head[-1].bias.fill_(0.2)
        core.text_distribution.logvar_head[-1].bias.add_(0.3)
        core.video_distribution.logvar_head[-1].bias.add_(0.4)
    text_tokens, text_mask, video_tokens, video_mask = _inputs(requires_grad=True)
    output = core(
        text_tokens,
        text_mask,
        video_tokens,
        video_mask,
        sample_count=4,
        detach_samples=True,
        text_noise=_noise(3, 4, 4, 0.1),
        video_noise=_noise(3, 4, 4, -0.2),
    )
    rank_output = core.rank_loss(
        output.stochastic_pair_scores,
        torch.tensor([0, 1, 2]),
        output.probabilistic_logits,
    )

    assert not output.text_distribution.samples.requires_grad
    assert not output.video_distribution.samples.requires_grad
    assert not output.probabilistic_logits.requires_grad
    assert not rank_output.loss.requires_grad
    assert all(parameter.grad is None for parameter in core.parameters())

    output.anchor_kl.backward()

    _assert_nonzero_finite_gradients(core, text_tokens, video_tokens)


def test_eval_fixed_noise_is_antithetic_repeatable_and_survives_state_dict_round_trip():
    core = _core()
    assert core.fixed_text_noise.shape == (6, 4)
    assert core.fixed_video_noise.shape == (6, 4)
    torch.testing.assert_close(
        core.fixed_text_noise.reshape(3, 2, 4)[:, 0],
        -core.fixed_text_noise.reshape(3, 2, 4)[:, 1],
    )
    torch.testing.assert_close(
        core.fixed_video_noise.reshape(3, 2, 4)[:, 0],
        -core.fixed_video_noise.reshape(3, 2, 4)[:, 1],
    )

    core.eval()
    inputs = _inputs()
    first = core(*inputs, sample_count=4)
    second = core(*inputs, sample_count=4)
    restored = _core(eval_seed=999)
    restored.load_state_dict(copy.deepcopy(core.state_dict()))
    restored.eval()
    loaded = restored(*inputs, sample_count=4)

    for first_value, second_value, loaded_value in zip(
        (
            first.text_distribution.samples,
            first.video_distribution.samples,
            first.probabilistic_logits,
            first.pair_uncertainty,
        ),
        (
            second.text_distribution.samples,
            second.video_distribution.samples,
            second.probabilistic_logits,
            second.pair_uncertainty,
        ),
        (
            loaded.text_distribution.samples,
            loaded.video_distribution.samples,
            loaded.probabilistic_logits,
            loaded.pair_uncertainty,
        ),
    ):
        torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
        torch.testing.assert_close(first_value, loaded_value, rtol=0, atol=0)


def test_eval_seed_uses_local_noise_generators_without_changing_global_rng_progression():
    torch.manual_seed(71)
    first = _core(eval_seed=4)
    first_next_random = torch.randn(5)
    torch.manual_seed(71)
    second = _core(eval_seed=48)
    second_next_random = torch.randn(5)

    torch.testing.assert_close(first_next_random, second_next_random, rtol=0, atol=0)
    assert not torch.equal(first.fixed_text_noise, second.fixed_text_noise)
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def test_explicit_eval_noise_takes_precedence_over_fixed_buffers():
    core = _core()
    core.eval()
    text_tokens, text_mask, video_tokens, video_mask = _inputs()
    text_noise = _noise(3, 4, 4, 0.8)
    video_noise = _noise(3, 4, 4, -0.6)

    output = core(
        text_tokens,
        text_mask,
        video_tokens,
        video_mask,
        sample_count=4,
        text_noise=text_noise,
        video_noise=video_noise,
    )
    expected_text = core.text_distribution(
        text_tokens,
        text_mask,
        sample_count=4,
        noise=text_noise,
    )
    expected_video = core.video_distribution(
        video_tokens,
        video_mask,
        sample_count=4,
        noise=video_noise,
    )

    torch.testing.assert_close(output.text_distribution.samples, expected_text.samples)
    torch.testing.assert_close(output.video_distribution.samples, expected_video.samples)


def test_core_enforces_sampling_and_eval_noise_contracts():
    core = _core()
    inputs = _inputs()
    core.eval()

    mean_only = core(*inputs, sample_count=1, mean_only=True)
    assert mean_only.stochastic_pair_scores.shape == (3, 3, 1)
    with pytest.raises(ValueError, match="positive even integer"):
        core(*inputs, sample_count=1)
    with pytest.raises(ValueError, match="eval_sample_count"):
        core(*inputs, sample_count=8)
    with pytest.raises(ValueError, match="noise must have shape"):
        core(*inputs, sample_count=4, text_noise=torch.randn(3, 2, 4))
    with pytest.raises(ValueError, match="positive even integer"):
        RSPRCore(dim=4, sample_count=4, eval_sample_count=3)
