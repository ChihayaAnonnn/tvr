import copy

import torch

from modules import module_clip


def _forward_backward(model, value):
    output = model(value)
    output.square().mean().backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return output.detach(), value.grad.detach().clone(), gradients


def test_transformer_checkpointing_preserves_outputs_and_gradients(monkeypatch):
    torch.manual_seed(42)
    baseline = module_clip.Transformer(width=8, layers=2, heads=2).train()
    checkpointed = copy.deepcopy(baseline).train()
    checkpointed.grad_checkpointing = True
    baseline_input = torch.randn(5, 3, 8, requires_grad=True)
    checkpointed_input = baseline_input.detach().clone().requires_grad_(True)

    real_checkpoint = module_clip.checkpoint
    calls = []

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(module_clip, "checkpoint", recording_checkpoint)
    expected_output, expected_input_grad, expected_parameter_grads = (
        _forward_backward(baseline, baseline_input)
    )
    actual_output, actual_input_grad, actual_parameter_grads = (
        _forward_backward(checkpointed, checkpointed_input)
    )

    assert len(calls) == checkpointed.layers
    assert all(call == {"use_reentrant": False} for call in calls)
    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_input_grad, expected_input_grad)
    assert actual_parameter_grads.keys() == expected_parameter_grads.keys()
    for name in actual_parameter_grads:
        torch.testing.assert_close(
            actual_parameter_grads[name], expected_parameter_grads[name]
        )


def test_transformer_checkpointing_is_disabled_during_evaluation(monkeypatch):
    transformer = module_clip.Transformer(width=8, layers=2, heads=2).eval()
    transformer.grad_checkpointing = True

    def fail_checkpoint(*_args, **_kwargs):
        raise AssertionError("evaluation must not use activation checkpointing")

    monkeypatch.setattr(module_clip, "checkpoint", fail_checkpoint)
    output = transformer(torch.randn(5, 3, 8))

    assert output.shape == (5, 3, 8)


def test_transformer_checkpoints_only_requested_leading_layers(monkeypatch):
    transformer = module_clip.Transformer(width=8, layers=3, heads=2).train()
    transformer.grad_checkpointing = True
    transformer.grad_checkpointing_layers = 1
    real_checkpoint = module_clip.checkpoint
    calls = []

    def recording_checkpoint(function, *args, **kwargs):
        calls.append(kwargs)
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(module_clip, "checkpoint", recording_checkpoint)
    value = torch.randn(5, 3, 8, requires_grad=True)
    transformer(value).sum().backward()

    assert calls == [{"use_reentrant": False}]
