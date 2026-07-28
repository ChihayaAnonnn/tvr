"""--clip_gradient_checkpointing has to actually checkpoint something.

The flag was parsed, validated, logged and written to the experiment
manifest, but the code that set grad_checkpointing on the visual transformer
was deleted in the 2026-07-18 cleanup. Nothing noticed, because every run
since then froze eight resblocks and used eight frames at 64 clips per rank,
which fits regardless. The official-parity recipe -- nothing frozen, twelve
frames, 128 clips per rank -- does not, and OOMed on all four ranks with the
non-checkpointed forward in the traceback.
"""

from types import SimpleNamespace

import pytest

from modules.modeling import configure_clip_gradient_checkpointing

VISUAL_LAYERS = 12


def _backbone(text_layers=12, visual_layers=VISUAL_LAYERS):
    """A stand-in with the two attributes the real CLIP exposes."""

    def _transformer(layers):
        return SimpleNamespace(
            layers=layers,
            resblocks=[object()] * layers,
            grad_checkpointing=False,
            grad_checkpointing_layers=layers,
        )

    return SimpleNamespace(
        transformer=_transformer(text_layers),
        visual=SimpleNamespace(transformer=_transformer(visual_layers)),
    )


def test_enabling_it_turns_on_the_requested_visual_layers():
    backbone = _backbone()

    configure_clip_gradient_checkpointing(backbone, enabled=True, visual_layers=12)

    assert backbone.visual.transformer.grad_checkpointing is True
    assert backbone.visual.transformer.grad_checkpointing_layers == 12


def test_a_partial_request_checkpoints_only_that_many_layers():
    backbone = _backbone()

    configure_clip_gradient_checkpointing(backbone, enabled=True, visual_layers=4)

    assert backbone.visual.transformer.grad_checkpointing is True
    assert backbone.visual.transformer.grad_checkpointing_layers == 4


def test_disabling_it_leaves_the_forward_pass_alone():
    backbone = _backbone()

    configure_clip_gradient_checkpointing(backbone, enabled=False, visual_layers=12)

    assert backbone.visual.transformer.grad_checkpointing is False
    assert backbone.visual.transformer.grad_checkpointing_layers == 0


def test_requesting_zero_layers_is_the_same_as_disabling():
    backbone = _backbone()

    configure_clip_gradient_checkpointing(backbone, enabled=True, visual_layers=0)

    assert backbone.visual.transformer.grad_checkpointing is False


def test_the_text_tower_is_never_checkpointed():
    """32 tokens of text is not where the memory goes; recompute is pure cost."""

    backbone = _backbone()

    configure_clip_gradient_checkpointing(backbone, enabled=True, visual_layers=12)

    assert backbone.transformer.grad_checkpointing is False
    assert backbone.transformer.grad_checkpointing_layers == 0


def test_it_reports_how_many_layers_it_turned_on():
    """The count is logged, so a silent no-op is visible in the run log."""

    assert (
        configure_clip_gradient_checkpointing(
            _backbone(), enabled=True, visual_layers=12
        )
        == 12
    )
    assert (
        configure_clip_gradient_checkpointing(
            _backbone(), enabled=False, visual_layers=12
        )
        == 0
    )


@pytest.mark.parametrize("visual_layers", (-1, VISUAL_LAYERS + 1))
def test_it_rejects_a_layer_count_the_backbone_cannot_honour(visual_layers):
    with pytest.raises(ValueError, match="clip_visual_checkpoint_layers"):
        configure_clip_gradient_checkpointing(
            _backbone(), enabled=True, visual_layers=visual_layers
        )


def test_it_refuses_to_silently_skip_a_backbone_it_cannot_configure():
    backbone = SimpleNamespace(transformer=None, visual=None)

    with pytest.raises(RuntimeError, match="visual transformer"):
        configure_clip_gradient_checkpointing(
            backbone, enabled=True, visual_layers=12
        )
