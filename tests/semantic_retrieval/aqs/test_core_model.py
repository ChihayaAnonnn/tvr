import ast
import copy
import inspect
import textwrap
import pytest
import torch


def test_extended_core_is_original_algorithm_plus_observer():
    from semantic_retrieval.aqs.core import AqsCore
    from semantic_retrieval.full_data.upstream_uatvr.modeling import UATVR
    old = ast.parse(textwrap.dedent(inspect.getsource(UATVR._loose_similarity)))
    new = ast.parse(textwrap.dedent(inspect.getsource(AqsCore._loose_similarity)))
    method = new.body[0]
    assert len(method.args.kwonlyargs) == 1
    assert method.args.kwonlyargs[0].arg == 'training_observer'
    method.args.kwonlyargs.clear(); method.args.kw_defaults.clear()
    method.body = [node for node in method.body if not (
        isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == 'training_observer')]
    assert ast.dump(new, include_attributes=False) == ast.dump(old, include_attributes=False)


def test_core_components_match_original_and_preserve_state_rng(tiny_clip, paired_batch):
    from semantic_retrieval.aqs.core import AqsCore
    from semantic_retrieval.full_data.models import BaselineModel
    class Extended(BaselineModel):
        @staticmethod
        def _uatvr_core_class(): return AqsCore
    base = BaselineModel('uatvr', tiny_clip.config, tiny_clip.state_dict(), temporal_layers=1)
    ext = Extended('uatvr', tiny_clip.config, tiny_clip.state_dict(), temporal_layers=1)
    ext.load_state_dict(base.state_dict(), strict=True)
    assert base.state_dict().keys() == ext.state_dict().keys()
    text = base.encode_text(*paired_batch[:2]); video = base.encode_video(*paired_batch[2:4])
    rng = torch.get_rng_state().clone()
    expected = base._logits(text, video)
    end = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = ext.core.training_components(text, video)
    for left, right in zip(expected, actual[:3]): torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), end)
    torch.testing.assert_close(actual[0].detach(), actual[3]*ext.clip.logit_scale.exp().detach(), rtol=0, atol=0)
    assert actual[4].shape == (2, tiny_clip.config.embed_dim)
    assert not actual[3].requires_grad and not actual[4].requires_grad
    ext.eval()
    with pytest.raises(ValueError, match='training'):
        ext.core.training_components(text, video)


@pytest.mark.parametrize('enabled,weight', [(False, .2), (True, 0.)])
def test_disabled_matches_baseline_loss_rng_and_gradients(tiny_clip, paired_batch, enabled, weight, monkeypatch):
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.model import AqsBaselineModel
    from semantic_retrieval.full_data.models import BaselineModel
    base = BaselineModel('uatvr', tiny_clip.config, tiny_clip.state_dict(), temporal_layers=1)
    new = AqsBaselineModel(tiny_clip.config, tiny_clip.state_dict(), aqs=AqsConfig(enabled, 2, weight),
                           id_mapping_sha256='a'*64, temporal_layers=1)
    new.load_state_dict(base.state_dict(), strict=True)
    def forbidden(*args, **kwargs): raise AssertionError('disabled selection ran')
    monkeypatch.setattr('semantic_retrieval.aqs.model.select_negatives', forbidden)
    rng = torch.get_rng_state().clone()
    expected = base(*paired_batch); expected['loss'].backward()
    end = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual = new(*paired_batch); actual['loss'].backward()
    assert torch.equal(torch.get_rng_state(), end)
    assert actual.keys() == expected.keys()
    for key in actual: torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for (name, p), (other, q) in zip(base.named_parameters(), new.named_parameters()):
        assert name == other
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_active_loss_reaches_encoders_and_inference_unchanged(tiny_clip, paired_batch):
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.model import AqsBaselineModel
    from semantic_retrieval.full_data.models import BaselineModel
    model = AqsBaselineModel(tiny_clip.config, tiny_clip.state_dict(), aqs=AqsConfig(True, 2, .2),
                             id_mapping_sha256='a'*64, temporal_layers=1)
    out = model(*paired_batch)
    torch.testing.assert_close(out['loss'], out['contrastive']+out['mil']+out['kl']+.2*out['aqs'], rtol=0, atol=0)
    parameters = [model.clip.text_projection, model.clip.visual.proj, model.core.frame_position_embeddings.weight,
                  model.clip.logit_scale]
    grads = torch.autograd.grad(out['aqs'], parameters, retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    out['loss'].backward()
    assert model.clip.visual.conv1.weight.grad is None
    assert model.core.uncertain_net_text.fc.weight.grad.abs().sum() > 0
    assert out['aqs_stats']['mean_negatives'] == 1
    base = BaselineModel('uatvr', tiny_clip.config, tiny_clip.state_dict(), temporal_layers=1)
    base.load_state_dict(model.state_dict(), strict=True)
    model.eval(); base.eval()
    with torch.no_grad():
        text, video = model.encode_text(*paired_batch[:2]), model.encode_video(*paired_batch[2:4])
        before = torch.get_rng_state().clone()
        score = model.score(text, video)
        torch.testing.assert_close(score, base.score(text, video), rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), before)


def test_model_rejects_wrong_policy_and_global_duplicates(tiny_clip, paired_batch):
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.model import AqsBaselineModel
    with pytest.raises(ValueError):
        AqsBaselineModel(tiny_clip.config, tiny_clip.state_dict(), aqs=AqsConfig(True, 2, .2),
                         id_mapping_sha256='a'*64, temporal_layers=1, batching='caption_shuffle_v1')
    model = AqsBaselineModel(tiny_clip.config, tiny_clip.state_dict(), aqs=AqsConfig(True, 2, .2),
                             id_mapping_sha256='a'*64, temporal_layers=1)
    with pytest.raises(ValueError, match='unique'):
        model(*paired_batch[:-1], torch.tensor([1, 1]))


def test_active_gradient_layout_guard_preserves_values():
    from semantic_retrieval.aqs.core import _ContiguousBackward
    value = torch.ones(3, 4, requires_grad=True)
    upstream = torch.arange(12.).reshape(4, 3).T
    assert not upstream.is_contiguous()
    result = torch.autograd.grad(_ContiguousBackward.apply(value), value, upstream)[0]
    assert result.is_contiguous()
    torch.testing.assert_close(result, upstream, rtol=0, atol=0)
