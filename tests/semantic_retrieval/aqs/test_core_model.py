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
