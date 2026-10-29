import copy
import pytest
import torch
from semantic_retrieval.full_data.models import BaselineModel
from semantic_retrieval.aqs.local_visual import visual_tokens


def test_single_pass_cls_patches_and_gradients(local_clip,local_batch):
    base=BaselineModel('uatvr',local_clip.config,local_clip.state_dict(),temporal_layers=1,samples=2)
    visual=base.clip.visual
    other=copy.deepcopy(visual)
    images=local_batch[2].flatten(0,1)
    calls=[]
    handles=[m.register_forward_hook(lambda *args: calls.append(1)) for m in (visual.conv1,visual.transformer)]
    before=list(visual.state_dict())
    cls,patch=visual_tokens(visual,images,video_frame=3)
    for h in handles: h.remove()
    expected=other(images,video_frame=3)
    assert calls==[1,1]
    torch.testing.assert_close(cls,expected,atol=0,rtol=0)
    assert patch.shape==(6,49,64) and not torch.equal(patch[:,0],patch[:,1])
    assert list(visual.state_dict())==before
    cls[:,0].sum().backward(); expected[:,0].sum().backward()
    for (name,p),(name2,q) in zip(visual.named_parameters(),other.named_parameters()):
        assert name==name2 and p.requires_grad==q.requires_grad
        if p.requires_grad: torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    with pytest.raises(ValueError): visual_tokens(visual,images[:1],video_frame=3)
    with pytest.raises(ValueError): visual_tokens(visual,images[:,:,:16,:16],video_frame=3)
