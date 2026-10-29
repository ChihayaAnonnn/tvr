import copy
import pytest
import torch
import torch.nn.functional as F
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.fusion_config import FusionConfig
from semantic_retrieval.aqs.local_config import LocalQueryConfig
from semantic_retrieval.aqs.model import AqsBaselineModel
from semantic_retrieval.aqs.local_model import build_local_model
from semantic_retrieval.aqs.local_scoring import local_scores


def build(clip, *, enabled=True, beta=.2, aqs=True, **kw):
    return build_local_model(clip.config,clip.state_dict(),local=LocalQueryConfig(enabled,beta,.1),
        fusion=FusionConfig(False,0),aqs=AqsConfig(aqs,1,.2),id_mapping_sha256='a'*64,
        temporal_layers=1,samples=2,query_chunk=1,video_chunk=1,**kw)


@pytest.mark.parametrize('enabled,beta',[(False,.2),(True,0)])
def test_disabled_exact_legacy(local_clip,local_batch,enabled,beta):
    torch.manual_seed(42)
    old=AqsBaselineModel(local_clip.config,local_clip.state_dict(),aqs=AqsConfig(True,1,.2),
        id_mapping_sha256='a'*64,temporal_layers=1,samples=2)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(42); new=build(local_clip,enabled=enabled,beta=beta)
    assert type(new) is AqsBaselineModel and torch.equal(rng,torch.get_rng_state())
    assert old.state_dict().keys()==new.state_dict().keys()
    for k,v in old.state_dict().items(): assert torch.equal(v,new.state_dict()[k])
    torch.manual_seed(91); a=old(*local_batch)
    torch.manual_seed(91); b=new(*local_batch)
    torch.testing.assert_close(a['loss'],b['loss'],rtol=0,atol=0)
    a['loss'].backward(); b['loss'].backward()
    for p,q in zip(old.parameters(),new.parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_local_train_score_masks_and_legacy(local_clip,local_batch):
    model=build(local_clip)
    result=model(*local_batch)
    result['loss'].backward()
    for p in model.local_queries.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    model.zero_grad()
    text=model.encode_text(local_batch[0],local_batch[1]); video=model.encode_video(local_batch[2],local_batch[3])
    local=local_scores(text['local_tokens'],text['local_mask'],video['local_tokens'],video['mask'],tau=.1,query_chunk=1,video_chunk=1)
    F.cross_entropy(local,torch.arange(2)).backward()
    assert model.clip.visual.proj.grad.abs().sum()>0
    for p in model.local_queries.parameters(): assert p.grad.abs().sum()>0
    torch.manual_seed(9); old=model.core.training_components(text,video)
    torch.manual_seed(9); new=model(*local_batch)
    torch.testing.assert_close(old[1],new['mil']); torch.testing.assert_close(old[2],new['kl'])
    model.eval()
    with torch.no_grad():
        text=model.encode_text(local_batch[0],local_batch[1]); video=model.encode_video(local_batch[2],local_batch[3])
        rng=torch.get_rng_state().clone(); score=model.score(text,video)
        assert torch.equal(rng,torch.get_rng_state())
        global_logits=model.core.get_similarity_logits(text['pooled'],text['tokens'],video['tokens'],text['mask'].long(),video['mask'].long(),shaped=True,loose_type=True)[0]
        local=local_scores(text['local_tokens'],text['local_mask'],video['local_tokens'],video['mask'],tau=.1,query_chunk=1,video_chunk=1)
        torch.testing.assert_close(score,global_logits+model.clip.logit_scale.exp()*.2*local)
        changed=local_batch[2].clone(); changed[~local_batch[3]]=100
        torch.testing.assert_close(score,model.score(text,model.encode_video(changed,local_batch[3])))
        short={k:v[:1] for k,v in text.items()}
        assert model.score(short,video).shape==(1,2)


def test_aqs_off_still_local_and_config_guard(local_clip,local_batch):
    model=build(local_clip,aqs=False)
    result=model(*local_batch)
    assert result['aqs'].item()==0
    result['contrastive'].backward()
    assert model.local_queries.query.grad.abs().sum()>0
    model.local_config=LocalQueryConfig(True,.3,.1)
    with pytest.raises(RuntimeError,match='changed'): model(*local_batch)
    for fusion in (FusionConfig(True,0),FusionConfig(False,.1)):
        with pytest.raises(RuntimeError,match='KL'):
            build_local_model(local_clip.config,local_clip.state_dict(),local=LocalQueryConfig(True,.2,.1),
                fusion=fusion,aqs=AqsConfig(True,1,.2),id_mapping_sha256='a'*64,temporal_layers=1)


def test_selection_reads_fused_raw_and_keeps_loss_gradients(local_clip,local_batch,monkeypatch):
    import semantic_retrieval.aqs.local_model as module
    from semantic_retrieval.aqs.selection import select_negatives
    model=build(local_clip,beta=100.)
    ids=local_batch[0].repeat(2,1)[:3]; mask=local_batch[1].repeat(2,1)[:3]
    video=local_batch[2].repeat(2,1,1,1,1)[:3]; frames=local_batch[3].repeat(2,1)[:3]
    batch=(ids,mask,video,frames,torch.arange(3))
    fixed=torch.tensor([[0.,1.,-1.],[-1.,0.,1.],[1.,-1.,0.]],requires_grad=True)
    observed=[]
    real=module.select_negatives
    def selection(raw,reps,identities,n):
        out=real(raw,reps,identities,n)
        observed.append((raw,reps,identities,out))
        return out
    monkeypatch.setattr(module,'local_scores',lambda *a,**kw:fixed)
    monkeypatch.setattr(module,'select_negatives',selection)
    result=model(*batch)
    raw,reps,identities,out=observed[0]
    assert not raw.requires_grad
    assert out.indices[:,0].tolist()==[1,2,0]
    torch.testing.assert_close(raw,result['logits'].detach()/model.clip.logit_scale.exp().detach())
    result['aqs'].backward()
    assert fixed.grad is not None and fixed.grad.abs().sum()>0
    assert not out.indices.requires_grad
