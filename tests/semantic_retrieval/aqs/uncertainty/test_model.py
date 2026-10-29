import pytest
import torch


@pytest.mark.parametrize('enabled,eta',[(False,.1),(True,0.)])
@pytest.mark.parametrize('aqs_active',[False,True])
def test_disabled_exact(tiny_clip,paired_batch,fusion_factory,enabled,eta,aqs_active,monkeypatch):
    from semantic_retrieval.aqs.model import AqsBaselineModel
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    aqs=AqsConfig(True,2,.2 if aqs_active else 0)
    old=AqsBaselineModel(tiny_clip.config,tiny_clip.state_dict(),aqs=aqs,id_mapping_sha256='a'*64,temporal_layers=1)
    new=fusion_factory(aqs=aqs,fusion=FusionConfig(enabled,eta));new.load_state_dict(old.state_dict(),strict=True)
    def forbidden(*a,**k): raise AssertionError('disabled fusion ran')
    monkeypatch.setattr('semantic_retrieval.aqs.fusion_core.symmetric_kl',forbidden)
    rng=torch.get_rng_state().clone()
    expected=old(*paired_batch);expected['loss'].backward();end=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual=new(*paired_batch);actual['loss'].backward()
    assert actual.keys()==expected.keys() and torch.equal(end,torch.get_rng_state())
    for k in actual:
        if isinstance(actual[k],torch.Tensor): torch.testing.assert_close(actual[k],expected[k],atol=0,rtol=0)
        else: assert actual[k]==expected[k]
    for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
        assert name==other
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    old.eval();new.eval()
    with torch.no_grad():
        text=old.encode_text(*paired_batch[:2]);video=old.encode_video(*paired_batch[2:4])
        torch.testing.assert_close(new.score(text,video),old.score(text,video),atol=0,rtol=0)


@pytest.mark.parametrize('aqs_active',[False,True])
def test_active_loss_and_isolated_gradients(fusion_factory,paired_batch,aqs_active):
    from semantic_retrieval.aqs.config import AqsConfig
    model=fusion_factory(aqs=AqsConfig(True,2,.2 if aqs_active else 0))
    out=model(*paired_batch)
    torch.testing.assert_close(out['loss'],out['contrastive']+out['mil']+out['kl']+model.aqs_config.weight*out['aqs'],atol=0,rtol=0)
    params=[model.core.uncertain_net_text.fc.weight,model.core.uncertain_net_video.fc.weight,
            model.core.pie_net_text.fc.weight,model.core.pie_net_video.fc.weight,
            model.clip.text_projection,model.clip.visual.proj,model.core.frame_position_embeddings.weight]
    objective=out['aqs'] if aqs_active else out['contrastive']
    grads=torch.autograd.grad(objective,params)
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    assert model.clip.visual.conv1.weight.grad is None
    assert all(isinstance(v,float) for v in out['fusion_stats'].values())
    model.eval()
    with torch.no_grad():
        text=model.encode_text(*paired_batch[:2]);video=model.encode_video(*paired_batch[2:4])
        prepared=model.core.prepare_modalities(text,video,gather=False)
        score=model.core.score_prepared(*prepared)
        torch.testing.assert_close(model.score(text,video),score['logits'])
        assert not torch.equal(score['logits'],score['wti']*model.clip.logit_scale.exp())


def test_variance_changes_selected_id():
    from semantic_retrieval.aqs.fusion_math import score_gaussian,symmetric_kl
    from semantic_retrieval.aqs.selection import select_negatives
    means=torch.zeros(3,2); wti=torch.ones(3,3); reps=torch.eye(3); ids=torch.arange(3)
    text=score_gaussian(means,torch.zeros_like(means))
    choices=[]
    for a in (torch.tensor([[0.,0.],[0.,0.],[1.,1.]]),torch.tensor([[0.,0.],[1.,1.],[0.,0.]])):
        score=wti-.1*symmetric_kl(text,score_gaussian(means,a))
        choices.append(select_negatives(score.detach(),reps,ids,1).selected_ids[0,0].item())
    assert choices==[1,2]
