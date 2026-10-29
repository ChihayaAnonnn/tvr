import pytest
import torch


def build_core(tiny_clip):
    from semantic_retrieval.full_data.models import BaselineModel
    from semantic_retrieval.aqs.fusion_core import FusionCore
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    class Model(BaselineModel):
        @staticmethod
        def _uatvr_core_class(): return FusionCore
    model=Model('uatvr',tiny_clip.config,tiny_clip.state_dict(),temporal_layers=1)
    model.core.fusion_config=FusionConfig(True,.1)
    return model


def test_core_legacy_components_and_parameters(tiny_clip,paired_batch):
    from semantic_retrieval.aqs.core import AqsCore
    from semantic_retrieval.full_data.models import BaselineModel
    class Old(BaselineModel):
        @staticmethod
        def _uatvr_core_class(): return AqsCore
    old=Old('uatvr',tiny_clip.config,tiny_clip.state_dict(),temporal_layers=1)
    new=build_core(tiny_clip); new.load_state_dict(old.state_dict(),strict=True)
    text=old.encode_text(*paired_batch[:2]); video=old.encode_video(*paired_batch[2:4])
    originals=({k:v.detach().clone() for k,v in text.items()},{k:v.detach().clone() for k,v in video.items()})
    rng=torch.get_rng_state().clone()
    expected=old.core.training_components(text,video); end=torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual=new.core.fusion_components(text,video)
    torch.testing.assert_close(actual.wti,expected[3],atol=0,rtol=0)
    for left,right in zip((actual.mil,actual.kl),expected[1:3]): torch.testing.assert_close(left,right,atol=0,rtol=0)
    assert torch.equal(end,torch.get_rng_state())
    assert new.state_dict().keys()==old.state_dict().keys()
    for original,current in zip(originals,(text,video)):
        for k in original: assert torch.equal(original[k],current[k])
    prepared_t,prepared_v=new.core.prepare_modalities(text,video,gather=False)
    p=new.core.distribution_parameters(prepared_v,modality='video')
    torch.set_rng_state(rng)
    legacy=new.core.probabilistic_video(prepared_v.pooled,prepared_v.tokens[:,:prepared_v.real_count].contiguous())
    for k in ('mean','logsigma'): torch.testing.assert_close(p[k],legacy[k],atol=0,rtol=0)


def test_eval_rectangular_chunks_no_sampling_or_mutation(tiny_clip,paired_batch,monkeypatch):
    model=build_core(tiny_clip).eval()
    text=model.encode_text(*paired_batch[:2]); video=model.encode_video(*paired_batch[2:4])
    text={k:torch.cat([v,v[:1]]) for k,v in text.items()}
    original=[{k:v.clone() for k,v in x.items()} for x in (text,video)]
    def forbidden(*a,**k): raise AssertionError('eval sampled')
    monkeypatch.setattr('semantic_retrieval.aqs.fusion_core.sample_gaussian_tensors',forbidden)
    rng=torch.get_rng_state().clone()
    with torch.no_grad():
        expected=model.score(text,video)
        chunks=torch.cat([torch.cat([model.score({k:v[q:q+1] for k,v in text.items()},
            {k:v[g:g+1] for k,v in video.items()}) for g in range(2)],dim=1) for q in range(3)])
        torch.testing.assert_close(chunks,expected,atol=2e-6,rtol=2e-4)
        torch.testing.assert_close(model.score(text,video),expected,atol=0,rtol=0)
    assert torch.equal(rng,torch.get_rng_state())
    for before,after in zip(original,(text,video)):
        for k in before: assert torch.equal(before[k],after[k])


def test_eval_permutation_distribution_independence_and_rng(fusion_factory,paired_batch,monkeypatch):
    import random
    import numpy as np
    model=fusion_factory().eval()
    text=model.encode_text(*paired_batch[:2]);video=model.encode_video(*paired_batch[2:4])
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    with torch.no_grad():
        expected=model.score(text,video)
        perm=torch.tensor([1,0])
        actual=model.score({k:v[perm] for k,v in text.items()},{k:v[perm] for k,v in video.items()})
        torch.testing.assert_close(actual[perm][:,perm],expected,atol=2e-6,rtol=2e-4)
        tp,vp=model.core.prepare_modalities(text,video,gather=False)
        single,_=model.core.prepare_modalities({k:v[:1] for k,v in text.items()},video,gather=False)
        p=model.core.distribution_parameters(tp,modality='text')
        ps=model.core.distribution_parameters(single,modality='text')
        for k in p: torch.testing.assert_close(p[k][:1],ps[k],atol=2e-6,rtol=2e-4)
    assert random.getstate()==before[0]
    after=np.random.get_state()
    assert after[0]==before[1][0] and np.array_equal(after[1],before[1][1]) and after[2:]==before[1][2:]
    assert torch.equal(before[2],torch.get_rng_state())


def test_score_rejects_training_and_changed_configuration(fusion_factory,paired_batch):
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    model=fusion_factory()
    text=model.encode_text(*paired_batch[:2]);video=model.encode_video(*paired_batch[2:4])
    with pytest.raises(ValueError,match='evaluation'): model.score(text,video)
    model.eval();model.fusion_config=FusionConfig(True,.2)
    with pytest.raises(ValueError,match='configuration changed'): model.score(text,video)


def test_legacy_sampling_overflow_rejected_inside_component_collective(tiny_clip,paired_batch):
    model=build_core(tiny_clip)
    with torch.no_grad():model.core.uncertain_net_video.fc.bias.fill_(1000.)
    text=model.encode_text(*paired_batch[:2]);video=model.encode_video(*paired_batch[2:4])
    with pytest.raises(RuntimeError,match='nonfinite legacy probability'):
        model.core.fusion_components(text,video)
