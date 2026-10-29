import json
import pytest
import torch
from semantic_retrieval.aqs.local_final import load_frozen_model,validate_final_scope
from semantic_retrieval.aqs.local_training import save_local_state
from semantic_retrieval.full_data.training import make_optimizer
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.data.episodes import file_sha256
from experiment_artifacts import ArtifactArrayStore
from .test_model import build
from .test_resume_runtime import binding_for


def test_frozen_model_only_load_exact_and_reject_wrong_epoch(tmp_path,local_clip):
    original=build(local_clip);opt=make_optimizer(original,learning_rate=5e-5,total_steps=10)
    (tmp_path/'checkpoints').mkdir();path=tmp_path/'checkpoints/epoch-03.pt'
    save_local_state(path,original,opt,binding_for(original),dict(epoch=3,batch_cursor=0,update=7731),ArtifactArrayStore(tmp_path/'arrays'))
    frozen=dict(checkpoint=dict(path=str(path),sha256=file_sha256(path),metadata_sha256=file_sha256(path.with_suffix('.json')),local_sha256=file_sha256(path.with_suffix('.local.json'))),selected_epoch=3,model_sha256=model_sha256(original))
    fresh=build(local_clip);rng=torch.get_rng_state().clone()
    load_frozen_model(fresh,frozen)
    assert model_sha256(fresh)==model_sha256(original) and torch.equal(rng,torch.get_rng_state())
    before=model_sha256(fresh)
    with pytest.raises(ValueError):load_frozen_model(fresh,dict(frozen,selected_epoch=4))
    assert model_sha256(fresh)==before
    with pytest.raises(ValueError):load_frozen_model(fresh,dict(frozen,model_sha256='a'*64))
    assert model_sha256(fresh)==before


def test_test_scope_is_explicit_and_one_checkpoint():
    frozen=dict(parent_run_id='train',selected_epoch=3,checkpoint={'sha256':'a'*64})
    m=dict(study='uatvr-aqs-local-query-training-v1',dataset='msrvtt',stage='final',test_authorized=True,validation_authorized=False,parent_run_ids=['train'],config={'evaluation':{'mode':'test','dsl':False,'frozen':frozen}},environment={'gpu_count':2,'precision':'fp32'})
    validate_final_scope(m,{'authorized':True,'frozen':frozen},'test')
    for key,value in [('authorized',False),('frozen',dict(frozen,selected_epoch=4))]:
        with pytest.raises(ValueError):validate_final_scope(m,{'authorized':True,'frozen':frozen,**{key:value}},'test')


def test_cached_inference_checks_masks_chunks_and_all_state(local_clip,local_batch):
    from semantic_retrieval.aqs.local_final import check_inference
    model=build(local_clip).eval();model.requires_grad_(False)
    batch=tuple(x.repeat((8,)+(1,)*(x.ndim-1)) for x in local_batch)
    before=model_sha256(model);rng=torch.get_rng_state().clone()
    checks=check_inference(model,batch)
    assert checks==dict(mask=True,chunks=True,model_unchanged=True,rng_unchanged=True)
    assert model_sha256(model)==before and torch.equal(rng,torch.get_rng_state())
