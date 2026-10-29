import hashlib
import json
import pytest
import torch
from experiment_artifacts import ArtifactArrayStore,artifact_json_bytes
from semantic_retrieval.full_data.training import make_optimizer
from semantic_retrieval.full_data.qualification import optimizer_fingerprint
from semantic_retrieval.runtime.checkpoint import rng_state,model_sha256


def binding_for(model,world=1):
    from semantic_retrieval.aqs.fusion_training import FusionBinding,fusion_recipe
    recipe=dict(fusion_recipe(model),sampler=dict(seed=713,policy='unique_video_all_captions_v1'))
    identity=dict(source_sha='a'*40,input_sha256='b'*64,method='uatvr',world_size=world,
                  recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest())
    return FusionBinding(identity,recipe,model.id_mapping_sha256)


def fingerprint(model,optimizer):
    return model_sha256(model),optimizer_fingerprint(optimizer.state_dict()),optimizer_fingerprint(rng_state())


@pytest.mark.parametrize('fusion_active,aqs_active',[(True,True),(True,False),(False,True),(False,False)])
def test_step_fresh_restore_exact(tmp_path,fusion_factory,paired_batch,fusion_active,aqs_active):
    from semantic_retrieval.aqs.fusion_training import fusion_step,save_fusion_state,restore_fusion_state
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    from semantic_retrieval.aqs.config import AqsConfig
    args=dict(fusion=FusionConfig(True,.1 if fusion_active else 0),aqs=AqsConfig(True,2,.2 if aqs_active else 0))
    model=fusion_factory(**args);binding=binding_for(model)
    optimizer=make_optimizer(model,learning_rate=5e-5,total_steps=10)
    before=model.core.pie_net_video.fc.weight.detach().clone()
    first=fusion_step(model,model,optimizer,paired_batch)
    assert first['fusion_active']==float(fusion_active) and first['aqs_active']==float(aqs_active)
    (tmp_path/'checkpoints').mkdir();path=tmp_path/'checkpoints/first.pt';store=ArtifactArrayStore(tmp_path/'arrays')
    progress=dict(epoch=0,batch_cursor=1,update=1)
    save_fusion_state(path,model,optimizer,binding,progress,store)
    expected=fusion_step(model,model,optimizer,paired_batch);end=fingerprint(model,optimizer)
    assert not torch.equal(before,model.core.pie_net_video.fc.weight)
    fresh=fusion_factory(**args);opt=make_optimizer(fresh,learning_rate=5e-5,total_steps=10)
    assert restore_fusion_state(path,fresh,opt,binding)==progress
    assert fusion_step(fresh,fresh,opt,paired_batch)==expected
    assert fingerprint(fresh,opt)==end
    with pytest.raises((FileExistsError,RuntimeError)):
        save_fusion_state(path,fresh,opt,binding,progress,store)


@pytest.mark.parametrize('defect',['eta','source','sidecar','old','optimizer','binding_mutation','mapping','aqs','world'])
def test_restore_refuses_before_mutation(tmp_path,fusion_factory,paired_batch,defect,monkeypatch):
    from semantic_retrieval.aqs.fusion_training import fusion_step,save_fusion_state,restore_fusion_state
    import semantic_retrieval.aqs.fusion_training as training
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    model=fusion_factory();opt=make_optimizer(model,learning_rate=5e-5,total_steps=10);binding=binding_for(model)
    fusion_step(model,model,opt,paired_batch)
    (tmp_path/'checkpoints').mkdir();path=tmp_path/'checkpoints/one.pt'
    save_fusion_state(path,model,opt,binding,dict(epoch=0,batch_cursor=1,update=1),ArtifactArrayStore(tmp_path/'arrays'))
    if defect=='eta': model.fusion_config=FusionConfig(True,.2)
    if defect=='mapping': model.id_mapping_sha256='c'*64
    if defect=='aqs':
        from semantic_retrieval.aqs.config import AqsConfig
        model.aqs_config=AqsConfig(True,3,.2)
    if defect=='world': binding=binding_for(model,world=2)
    if defect=='source':
        original=training.source_identity
        monkeypatch.setattr(training,'source_identity',lambda:dict(original(),changed='c'*64))
    if defect=='sidecar':
        p=path.with_suffix('.fusion.json');r=json.loads(p.read_text());r['architecture']='wrong';p.write_text(json.dumps(r))
    if defect=='old':path.with_suffix('.fusion.json').unlink()
    if defect=='optimizer':opt=make_optimizer(fusion_factory(),learning_rate=5e-5,total_steps=10)
    if defect=='binding_mutation':
        changed=binding.base_recipe;changed['sampler']['seed']=0
        assert binding.base_recipe['sampler']['seed']==713
        return
    before=fingerprint(model,opt)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):restore_fusion_state(path,model,opt,binding)
    assert fingerprint(model,opt)==before
