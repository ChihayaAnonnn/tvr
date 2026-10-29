import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from experiment_artifacts import ArtifactArrayStore,artifact_json_bytes
from semantic_retrieval.full_data.training import make_optimizer
from semantic_retrieval.full_data.qualification import optimizer_fingerprint
from semantic_retrieval.runtime.checkpoint import model_sha256,rng_state
from semantic_retrieval.models.clip import ClipConfig
from semantic_retrieval.aqs.local_training import (LocalBinding,local_recipe,local_step,
    save_local_state,restore_local_state,initialize_local_parent)
from semantic_retrieval.aqs.local_runtime import local_spec_from_recipe
from .test_model import build


def binding_for(model,world=1):
    recipe=dict(local_recipe(model),sampler=dict(seed=713,policy='unique_video_all_captions_v1'))
    identity=dict(source_sha='a'*40,input_sha256='b'*64,method='uatvr',world_size=world,
        recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest())
    return LocalBinding(identity,recipe,model.id_mapping_sha256)


def fingerprint(model,opt):
    return model_sha256(model),optimizer_fingerprint(opt.state_dict()),optimizer_fingerprint(rng_state())


def test_resume_exact(tmp_path,local_clip,local_batch):
    model=build(local_clip);opt=make_optimizer(model,learning_rate=5e-5,total_steps=10)
    binding=binding_for(model)
    first=local_step(model,model,opt,local_batch)
    assert first['local_active']==1 and first['query_gradient_norm']>0
    (tmp_path/'checkpoints').mkdir();path=tmp_path/'checkpoints/one.pt'
    progress=dict(epoch=0,batch_cursor=1,update=1)
    save_local_state(path,model,opt,binding,progress,ArtifactArrayStore(tmp_path/'arrays'))
    expected=local_step(model,model,opt,local_batch);end=fingerprint(model,opt)
    fresh=build(local_clip);fresh_opt=make_optimizer(fresh,learning_rate=5e-5,total_steps=10)
    assert restore_local_state(path,fresh,fresh_opt,binding)==progress
    assert local_step(fresh,fresh,fresh_opt,local_batch)==expected
    assert fingerprint(fresh,fresh_opt)==end
    with pytest.raises((FileExistsError,RuntimeError)):
        save_local_state(path,fresh,fresh_opt,binding,progress,ArtifactArrayStore(tmp_path/'arrays'))


@pytest.mark.parametrize('defect',['beta','tau','source','window','architecture','mapping','optimizer','rng','array','fusion'])
def test_reject_before_mutation(tmp_path,local_clip,local_batch,defect,monkeypatch):
    import semantic_retrieval.aqs.local_training as training
    from semantic_retrieval.aqs.local_config import LocalQueryConfig
    model=build(local_clip);opt=make_optimizer(model,learning_rate=5e-5,total_steps=10)
    # Real externalized array, not a JSON tensor; test-only buffer is >1 MiB.
    if defect=='array':model.register_buffer('large_fixture',torch.zeros(262145))
    binding=binding_for(model);local_step(model,model,opt,local_batch)
    (tmp_path/'checkpoints').mkdir();path=tmp_path/'checkpoints/one.pt'
    save_local_state(path,model,opt,binding,dict(epoch=0,batch_cursor=1,update=1),ArtifactArrayStore(tmp_path/'arrays'))
    if defect in ('beta','tau'):model.local_config=LocalQueryConfig(True,.3 if defect=='beta' else .2,.2 if defect=='tau' else .1)
    if defect=='mapping':model.id_mapping_sha256='c'*64
    if defect=='optimizer':opt=make_optimizer(build(local_clip),learning_rate=5e-5,total_steps=10)
    if defect=='source':monkeypatch.setattr(training,'local_source_identity',lambda:dict(changed='c'*64))
    if defect in ('window','architecture'):
        p=path.with_suffix('.local.json');record=json.loads(p.read_text())
        if defect=='window':record['recipe']['base_recipe']['local']['windows']='bad'
        else:record['architecture']='bad'
        p.write_text(json.dumps(record))
    if defect=='fusion':path.with_suffix('.local.json').rename(path.with_suffix('.fusion.json'))
    if defect=='rng':
        payload=torch.load(path,weights_only=True);payload['rng'][0]['torch']=torch.tensor([1])
        torch.save(payload,path);meta=json.loads(path.with_suffix('.json').read_text())
        meta['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.json').write_text(json.dumps(meta))
    if defect=='array':
        arrays=list((tmp_path/'arrays').rglob('*.npy'));assert arrays
        arrays[0].write_bytes(b'corrupted')
    before=fingerprint(model,opt)
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):
        restore_local_state(path,model,opt,binding)
    assert fingerprint(model,opt)==before


def test_parent_initialization_is_not_resume(tmp_path,local_clip):
    from semantic_retrieval.aqs.model import AqsBaselineModel
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.training import CheckpointBinding,model_recipe,save_aqs_state
    old=AqsBaselineModel(local_clip.config,local_clip.state_dict(),aqs=AqsConfig(True,1,.2),id_mapping_sha256='a'*64,temporal_layers=1,samples=2)
    recipe=dict(model=model_recipe(old),sampler=dict(seed=713))
    identity=dict(source_sha='a'*40,input_sha256='b'*64,method='uatvr',world_size=1,
        recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest())
    binding=CheckpointBinding(identity,recipe,'a'*64)
    opt=make_optimizer(old,learning_rate=5e-5,total_steps=10)
    (tmp_path/'checkpoints').mkdir();path=tmp_path/'checkpoints/parent.pt'
    save_aqs_state(path,old,opt,binding,dict(epoch=0,batch_cursor=0,update=0),ArtifactArrayStore(tmp_path/'arrays'))
    model=build(local_clip);query=model.local_queries.query.detach().clone();rng=torch.get_rng_state().clone()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    record=initialize_local_parent(path,model,expected_sha256=digest)
    assert record['sha256']==digest and model.parent_initialization==record
    for k,v in old.state_dict().items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
    assert torch.equal(model.local_queries.query,query) and torch.equal(torch.get_rng_state(),rng)
    before=model_sha256(model)
    with pytest.raises((ValueError,RuntimeError)):initialize_local_parent(path,model,expected_sha256='0'*64)
    assert model_sha256(model)==before


def runtime_recipe(model):
    recipe=local_recipe(model)
    # Pure parser uses production dimensions; no production model is constructed.
    from dataclasses import asdict
    recipe['model']['clip']=asdict(ClipConfig(512,224,768,12,32,77,49408,512,8,12))
    return dict(schema='uatvr-aqs-local-query-software-recipe-v1',local=recipe['local'],fusion=recipe['fusion'],
        aqs=recipe['aqs'],model=recipe['model'],chunks=recipe['chunks'],
        sampler=dict(policy='unique_video_all_captions_v1',global_batch=64,seed=713,epochs=5),
        identity=dict(source_sha='a'*40,input_sha256='b'*64,world_size=2),id_mapping_sha256='a'*64,
        initialization=dict(parent=None,resume=None),dsl=False,test_authorized=False)


def test_offline_parser_and_isolation(local_clip,monkeypatch):
    r=runtime_recipe(build(local_clip))
    def forbidden(*a,**kw):raise AssertionError('parser performed I/O or CUDA work')
    monkeypatch.setattr(Path,'open',forbidden);monkeypatch.setattr(torch.cuda,'init',forbidden)
    spec=local_spec_from_recipe(r)
    assert spec.local.active and spec.architecture=='uatvr-aqs-local-query-v1'
    r['model']['trainable_parameters'].clear()
    assert json.loads(spec.recipe_json)['model']['trainable_parameters']


@pytest.mark.parametrize('defect',['unknown','missing','test','kl','fixture','chunks','initialization','mapping','inactive'])
def test_runtime_strict(local_clip,defect):
    r=runtime_recipe(build(local_clip))
    if defect=='unknown':r['extra']=0
    if defect=='missing':del r['local']['tau']
    if defect=='test':r['test_authorized']=True
    if defect=='kl':r['fusion']['enabled']=True
    if defect=='fixture':r['model']['clip']['fixture']=True
    if defect=='chunks':r['chunks']['query']=True
    if defect=='initialization':r['initialization']=dict(parent={'sha256':'c'*64,'architecture':'uatvr-aqs-v1'},resume={'sha256':'d'*64,'architecture':'uatvr-aqs-local-query-v1'})
    if defect=='mapping':r['id_mapping_sha256']='bad'
    if defect=='inactive':
        r['local']['beta']=0.
        r['model']['trainable_parameters']=[n for n in r['model']['trainable_parameters'] if not n.startswith('local_queries.')]
        assert local_spec_from_recipe(r).architecture=='uatvr-aqs-v1';return
    with pytest.raises(ValueError):local_spec_from_recipe(r)


def test_disabled_factory_never_constructs_local(local_clip,monkeypatch):
    import semantic_retrieval.aqs.local_model as module
    def forbidden(*a,**kw):raise AssertionError('disabled local work')
    monkeypatch.setattr(module,'LocalQueryTokens',forbidden);monkeypatch.setattr(module,'visual_tokens',forbidden)
    model=build(local_clip,beta=0)
    assert type(model).__name__=='AqsBaselineModel'


def test_inactive_recipe_rejects_new_parameter_contract(local_clip):
    r=runtime_recipe(build(local_clip));r['local']['beta']=0.
    with pytest.raises(ValueError,match='inactive'):
        local_spec_from_recipe(r)


def test_single_video_step_accepts_zero_local_gradient(local_clip,local_batch):
    model=build(local_clip);opt=make_optimizer(model,learning_rate=5e-5,total_steps=10)
    result=local_step(model,model,opt,tuple(x[:1] for x in local_batch))
    assert result['query_gradient_norm']==0 and result['contrastive']==0


def test_external_dependencies_unavailable_in_fresh_process(tmp_path):
    import os,subprocess,sys
    script=r'''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(x in fullname.lower() for x in ('laya','querymatch','fusion_core','fusion_math','fusion_model','fusion_training')):
            raise ImportError('excluded external/scoring dependency')
sys.meta_path.insert(0,Block())
import torch
from semantic_retrieval.models.clip import ClipBackbone,ClipConfig
from semantic_retrieval.aqs.local_model import build_local_model
from semantic_retrieval.aqs.local_config import LocalQueryConfig
from semantic_retrieval.aqs.fusion_config import FusionConfig
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.local_training import local_step
from semantic_retrieval.full_data.training import make_optimizer
torch.set_num_threads(1)
clip=ClipBackbone(ClipConfig(64,28,64,1,4,32,128,64,1,1,fixture=True))
model=build_local_model(clip.config,clip.state_dict(),local=LocalQueryConfig(True,.2,.1),fusion=FusionConfig(False,0),aqs=AqsConfig(True,1,.2),id_mapping_sha256='a'*64,temporal_layers=1,samples=2)
ids=torch.tensor([[126,4,127,0],[126,5,127,0]])
frames=torch.randn(2,2,3,28,28);fm=torch.ones(2,2,dtype=torch.bool)
opt=make_optimizer(model,learning_rate=5e-5,total_steps=10)
local_step(model,model,opt,(ids,ids!=0,frames,fm,torch.arange(2)))
model.eval()
with torch.no_grad():assert model.score(model.encode_text(ids,ids!=0),model.encode_video(frames,fm)).shape==(2,2)
assert not any('laya' in x.lower() or 'querymatch' in x.lower() for x in sys.modules)
print('ISOLATED')
'''
    env=dict(os.environ,TMPDIR=str(tmp_path))
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=45,env=env)
    assert result.returncode==0,result.stderr
    assert 'ISOLATED' in result.stdout
