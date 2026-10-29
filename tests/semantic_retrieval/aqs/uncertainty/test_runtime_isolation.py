import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


def runtime_recipe(model):
    from semantic_retrieval.aqs.fusion_training import fusion_recipe
    r=fusion_recipe(model)
    return dict(schema='uatvr-aqs-kl-software-recipe-v1',fusion=r['fusion'],aqs=r['aqs'],model=r['model'],
        sampler=dict(policy='unique_video_all_captions_v1',global_batch=64,seed=713,epochs=5),
        identity=dict(source_sha='a'*40,input_sha256='b'*64,world_size=2),dsl=False,test_authorized=False)


def test_offline_recipe_strict_immutable_and_no_cuda(fusion_factory,monkeypatch):
    from semantic_retrieval.aqs.fusion_runtime import fusion_spec_from_recipe
    import torch
    recipe=runtime_recipe(fusion_factory())
    def forbidden(*a,**k):raise AssertionError('offline parser tried CUDA')
    monkeypatch.setattr(torch.cuda,'init',forbidden)
    result=fusion_spec_from_recipe(recipe)
    assert result.fusion.active and result.world_size==2
    recipe['model']['trainable_parameters'].clear()
    assert json.loads(result.model_recipe_json)['trainable_parameters']
    assert not result.dsl and not result.test_authorized


@pytest.mark.parametrize('defect',['old','eta','test','dsl','unknown','batch','world','clip','clip_missing','model','sampler','identity','aqs_bool'])
def test_offline_rejects_incompatible_configuration(fusion_factory,defect):
    from semantic_retrieval.aqs.fusion_runtime import fusion_spec_from_recipe
    r=runtime_recipe(fusion_factory())
    if defect=='old':r['schema']='uatvr-aqs-training-v1'
    if defect=='eta':del r['fusion']['eta']
    if defect=='test':r['test_authorized']=True
    if defect=='dsl':r['dsl']='false'
    if defect=='unknown':r['extra']=1
    if defect=='batch':r['sampler']['global_batch']=63
    if defect=='world':r['identity']['world_size']=True
    if defect=='clip':r['model']['clip']['embed_dim']=513
    if defect=='clip_missing':
        r['model']['clip']=dict(embed_dim=512,image_size=224,vision_width=768,vision_layers=12,patch_size=32,context_length=77,vocab_size=49408,text_width=512,text_heads=8,text_layers=12)
    if defect=='model':r['model']['extra']=1
    if defect=='sampler':r['sampler']['extra']=1
    if defect=='identity':r['identity']['extra']=1
    if defect=='aqs_bool':r['aqs']['weight']=True
    with pytest.raises(ValueError):fusion_spec_from_recipe(r)


def test_fresh_process_active_path_isolated(tmp_path):
    root=Path(__file__).resolve().parents[4]
    script=r'''
import os, sys, pathlib, importlib.abc
scratch = pathlib.Path(sys.argv[1]).resolve()
class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        name = fullname.lower()
        if 'laya' in name or 'querymatch' in name or name.startswith('transformers'):
            raise ImportError('external reranker unavailable')
sys.meta_path.insert(0, BlockExternal())
def audit(event, args):
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = pathlib.Path(os.fsdecode(args[0])).resolve()
        text = str(path).lower()
        if 'laya' in text or 'querymatch' in text:
            raise PermissionError('external reranker path unavailable')
        if event == 'open':
            mode = args[1] or ''
            flags = args[2] if len(args)>2 else 0
            writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and path != scratch and scratch not in path.parents and str(path) != '/dev/null':
                raise PermissionError('test writes must stay inside scratch: '+str(path))
sys.addaudithook(audit)
try:
    __import__('semantic_retrieval.laya_rerank')
except ImportError:
    pass
else:
    raise AssertionError('import blocker did not work')
try:
    open(scratch/'artifacts/laya-rerank-validation-v1/forbidden', 'rb')
except PermissionError:
    pass
else:
    raise AssertionError('path blocker did not work')
import torch
from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.fusion_model import FusionAqsModel
from semantic_retrieval.aqs.fusion_config import FusionConfig
from semantic_retrieval.aqs.training import video_id_mapping
torch.set_num_threads(1)
torch.manual_seed(713)
clip = ClipBackbone(ClipConfig(64,16,64,1,4,32,128,64,1,1,fixture=True))
_, digest = video_id_mapping(['v0','v1'])
model = FusionAqsModel(clip.config, clip.state_dict(), fusion=FusionConfig(True,.1), aqs=AqsConfig(True,2,.2), id_mapping_sha256=digest, temporal_layers=1)
ids = torch.tensor([[126,4,127,0],[126,5,127,0]])
video = torch.randn(2,3,3,16,16)
batch = (ids, ids!=0, video, torch.ones(2,3,dtype=torch.bool), torch.arange(2))
model(*batch)['loss'].backward()
disabled = FusionAqsModel(clip.config, clip.state_dict(), fusion=FusionConfig(True,.1), aqs=AqsConfig(False,2,.2), id_mapping_sha256=digest, temporal_layers=1)
disabled(*batch)['loss'].backward()
model.eval()
with torch.no_grad():
    result = model.score(model.encode_text(*batch[:2]), model.encode_video(*batch[2:4]))
assert result.shape == (2,2) and torch.isfinite(result).all()
assert not any('laya' in name.lower() or 'querymatch' in name.lower() for name in sys.modules)
from semantic_retrieval.aqs.fusion_training import fusion_recipe,FusionBinding,fusion_step,save_fusion_state,restore_fusion_state
from semantic_retrieval.full_data.training import make_optimizer
from experiment_artifacts import ArtifactArrayStore,artifact_json_bytes
import hashlib
optimizer=make_optimizer(model,learning_rate=5e-5,total_steps=10)
recipe=fusion_recipe(model)
identity=dict(source_sha='a'*40,input_sha256='b'*64,method='uatvr',world_size=1,recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest())
binding=FusionBinding(identity,recipe,digest)
step=fusion_step(model,model,optimizer,batch)
(scratch/'checkpoints').mkdir()
checkpoint=scratch/'checkpoints/one.pt'
save_fusion_state(checkpoint,model,optimizer,binding,dict(epoch=0,batch_cursor=1,update=1),ArtifactArrayStore(scratch/'arrays'))
restore_fusion_state(checkpoint,model,optimizer,binding)
assert step['fusion_active']==1
print('isolated fusion ok')
'''
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=os.pathsep.join((str(root/'src'),str(root))),TMPDIR=str(tmp_path),OMP_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B','-c',script,str(tmp_path)],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=45)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert 'isolated fusion ok' in result.stdout
