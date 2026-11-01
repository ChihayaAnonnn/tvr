"""Four-rank Gaussian KL launch contracts; all checks here are CPU-only."""
import copy
import subprocess
from dataclasses import asdict
from pathlib import Path
import pytest
from semantic_retrieval.aqs.config import AqsConfig
from semantic_retrieval.aqs.fusion_config import FusionConfig
from semantic_retrieval.full_data.campaign import FORMAL


def manifest():
    recipe=dict(epochs=5,global_batch=64,learning_rate=5e-5,seed=713,world_size=4,
                negatives=16,weight=.2,validation_only=True)
    return dict(study='uatvr-aqs-kl-training-v1',dataset='synthetic',test_authorized=False,
        environment=dict(gpu_count=4,precision='fp32'),config=dict(
        model=dict(architecture='uatvr-aqs-kl-v1',clip=asdict(FORMAL),fusion=FusionConfig(True,.1).recipe()),
        loss=dict(aqs=AqsConfig(True,16,.2).recipe()),optimizer=dict(kind='pinned_bert_adam'),
        scheduler=dict(kind='warmup_cosine'),evaluation=dict(mode='qualify',dsl=False,training=recipe)))


def test_four_rank_contract_and_global_batch():
    from semantic_retrieval.aqs.fusion_launch import validate_launch,rank_batches
    selected=validate_launch(manifest(),'qualify')
    assert selected['fusion'].active and selected['recipe']['global_batch']//4==16
    batches=[list(range(64)),list(range(64,88))]
    ranks=[rank_batches(batches,r,4) for r in range(4)]
    assert all([len(b) for b in shard]==[16,6] for shard in ranks)
    for i,b in enumerate(batches):assert sum([s[i] for s in ranks],[])==b
    with pytest.raises(ValueError):rank_batches([[1,2,3]],0,4)
    with pytest.raises(ValueError):rank_batches(batches,0,2)


@pytest.mark.parametrize('defect',['test','world','batch','eta','architecture','fixture','mode'])
def test_refuse_wrong_experiment(defect):
    from semantic_retrieval.aqs.fusion_launch import validate_launch
    m=manifest()
    if defect=='test':m['test_authorized']=True
    if defect=='world':m['config']['evaluation']['training']['world_size']=2
    if defect=='batch':m['config']['evaluation']['training']['global_batch']=63
    if defect=='eta':m['config']['model']['fusion']=FusionConfig(True,0).recipe()
    if defect=='architecture':m['config']['model']['architecture']='uatvr-aqs-local-query-v1'
    if defect=='fixture':m['config']['model']['clip']['fixture']=True
    if defect=='mode':m['config']['evaluation']['mode']='train'
    with pytest.raises(ValueError):validate_launch(m,'qualify')


def test_require_matching_archived_four_rank_qualification():
    from semantic_retrieval.aqs.fusion_launch import validate_launch,require_qualification
    s=validate_launch(manifest(),'qualify');args=({'source':'hash'},s['recipe'],s['fusion'].recipe())
    report=dict(passed=True,world_size=4,synthetic=True,test_accessed=False,
                source_files=args[0],recipe=args[1],fusion=args[2])
    require_qualification(report,*args,'completed')
    for key,value in [('world_size',2),('fusion',FusionConfig(True,.2).recipe()),('source_files',{}),('passed',False)]:
        wrong=copy.deepcopy(report);wrong[key]=value
        with pytest.raises(ValueError):require_qualification(wrong,*args,'completed')
    with pytest.raises(ValueError):require_qualification(report,*args,'running')


def test_shell_launch_is_four_rank_and_uses_fusion_module():
    root=Path(__file__).resolve().parents[4]
    command=['bash',str(root/'scripts/run_uatvr_gaussian_kl_4gpu.sh'),'--mode','qualify','--run-dir','/reserved/run','--dry-run']
    import os
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3')
    result=subprocess.run(command,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--nproc_per_node=4' in result.stdout
    assert 'semantic_retrieval.aqs.fusion_launch' in result.stdout
    env['CUDA_VISIBLE_DEVICES']='0,1'
    assert subprocess.run(command,env=env,capture_output=True).returncode!=0


def test_device_ids_and_order_are_bound_to_manifest():
    from semantic_retrieval.aqs.fusion_launch import validate_devices
    registered=['GPU-a','GPU-b','GPU-c','GPU-d'];mapping=dict(zip('0123',registered))
    validate_devices('0,1,2,3',registered,mapping)
    validate_devices(','.join(registered),registered,mapping)
    for wrong in ('0,1','0,1,2,2','3,2,1,0','4,5,6,7'):
        with pytest.raises(ValueError):validate_devices(wrong,registered,mapping)


def _four_rank_worker(rank, rendezvous, directory, config, clip_state):
    import datetime
    import hashlib
    import torch
    import torch.distributed as dist
    from experiment_artifacts import ArtifactArrayStore,artifact_json_bytes
    from semantic_retrieval.aqs.fusion_model import FusionAqsModel
    from semantic_retrieval.aqs.fusion_training import FusionBinding,fusion_recipe,fusion_step,save_fusion_state,restore_fusion_state
    from semantic_retrieval.full_data.training import make_optimizer
    from semantic_retrieval.full_data.qualification import optimizer_fingerprint
    from semantic_retrieval.runtime.checkpoint import model_sha256
    torch.set_num_threads(1);torch.manual_seed(713)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,timeout=datetime.timedelta(seconds=60))
    try:
        def build():
            return FusionAqsModel(config,clip_state,aqs=AqsConfig(True,2,.2),fusion=FusionConfig(True,.1),id_mapping_sha256='a'*64,temporal_layers=1)
        model=build();opt=make_optimizer(model,learning_rate=5e-5,total_steps=10)
        network=torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
        ids=torch.tensor([[126,4+i,127,0,0,0,0,0] for i in range(8)])
        videos=torch.randn(8,3,3,16,16,generator=torch.Generator().manual_seed(714))
        batch=(ids,ids!=0,videos,torch.ones(8,3,dtype=torch.bool),torch.arange(8))
        local=tuple(x[rank*2:rank*2+2].clone() for x in batch)
        recipe=dict(fusion_recipe(model),sampler=dict(global_batch=8,world_size=4,seed=713))
        identity=dict(source_sha='a'*40,input_sha256='b'*64,method='uatvr',world_size=4,recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest())
        binding=FusionBinding(identity,recipe,'a'*64)
        fusion_step(model,network,opt,local)
        fusion_step(model,network,opt,local)
        root=Path(directory);path=root/'checkpoints/step2.pt'
        save_fusion_state(path,model,opt,binding,dict(epoch=0,batch_cursor=2,update=2),ArtifactArrayStore(root/'arrays'))
        expected=fusion_step(model,network,opt,local)
        digest=model_sha256(model);opt_digest=optimizer_fingerprint(opt.state_dict())
        fresh=build();resumed=make_optimizer(fresh,learning_rate=5e-5,total_steps=10)
        ddp=torch.nn.parallel.DistributedDataParallel(fresh,find_unused_parameters=True)
        progress=restore_fusion_state(path,fresh,resumed,binding)
        assert progress==dict(epoch=0,batch_cursor=2,update=2)
        assert fusion_step(fresh,ddp,resumed,local)==expected
        assert model_sha256(fresh)==digest
        assert optimizer_fingerprint(resumed.state_dict())==opt_digest
        hashes=[None]*4;dist.all_gather_object(hashes,digest)
        assert len(set(hashes))==1
    finally:dist.destroy_process_group()


def test_four_cpu_ranks_train_and_resume_exactly(tmp_path,tiny_clip):
    import torch.multiprocessing as mp
    (tmp_path/'checkpoints').mkdir()
    mp.spawn(_four_rank_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path),tiny_clip.config,tiny_clip.state_dict()),nprocs=4,join=True)


def test_outputs_must_use_reserved_directory(tmp_path):
    from semantic_retrieval.aqs.fusion_launch import validate_output_root
    original=tmp_path/'reserved';copied=tmp_path/'copied'
    validate_output_root(original,dict(artifact_path=str(original)))
    with pytest.raises(ValueError):validate_output_root(copied,dict(artifact_path=str(original)))


def test_source_identity_rejects_empty_or_partial_inventory():
    from semantic_retrieval.aqs.fusion_launch import source_inventory,validate_source_inventory
    repo=Path(__file__).resolve().parents[4];complete=source_inventory(repo)
    assert 'src/semantic_retrieval/aqs/fusion_launch.py' in complete
    assert 'scripts/run_uatvr_gaussian_kl_4gpu.sh' in complete
    validate_source_inventory(repo,complete)
    with pytest.raises(ValueError):validate_source_inventory(repo,{})
    partial=dict(complete);partial.pop('src/semantic_retrieval/aqs/fusion_core.py')
    with pytest.raises(ValueError):validate_source_inventory(repo,partial)
