"""Explicitly authorized, frozen validation-selected JSFUSION final evaluation."""
import csv
import hashlib
from pathlib import Path

import cv2
import torch
from experiment_artifacts import strict_json_load, manifest_sha256, load_validated_run_lifecycle, make_run_paths, artifact_json_bytes
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import safe_path
from semantic_retrieval.runtime.checkpoint import _hydrate_array_references, model_sha256
from .runtime import settings_from_manifest

ENTRY = 'evaluate_full_data_test.py'
CSV = Path('/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv')
VIDEOS = Path('/data2/hxj/data/MSRVTT/videos/all')


def read_test_csv(path, *, expected_count, forbidden_ids):
    with Path(path).open() as f: rows=list(csv.DictReader(f))
    ids=[r['video_id'] for r in rows]
    if (len(rows)!=expected_count or len(set(ids))!=expected_count or set(ids)&set(forbidden_ids)
            or any(not i or Path(i).name!=i for i in ids)
            or any(not isinstance(r['sentence'],str) or not r['sentence'].strip() for r in rows)):
        raise ValueError('incomplete, duplicated or overlapping official test membership')
    return [dict(query_id=r['video_id']+':jsfusion',caption_id='jsfusion',official_video_id=r['video_id'],query_text=r['sentence']) for r in rows]


def load_selected_model(path, model, identity, *, expected_epoch):
    """Inference-only strict load; never restore optimizer/RNG or permit continuation."""
    path=Path(path);safe_path(path);safe_path(path.with_suffix('.json'))
    with path.with_suffix('.json').open() as f: meta=strict_json_load(f)
    if (set(meta)!={'schema_version','identity','sha256','model_sha256'}
            or meta['schema_version']!='full-data-checkpoint-v1' or meta['identity']!=identity
            or file_sha256(path)!=meta['sha256']):raise ValueError('selected checkpoint identity/hash mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if (set(payload)!={'schema_version','identity','progress','model','optimizer','rng'}
            or payload['identity']!=identity or payload['schema_version']!=meta['schema_version']
            or payload['progress']['epoch']!=expected_epoch or payload['progress']['batch_cursor']!=0):
        raise ValueError('selected checkpoint payload/epoch mismatch')
    saved=_hydrate_array_references(payload['model'],path.parent.parent)
    current=model.state_dict()
    if (saved.keys()!=current.keys() or any(not isinstance(saved[k],torch.Tensor) or saved[k].shape!=v.shape
            or saved[k].dtype!=v.dtype or not torch.isfinite(saved[k]).all() for k,v in current.items())):
        raise ValueError('selected checkpoint strict state mismatch')
    digest=hashlib.sha256()
    for name,tensor in sorted(saved.items()):
        digest.update(name.encode());digest.update(str((tuple(tensor.shape),str(tensor.dtype))).encode());digest.update(tensor.contiguous().numpy().tobytes())
    if digest.hexdigest()!=meta['model_sha256']:raise ValueError('selected model hash mismatch')
    model.load_state_dict(saved,strict=True)
    if model_sha256(model)!=meta['model_sha256']:raise ValueError('loaded model differs from selected model')


def final_config(parent, permission, selected):
    return dict(model=parent['config']['model'],loss={},optimizer={},scheduler={},evaluation=dict(
        config_kind='full-data-test-v1',full_data_stage='test_evaluation',eval_split='test',synthetic=False,
        seed=parent['seed'],world_size=2,memory_limit_gib=60,frames=12,text_length=32,dsl=False,
        selection='frozen_validation_earliest_raw_t2v_r1',parent_run_id=parent['run_id'],
        selected_epoch=selected['selected_epoch'],checkpoint=selected['selected_checkpoint'],
        test_csv=permission['test_csv'],video_indices=permission['video_indices'],deterministic_algorithms=True))


def authenticated_parent(repo, permission, parent_id):
    from .authorization import CONTROL, _committed
    from experiment_artifacts import parse_registry_text
    frozen=[r for r in permission['selected'] if r['run_id']==parent_id]
    if len(frozen)!=1:raise ValueError('parent is not a frozen selected baseline')
    frozen=frozen[0]
    records=parse_registry_text(_committed(repo,repo/CONTROL/'registry.jsonl').decode())
    rows=[r for r in records if r['run_id']==parent_id and r['event']=='completed' and r['manifest_sha256']==frozen['manifest_sha256']]
    if len(rows)!=1:raise ValueError('selected parent is not committed completed evidence')
    root=Path(rows[0]['artifact_path']);paths=make_run_paths(root.parents[3],'full-data-retrieval-v1',parent_id)
    parent,status,_=load_validated_run_lifecycle(paths)
    if status['state']!='completed' or manifest_sha256(parent)!=frozen['manifest_sha256']:raise ValueError('parent identity mismatch')
    def read(name,sha):
        p=root/name;safe_path(p)
        if status['output_hashes'].get(name)!=sha or file_sha256(p)!=sha:raise ValueError('selected parent output changed')
        with p.open() as f:return strict_json_load(f)
    selected=read('provenance/baseline-result.json',frozen['result_sha256'])
    metrics=read('metrics/selected_validation.json',frozen['metrics_sha256'])
    if (selected['complete_epochs']!=5 or selected['selected_epoch']!=frozen['selected_epoch']
            or selected['selected_checkpoint']!=frozen['checkpoint'] or selected['method']!=frozen['method']
            or selected['seed']!=frozen['seed'] or selected['source_sha']!=parent['git_sha']
            or selected['manifest_sha256']!=frozen['manifest_sha256'] or metrics['summary_metrics']['epoch']!=frozen['selected_epoch']):
        raise ValueError('frozen validation selection mismatch')
    ckpt=Path(frozen['checkpoint']['path'])
    if ckpt.parent!=paths.checkpoints:raise ValueError('checkpoint outside parent')
    for p,sha in ((ckpt,frozen['checkpoint']['sha256']),(ckpt.with_suffix('.json'),frozen['checkpoint']['sidecar_sha256'])):
        safe_path(p)
        if status['output_hashes'].get(p.relative_to(root).as_posix())!=sha or file_sha256(p)!=sha:raise ValueError('selected checkpoint publication changed')
    return parent,selected


def reject_duplicate_test(records, *, method, seed):
    prefix=f'msrvtt-final-test-{method}-seed{seed:03d}-'
    if any(r['event']=='completed' and r['run_id'].startswith(prefix) for r in records):
        raise ValueError('successful final test cannot be repeated')


def assert_final_authorized(repo,study,manifest,entries):
    from .authorization import validate_prerequisites
    p=study.get('final_test',{})
    if (p.get('authorized') is not True or p.get('authorization')!='2026-10-12_explicit_user_use_1k_test'
            or entries!={ENTRY} or manifest['stage']!='final' or manifest['dataset']!='msrvtt'
            or manifest['test_authorized'] is not True or manifest['validation_authorized'] is not False
            or manifest['environment']['gpu_count']!=2 or manifest['environment']['precision']!='fp32'
            or manifest['environment']['distributed_topology']!='torchrun-2'):
        raise ValueError('final test lacks exact scoped authorization')
    ev=manifest['config']['evaluation']; parent,selected=authenticated_parent(repo,p,ev['parent_run_id'])
    settings=settings_from_manifest(parent)
    from experiment_artifacts import parse_registry_text
    from .authorization import CONTROL, _committed
    reject_duplicate_test(parse_registry_text(_committed(repo,repo/CONTROL/'registry.jsonl').decode()),method=settings.method,seed=settings.seed)
    if (manifest['config']!=final_config(parent,p,selected) or manifest['variant']!='test-'+settings.method
            or manifest['seed']!=settings.seed or set(manifest['parent_run_ids'])!={parent['run_id'],study['gpu_qualification']['run_id']}):
        raise ValueError('final test differs from frozen selected configuration')
    expected=[p['test_csv'],*p['video_indices']]
    if manifest['dataset_splits']!=expected:raise ValueError('test inputs differ from frozen official inventory')
    ckpt=selected['selected_checkpoint']
    expected_ckpts=[dict(path=str(settings.clip_weights),sha256=settings.clip_sha256),dict(path=ckpt['path'],sha256=ckpt['sha256']),dict(path=str(Path(ckpt['path']).with_suffix('.json')),sha256=ckpt['sidecar_sha256'])]
    if manifest['input_checkpoints']!=expected_ckpts:raise ValueError('test checkpoint inputs differ from selected parent')
    validate_prerequisites(repo,study,manifest,settings)


def test_dataset(references, *, forbidden_ids):
    from .loader import CaptionDataset, official_frame_indices
    if str(CSV) not in references or file_sha256(CSV)!=references[str(CSV)]:raise ValueError('unbound test CSV')
    captions=read_test_csv(CSV,expected_count=1000,forbidden_ids=forbidden_ids)
    videos={}
    cv2.setNumThreads(1)
    for r in captions:
        i=r['official_video_id'];p=VIDEOS/(i+'.mp4');safe_path(p)
        if str(p) not in references or file_sha256(p)!=references[str(p)]:raise ValueError('unbound/changed test video')
        cap=cv2.VideoCapture(str(p))
        try: count,fps=cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS)
        finally:cap.release()
        if not float(count).is_integer():raise ValueError('invalid test frame count')
        official_frame_indices(int(count),fps,12)
        videos[i]=dict(video_id=i,split='test',path=str(p),sha256=references[str(p)],frame_count=int(count),fps=fps)
    return CaptionDataset(videos,captions)


from dataclasses import dataclass
from semantic_retrieval.config import RuntimeConfig

@dataclass(frozen=True)
class FinalSystemConfig:
    runtime: RuntimeConfig
    config: dict


def main(repo,argv=None):
    import argparse,datetime,os,random,shlex
    import numpy as np
    import torch.distributed as dist
    from semantic_retrieval.runtime.artifacts import prepare_run,start_run,complete_run,fail_run,publish_metrics
    from semantic_retrieval.runtime.distributed import configure_fp32
    from semantic_retrieval.evidence.frame_cache import publish_json
    from .authorization import assert_full_data_authorized,CONTROL,_committed
    from .controller import validate_controller_launch
    from .qualification import initial_clip,checkpoint_identity
    from .models import BaselineModel
    from .worker import _collective_call
    from .baseline import evaluate_complete
    from types import SimpleNamespace
    import yaml
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args(argv)
    repo=Path(repo).resolve();root=args.run_dir
    with (root/'manifest.json').open() as f:m=strict_json_load(f)
    registry=repo/CONTROL/'registry.jsonl'
    assert_full_data_authorized(repo_root=repo,registry_path=registry,manifest=m,operation='start',command=shlex.split((root/'command.sh').read_text()))
    validate_controller_launch(repo,root,m)
    configure_fp32();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if int(os.environ.get('WORLD_SIZE','0'))!=2 or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('test requires two deterministic CUDA ranks')
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(60*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180));context=None
    runtime=RuntimeConfig(registry.with_name('study.yaml'),registry,root,m['run_id'],manifest_sha256(m),m['git_sha'],'test')
    try:
        context=_collective_call(lambda:prepare_run(runtime,repo_root=repo,system_config=FinalSystemConfig(runtime,m['config'])))
        _collective_call(lambda:start_run(context) if rank==0 else None)
        permission=yaml.safe_load(_committed(repo,registry.with_name('study.yaml')))['final_test']
        parent,selected=_collective_call(lambda:authenticated_parent(repo,permission,m['config']['evaluation']['parent_run_id']))
        settings=settings_from_manifest(parent)
        from .loader import read_complete
        from semantic_retrieval.runtime.input_bindings import manifest_input_references
        # Parent certification's authenticated inventory proves video-ID disjointness.
        parent_refs=_collective_call(lambda:manifest_input_references(repo,parent))
        videos,_=_collective_call(lambda:read_complete(settings.source_report,settings.source_report_sha256,parent_refs))
        dataset=_collective_call(lambda:test_dataset(context.input_references,forbidden_ids=set(videos)))
        random.seed(settings.seed+rank);np.random.seed(settings.seed+rank);torch.manual_seed(settings.seed+rank)
        model=_collective_call(lambda:BaselineModel(settings.method,settings.clip,initial_clip(settings),temporal_layers=4).to(device))
        identity=checkpoint_identity(SimpleNamespace(config=SimpleNamespace(source_sha=parent['git_sha'])),settings,settings.method)
        _collective_call(lambda:load_selected_model(selected['selected_checkpoint']['path'],model,identity,expected_epoch=selected['selected_epoch']))
        before=model_sha256(model);rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(device).clone()
        scores,result=evaluate_complete(model,dataset,device,expected_split='test')
        if not torch.equal(rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(device)) or before!=model_sha256(model):raise ValueError('test evaluation mutated model/RNG')
        def publish():
            if rank: return
            summary={f'{direction}_{name.lower()}':value for direction in ('t2v','v2t') for name,value in result[direction].items()}
            summary.update(test_video_count=1000,test_query_count=1000,selected_epoch=selected['selected_epoch'])
            arrays=[context.array_store.write('test_scores',scores).as_dict(),context.array_store.write('test_t2v_ranks',result['t2v_ranks']).as_dict(),context.array_store.write('test_v2t_ranks',result['v2t_ranks']).as_dict()]
            metrics=dict(split='test',summary_metrics=summary,array_references=arrays)
            publish_metrics(context,metrics);publish_json(context.paths.metrics/'final_test.json',metrics)
            publish_json(context.paths.provenance/'final-test-result.json',dict(schema_version='full-data-final-test-v1',source_sha=m['git_sha'],manifest_sha256=manifest_sha256(m),parent_run_id=parent['run_id'],selected_checkpoint=selected['selected_checkpoint'],model_sha256=before,dsl=False,seed=settings.seed,method=settings.method,test_authorized=True,training_videos=8246,selection='validation_only',unknown_cross_url_visual_duplicates='not_excluded',paper_training_protocol_differs=True))
            print(f'test method={settings.method} seed={settings.seed} T2V_R1={summary["t2v_r1"]:.4f} V2T_R1={summary["v2t_r1"]:.4f}',flush=True)
            complete_run(context)
        _collective_call(publish)
    except BaseException:
        if rank==0 and context is not None:
            _,s,_=load_validated_run_lifecycle(context.paths)
            if s['state'] in {'running','validating'}:fail_run(context,exit_code=1)
        raise
    finally:dist.destroy_process_group()
    return 0
