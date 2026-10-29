"""One explicitly authorized frozen local-query JSFUSION evaluation."""
import argparse
import hashlib
import os
from pathlib import Path
import random
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
import yaml
from experiment_artifacts import strict_json_load,artifact_json_bytes,load_registry,manifest_sha256,verify_committed_reservation,ArtifactArrayStore
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json,safe_path
from semantic_retrieval.full_data.final import load_selected_model,test_dataset
from semantic_retrieval.full_data.baseline import evaluate_complete
from semantic_retrieval.full_data.qualification import _synthetic_batch,optimizer_fingerprint
from semantic_retrieval.full_data.worker import _collective_call
from semantic_retrieval.models.clip import ClipConfig,ClipBackbone
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.runtime.checkpoint import model_sha256,rng_state
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.input_bindings import manifest_input_references
from .local_model import build_local_model
from .local_config import LocalQueryConfig
from .fusion_config import FusionConfig
from .config import AqsConfig

CONTROL=Path('experiments/uatvr-aqs-local-query-training-v1')


def read(path):
    with Path(path).open() as stream:return strict_json_load(stream)


def validate_final_scope(m,permission,mode):
    ev=m['config']['evaluation'];is_test=mode=='test'
    if (mode not in ('qualify','test') or permission.get('authorized') is not True
            or m['study']!='uatvr-aqs-local-query-training-v1' or ev['mode']!=mode or ev['dsl'] is not False
            or ev['frozen']!=permission['frozen'] or permission['frozen']['parent_run_id'] not in m['parent_run_ids']
            or m['dataset']!=('msrvtt' if is_test else 'synthetic') or m['stage']!=('final' if is_test else 'baseline')
            or m['test_authorized'] is not is_test or m['validation_authorized'] is is_test
            or m['environment']['gpu_count']!=2 or m['environment']['precision']!='fp32'):
        raise ValueError('missing exact frozen local-query test authorization')


def load_frozen_model(model,frozen):
    """Strict model-only load; no optimizer or RNG restoration."""
    ref=frozen['checkpoint'];p=Path(ref['path'])
    for path,key in ((p,'sha256'),(p.with_suffix('.json'),'metadata_sha256'),(p.with_suffix('.local.json'),'local_sha256')):
        safe_path(path)
        if file_sha256(path)!=ref[key]:raise ValueError('frozen checkpoint/publication hash mismatch')
    meta=read(p.with_suffix('.json'));side=read(p.with_suffix('.local.json'))
    if (meta['model_sha256']!=frozen['model_sha256'] or side.get('schema')!='uatvr-aqs-local-checkpoint-v1'
            or side.get('architecture')!='uatvr-aqs-local-query-v1' or side['identity']!=meta['identity']
            or hashlib.sha256(artifact_json_bytes(side['recipe'])).hexdigest()!=meta['identity']['recipe_sha256']):
        raise ValueError('frozen local checkpoint identity mismatch')
    base=side['recipe']['base_recipe']
    if (base['local']!=model.local_config.recipe() or base['fusion']!=model.fusion_config.recipe()
            or base['aqs']!=model.aqs_config.recipe() or base['chunks']!={'query':model.query_chunk,'video':model.video_chunk}):
        raise ValueError('inference configuration differs from selected checkpoint')
    load_selected_model(p,model,meta['identity'],expected_epoch=frozen['selected_epoch'])


def authenticate_selection(repo,frozen):
    rows=[r for r in load_registry(repo/CONTROL/'registry.jsonl') if r['run_id']==frozen['parent_run_id'] and r['event']=='completed']
    if len(rows)!=1 or rows[0]['manifest_sha256']!=frozen['parent_manifest_sha256']:raise ValueError('selected training parent not archived')
    root=Path(rows[0]['artifact_path']);parent=read(root/'manifest.json');status=read(root/'status.json')
    if status['state']!='completed' or manifest_sha256(parent)!=frozen['parent_manifest_sha256']:raise ValueError('parent identity mismatch')
    outputs=[(root/'provenance/result.json',frozen['parent_result_sha256'])]
    ref=frozen['checkpoint'];ckpt=Path(ref['path'])
    if ckpt.parent!=root/'checkpoints':raise ValueError('selected checkpoint outside training parent')
    outputs += [(ckpt,ref['sha256']),(ckpt.with_suffix('.json'),ref['metadata_sha256']),(ckpt.with_suffix('.local.json'),ref['local_sha256'])]
    for p,sha in outputs:
        safe_path(p)
        if status['output_hashes'].get(p.relative_to(root).as_posix())!=sha or file_sha256(p)!=sha:raise ValueError('archived selected output changed')
    result=read(root/'provenance/result.json')
    if (result['selected_epoch']!=frozen['selected_epoch'] or not result['completed'] or result['test_accessed'] is not False
            or len(result['epochs'])!=5 or result['manifest_sha256']!=frozen['parent_manifest_sha256']):
        raise ValueError('validation selection identity mismatch')
    side=read(ckpt.with_suffix('.local.json'))
    for name,sha in side['recipe']['sources'].items():
        if file_sha256(repo/name)!=sha:raise ValueError('qualified inference implementation changed: '+name)
    return parent


def verify_gate(repo,m):
    ev=m['config']['evaluation'];ref=ev['qualification'];p=Path(ref['path']);root=p.parent.parent
    status=read(root/'status.json');report=read(p);parent=read(root/'manifest.json')
    if (status['state']!='completed' or status['output_hashes'].get('provenance/result.json')!=ref['sha256']
            or file_sha256(p)!=ref['sha256'] or report.get('passed') is not True
            or report.get('manifest_sha256')!=manifest_sha256(parent) or report['frozen']!=ev['frozen']
            or report['source_files']!=ev['source_files'] or report['test_accessed'] is not False
            or not any(r['event']=='completed' and r['run_id']==ref['run_id'] and r['manifest_sha256']==manifest_sha256(parent) for r in load_registry(repo/CONTROL/'registry.jsonl'))):
        raise ValueError('matching archived final-inference qualification is required')


@torch.no_grad()
def check_inference(model,batch):
    """Check padding, score chunking, and state across the whole inference path."""
    model.eval()
    before=model_sha256(model);rng=optimizer_fingerprint(rng_state())
    ids,mask,video,frames=batch[:4]
    text=model.encode_text(ids.repeat(8,1),mask.repeat(8,1))
    visual=model.encode_video(video,frames)
    scores=model.score(text,visual)
    chunks=torch.cat([torch.cat([
        model.score({k:x[i:i+32] for k,x in text.items()},
                    {k:x[j:j+8] for k,x in visual.items()})
        for j in range(0,len(video),8)],dim=1)
        for i in range(0,len(ids)*8,32)])
    torch.testing.assert_close(scores,chunks,atol=2e-5,rtol=2e-5)
    changed_ids=ids.clone();changed_ids[~mask.bool()]=7
    changed_video=video.clone();changed_video[~frames.bool()]=.17
    changed_text=model.encode_text(changed_ids.repeat(8,1),mask.repeat(8,1))
    changed_visual=model.encode_video(changed_video,frames)
    torch.testing.assert_close(scores,model.score(changed_text,changed_visual),atol=2e-5,rtol=2e-5)
    if before!=model_sha256(model) or rng!=optimizer_fingerprint(rng_state()):
        raise ValueError('encoding/scoring mutated model or RNG')
    return dict(mask=True,chunks=True,model_unchanged=True,rng_unchanged=True)


def run(root,mode):
    repo=Path(__file__).resolve().parents[3];root=root.resolve();m=read(root/'manifest.json')
    verify_committed_reservation(repo,CONTROL/'registry.jsonl',m)
    study=yaml.safe_load((repo/CONTROL/'study.yaml').read_text());permission=study['final_test']
    validate_final_scope(m,permission,mode);ev=m['config']['evaluation']
    for name,sha in ev['source_files'].items():
        if file_sha256(repo/name)!=sha:raise ValueError('final evaluation source changed')
    parent=authenticate_selection(repo,ev['frozen'])
    if m['config']['model']!=parent['config']['model']:raise ValueError('final architecture differs from validation-selected model')
    if os.environ.get('WORLD_SIZE')!='2' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('two deterministic CUDA ranks required')
    import datetime
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    configure_fp32();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.cuda.set_per_process_memory_fraction(60*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=15))
    try:
        if mode=='test':_collective_call(lambda:verify_gate(repo,m))
        random.seed(713);np.random.seed(713);torch.manual_seed(713)
        cfg=ClipConfig(**m['config']['model']['clip']);clip=ClipBackbone(cfg)
        if mode=='test':
            ref=parent['input_checkpoints'][0];_collective_call(lambda:load_clip_state(Path(ref['path']),ref['sha256'],clip))
        model_cfg=m['config']['model'];aqs=AqsConfig.from_recipe(parent['config']['loss']['aqs'])
        side=read(Path(ev['frozen']['checkpoint']['path']).with_suffix('.local.json'))
        model=build_local_model(cfg,clip.state_dict(),local=LocalQueryConfig.from_recipe(model_cfg['local']),fusion=FusionConfig(False,0),aqs=aqs,
            id_mapping_sha256=side['recipe']['id_mapping_sha256'],query_chunk=model_cfg['query_chunk'],video_chunk=model_cfg['video_chunk']).to(device).eval()
        model.requires_grad_(False)
        if mode=='qualify':
            batch=_synthetic_batch(SimpleNamespace(clip=cfg,frames=12,text_length=32),16,rank,device)
            random_checks=check_inference(model,batch)
        _collective_call(lambda:load_frozen_model(model,ev['frozen']))
        hashes=[None]*2;dist.all_gather_object(hashes,model_sha256(model))
        if len(set(hashes))!=1:raise ValueError('selected checkpoint differs across ranks')
        if mode=='qualify':
            selected_checks=check_inference(model,batch)
            report=dict(passed=True,completed=True,manifest_sha256=manifest_sha256(m),source_files=ev['source_files'],frozen=ev['frozen'],world_size=2,test_accessed=False,
                        random_checks=random_checks,selected_checks=selected_checks)
        else:
            refs=_collective_call(lambda:manifest_input_references(repo,m))
            forbidden=read(ev['forbidden_ids']['path'])
            if file_sha256(ev['forbidden_ids']['path'])!=ev['forbidden_ids']['sha256']:raise ValueError('forbidden ID input changed')
            dataset=_collective_call(lambda:test_dataset(refs,forbidden_ids=forbidden['ids']))
            before=model_sha256(model);rng=optimizer_fingerprint(rng_state())
            print('TEST START JSFUSION 1000 queries x1000 videos; epoch3 beta0.2 tau0.1 DSL=false',flush=True)
            scores,metrics=evaluate_complete(model,dataset,device,expected_split='test')
            if before!=model_sha256(model) or rng!=optimizer_fingerprint(rng_state()):raise ValueError('test inference mutated model/RNG')
            store=ArtifactArrayStore(root/'arrays')
            if rank==0:
                score_ref=store.write('jsfusion_1k_scores',scores).as_dict()
                report=dict(completed=True,manifest_sha256=manifest_sha256(m),selected_epoch=ev['frozen']['selected_epoch'],
                    checkpoint=ev['frozen']['checkpoint'],model_sha256=before,t2v=metrics['t2v'],v2t=metrics['v2t'],scores=score_ref,
                    test_queries=1000,test_videos=1000,beta=.2,tau=.1,dsl=False,test_accessed=True)
                summary={direction+'_'+key:float(value) for direction in ('t2v','v2t') for key,value in metrics[direction].items()}
                publish_json(root/'metrics/final_test.json',dict(split='test',summary_metrics=summary,array_references=[score_ref]))
                print('FINAL TEST '+str(dict(t2v=metrics['t2v'],v2t=metrics['v2t'])),flush=True)
        _collective_call(lambda:publish_json(root/'provenance/result.json',report) if rank==0 else None)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--mode',choices=['qualify','test'],required=True)
    args=parser.parse_args();run(args.run_dir,args.mode)
