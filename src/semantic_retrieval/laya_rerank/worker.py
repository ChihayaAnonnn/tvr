"""Manifest-only frozen validation entry. No training or automatic weight downloads."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LayaSystemConfig:
    runtime: object
    config: dict


def main(repo, argv=None):
    import argparse
    import json
    import os
    import shlex
    import sys
    import numpy as np
    import torch
    from types import SimpleNamespace
    from experiment_artifacts import strict_json_load, manifest_sha256, verify_array_reference
    from semantic_retrieval.config import RuntimeConfig
    from semantic_retrieval.runtime.artifacts import prepare_run,start_run,complete_run,fail_run,publish_metrics
    from semantic_retrieval.runtime.distributed import configure_fp32
    from semantic_retrieval.runtime.input_bindings import manifest_input_references
    from semantic_retrieval.full_data.final import authenticated_parent,load_selected_model
    from semantic_retrieval.full_data.runtime import settings_from_manifest
    from semantic_retrieval.full_data.qualification import initial_clip,checkpoint_identity
    from semantic_retrieval.full_data.models import BaselineModel
    from semantic_retrieval.full_data.loader import read_complete,CaptionDataset,official_frame_indices
    from semantic_retrieval.full_data.baseline import evaluate_complete
    from semantic_retrieval.evidence.frame_cache import publish_json
    from .protocol import CONTROL,assert_laya_authorized,committed
    from .store import digest
    from .core import compare,evidence_indices
    from .evidence import decode_evidence
    from .evaluator import evaluate,validate_inputs
    from .visual_cache import VisualCache
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(argv);repo=Path(repo).resolve();root=args.run_dir.resolve()
    with (root/'manifest.json').open() as f: m=strict_json_load(f)
    registry=repo/CONTROL/'registry.jsonl'
    assert_laya_authorized(repo_root=repo,registry_path=registry,manifest=m,operation='start',
                           command=shlex.split((root/'command.sh').read_text()))
    # Force offline before loading any external model code.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    configure_fp32();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    if torch.cuda.device_count()!=1: raise ValueError('fixed validation worker requires one visible GPU')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise ValueError('deterministic cuBLAS required')
    runtime=RuntimeConfig(registry.with_name('study.yaml'),registry,root,m['run_id'],manifest_sha256(m),m['git_sha'],'val')
    context=None
    try:
        context=prepare_run(runtime,repo_root=repo,system_config=LayaSystemConfig(runtime,m['config']))
        start_run(context)
        fixed=m['config']['evaluation']
        inventory=json.loads(committed(repo,repo/CONTROL/'external.input-index.json'))
        for item in inventory['files']:
            if digest(item['path'])!=item['sha256']: raise ValueError('external code/weights changed')
        sys.path.insert(0,fixed['external_code'])
        from .judge import LayaJudge
        parent,selected=authenticated_parent(repo,{'selected':[fixed['parent']]},fixed['parent']['run_id'])
        settings=settings_from_manifest(parent)
        refs=manifest_input_references(repo,parent)
        videos,captions=read_complete(settings.source_report,settings.source_report_sha256,refs)
        gallery=sorted(i for i,r in videos.items() if r['split']=='val')
        val_videos={i:videos[i] for i in gallery}
        dataset=CaptionDataset(val_videos,captions['val'],frames=12,text_length=32)
        model=BaselineModel('uatvr',settings.clip,initial_clip(settings),temporal_layers=4)
        identity=checkpoint_identity(SimpleNamespace(config=SimpleNamespace(source_sha=parent['git_sha'])),settings,'uatvr')
        identity['world_size']=parent['environment']['gpu_count']
        load_selected_model(selected['selected_checkpoint']['path'],model,identity,expected_epoch=5)
        model.eval().requires_grad_(False).to('cuda')
        with torch.inference_mode(): scores,_=evaluate_complete(model,dataset,torch.device('cuda'))
        order=np.argsort(-scores,axis=1,kind='stable')
        queries=[dict(query_id=q['query_id'],query_text=q['query_text'],video_id=q['official_video_id'],split='val') for q in captions['val']]
        rows=[val_videos[i] for i in gallery]
        validate_inputs(queries,rows,order,expected_queries=10000,expected_videos=500)
        target=np.array([gallery.index(q['video_id']) for q in queries])
        metrics=compare(order,order,target)['baseline']
        parent_root=Path(fixed['parent']['checkpoint']['path']).parents[1]
        with (parent_root/'metrics/selected_validation.json').open() as f: archived=strict_json_load(f)
        for key in ('r1','r5','r10'):
            if abs(metrics[key]-archived['summary_metrics']['t2v_'+key])>1e-9:
                raise ValueError('unexplained archived baseline metric mismatch')
        # Scores are also checked against the hash-bound archived FP32 score matrix.
        reference=next(r for r in archived['array_references'] if r['key']=='epoch_5_scores')
        verify_array_reference(parent_root,reference)
        original=np.load(parent_root/reference['path'],allow_pickle=False)
        if not np.array_equal(order,np.argsort(-original,axis=1,kind='stable')):
            raise ValueError('unexplained archived baseline ranking mismatch')
        model.cpu();del model;torch.cuda.empty_cache()
        judge=LayaJudge(fixed['weights'],device='cuda')
        visual=VisualCache(root/'arrays/visual-cache',{'inventory':digest(repo/CONTROL/'external.input-index.json'),'precision':'fp32'})
        memory={}
        for row in rows:
            indices=evidence_indices(official_frame_indices(row['frame_count'],row['fps'],12))
            key={'video_sha256':row['sha256'],'indices':indices}
            memory[row['video_id']]=visual.get(key,lambda row=row:decode_evidence(row)[0],judge)
        report=evaluate(queries,rows,order,judge.score,lambda row:memory[row['video_id']],root/'arrays/query-blocks',
                        {'inventory':digest(repo/CONTROL/'external.input-index.json'),'config':fixed})
        report.update(source_sha=m['git_sha'],manifest_sha256=manifest_sha256(m),test_authorized=False,
                      performance_comparison_completed=False)
        publish_json(root/'provenance/laya-validation.json',report)
        publish_metrics(context,dict(split='validation',summary_metrics={'delta_r1_pp':report['delta_r1_pp'],
            **{'baseline_'+k:v for k,v in report['metrics']['baseline'].items()},
            **{'reranked_'+k:v for k,v in report['metrics']['reranked'].items()}},array_references=[]))
        complete_run(context)
    except BaseException:
        if context is not None: fail_run(context,exit_code=1)
        raise
    return 0
