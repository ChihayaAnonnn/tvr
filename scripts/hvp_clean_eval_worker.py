"""Read-only model evaluation with query-only word weighting."""
import argparse
import json
import logging
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import numpy as np
import torch
import torch.distributed as dist
from experiment_artifacts import ArtifactArrayStore,load_registry,manifest_sha256,verify_committed_reservation
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.runtime.input_bindings import manifest_input_references
from semantic_retrieval.runtime.checkpoint import model_sha256,rng_state
from semantic_retrieval.full_data.qualification import optimizer_fingerprint
from semantic_retrieval.full_data.campaign import RANDOM,PRETRAINED

REPO=Path(__file__).resolve().parents[1]
CONTROL=Path('experiments/hvp-net-clean-evaluation-v1')
def read(path):return json.loads(Path(path).read_text())


def run(root):
    m=read(root/'manifest.json');ev=m['config']['evaluation'];mode=ev['mode']
    verify_committed_reservation(REPO,CONTROL/'registry.jsonl',m)
    source=Path(ev['hvp_root'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=ev['hvp_sha']:raise ValueError('HVP source changed')
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=source,check=True)
    for name,sha in ev['source_files'].items():
        if file_sha256(name)!=sha:raise ValueError('source/dependency changed: '+name)
    references=manifest_input_references(REPO,m)
    print('VERIFY INPUTS',len(references),flush=True)
    for name,sha in references.items():
        if file_sha256(name)!=sha:raise ValueError('input bytes changed: '+name)
    print('INPUTS VERIFIED',flush=True)
    if mode=='evaluate':
        q=ev['qualification'];path=Path(q['path']);qr=path.parent.parent;status=read(qr/'status.json');report=read(path)
        if (file_sha256(path)!=q['sha256'] or status['state']!='completed'
            or status['output_hashes'].get('provenance/result.json')!=q['sha256']
            or report.get('passed') is not True or report['source_files']!=ev['source_files']
            or report['selected']!=ev['selected'] or report['manifest_sha256']!=manifest_sha256(read(qr/'manifest.json'))
            or not any(r['event']=='completed' and r['run_id']==q['run_id'] for r in load_registry(REPO/CONTROL/'registry.jsonl'))):
            raise ValueError('archived matching inference qualification required')
        if m['test_authorized'] is not True:raise ValueError('missing one-shot test permission')
    sys.path.insert(0,str(source));sys.path.insert(0,ev['dependency_root'])
    import main_retrieval as entry
    import models.module_clip as clip_module
    import models.modeling as modeling
    from models.tokenization_clip import SimpleTokenizer
    from dataloaders.data_dataloaders import dataloader_msrvtt_test
    from utils.local_runtime import load_checkpoint
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.cuda.set_per_process_memory_fraction(16*1024**3/torch.cuda.get_device_properties(local).total_memory,local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
    dist.init_process_group('nccl')
    try:
        sys.argv=['main_retrieval.py','--output_dir',str(root/'checkpoints'),'--fixed_output_dir','--do_eval','1',
            '--base_encoder','ViT-B/32','--layer_list','0,5,11','--max_words','24','--max_frames','12',
            '--batch_size_val','32','--workers','4','--split_batch','8','--clip_dtype','fp16']
        cfg=entry.get_args();cfg.local_rank=local;cfg.world_size=2;cfg.device=device
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');entry.logger=logging.getLogger('clean-evaluation')
        if mode=='qualify':
            with np.load(RANDOM,allow_pickle=False) as archive:state={k:torch.from_numpy(archive[k].copy()) for k in archive.files}
            p=root/'checkpoints'/f'random-clip-rank{rank}.pt';torch.save(state,p);del state
            clip_module._PT_NAME['ViT-B/32']=str(p)
        else:clip_module._PT_NAME['ViT-B/32']=str(PRETRAINED)
        model=modeling.MyModel(cfg).to(device).eval();model.requires_grad_(False)
        checks=[]
        if mode=='qualify':
            generator=torch.Generator().manual_seed(42+rank)
            ids=torch.randint(1,49000,(16,24),generator=generator);ids[:,0]=49406;ids[:,-1]=49407
            video=torch.randn(16,12,3,224,224,generator=generator);idx=torch.arange(rank*16,rank*16+16)
            class Loader:
                dataset=list(range(32))
                def __iter__(self):yield ids,torch.ones(16,24,dtype=torch.long),video,torch.ones(16,12,dtype=torch.long),idx,idx
            def check():
                before=model_sha256(model);rng=optimizer_fingerprint(rng_state());captured=[]
                entry.eval_epoch(cfg,model,Loader(),device,score_callback=lambda x:captured.append(x))
                if model_sha256(model)!=before or optimizer_fingerprint(rng_state())!=rng:raise ValueError('inference changed model/RNG')
                if rank==0 and (captured[0].shape!=(32,32) or not np.isfinite(captured[0]).all()):raise ValueError('bad inference scores')
            check();checks.append('random_production_inference')
        load_checkpoint(model,ev['selected']['path'])
        digest=model_sha256(model);hashes=[None]*2;dist.all_gather_object(hashes,digest)
        if len(set(hashes))!=1:raise ValueError('selected model differs across ranks')
        if mode=='qualify':
            check();checks.append('selected_model_inference')
            if rank==0:publish_json(root/'provenance/result.json',dict(passed=True,checks=checks,selected=ev['selected'],source_files=ev['source_files'],manifest_sha256=manifest_sha256(m),test_accessed=False))
            print('QUALIFICATION PASSED',rank,flush=True)
        else:
            csv=Path('/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv')
            import csv as csv_module
            with csv.open() as stream:rows=list(csv_module.DictReader(stream))
            if len(rows)!=1000 or len({r['video_id'] for r in rows})!=1000:raise ValueError('expected complete unique 1k test')
            anno=root/'provenance/evaluation-annotations'
            if rank==0:
                anno.mkdir();shutil.copyfile(csv,anno/'MSRVTT_test.1000.csv')
            dist.barrier();cfg.anno_path=str(anno);cfg.video_path='/data2/hxj/data/MSRVTT/videos/all'
            loader,count=dataloader_msrvtt_test(cfg,SimpleTokenizer());loader.generator=torch.Generator().manual_seed(42)
            if count!=1000:raise ValueError('test samples dropped')
            refs=[]
            def save_scores(scores):
                refs.append(ArtifactArrayStore(root/'arrays').write('clean_epoch1_scores',scores).as_dict())
            before=model_sha256(model);rng=optimizer_fingerprint(rng_state())
            print('EVALUATE fixed epoch1: query-only weights, 1000x1000',flush=True)
            metrics=entry.eval_epoch(cfg,model,loader,device,return_metrics=True,score_callback=save_scores)
            if model_sha256(model)!=before or optimizer_fingerprint(rng_state())!=rng:raise ValueError('evaluation changed model/RNG')
            if rank==0:
                values={direction:{k:float(v) for k,v in metrics[direction].items() if isinstance(v,(float,int))} for direction in ('t2v','v2t')}
                publish_json(root/'provenance/result.json',dict(completed=True,selected=ev['selected'],model_sha256=before,
                    manifest_sha256=manifest_sha256(m),metrics=values,scores=refs[0],queries=1000,videos=1000,
                    training_word_weighting='candidate_paired_caption',evaluation_word_weighting='query_only',test_accessed=True))
                summary={d+'_'+k:v for d in values for k,v in values[d].items()}
                publish_json(root/'metrics/clean-evaluation.json',dict(split='test',summary_metrics=summary,array_references=refs))
                print('FINAL CLEAN METRICS',json.dumps(values),flush=True)
            del loader
    finally:dist.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
