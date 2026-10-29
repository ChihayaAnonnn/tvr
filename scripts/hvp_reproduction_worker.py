"""Pinned HVP-Net two-rank qualification/training bridge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist
from experiment_artifacts import manifest_sha256,verify_committed_reservation
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.runtime.input_bindings import manifest_input_references
from semantic_retrieval.full_data.campaign import RANDOM
from semantic_retrieval.runtime.checkpoint import model_sha256

REPO=Path(__file__).resolve().parents[1]
CONTROL=Path('experiments/hvp-net-reproduction-v1')

def read(path):
    return json.loads(Path(path).read_text())


def run(root):
    m=read(root/'manifest.json');ev=m['config']['evaluation'];mode=ev['mode']
    verify_committed_reservation(REPO,CONTROL/'registry.jsonl',m)
    source=Path(ev['hvp_root'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=ev['hvp_sha']:
        raise ValueError('HVP source SHA changed')
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=source,check=True)
    for name,sha in ev['source_files'].items():
        if file_sha256(name)!=sha:raise ValueError('pinned source/dependency changed: '+name)
    # Expand and verify every video/input hash before consumption.
    references=manifest_input_references(REPO,m)
    print('VERIFY INPUTS '+str(len(references)),flush=True)
    for name,sha in references.items():
        if file_sha256(name)!=sha:raise ValueError('input bytes changed: '+name)
    print('INPUTS VERIFIED',flush=True)
    if mode=='train':
        q=ev['qualification'];report=read(q['path']);qroot=Path(q['path']).parent.parent
        status=read(qroot/'status.json')
        if (file_sha256(q['path'])!=q['sha256'] or status['state']!='completed'
                or status['output_hashes'].get('provenance/result.json')!=q['sha256']
                or report.get('passed') is not True or report['source_files']!=ev['source_files']):
            raise ValueError('matching archived qualification required')
        if m['test_authorized'] is not True:raise ValueError('author test selection not authorized')
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'.local-deps'))
    import main_retrieval as entry
    import models.module_clip as clip_module
    import models.modeling as modeling
    from utils.local_runtime import load_checkpoint
    from utils.metric_logger import MetricLogger
    import logging
    sys.argv=['main_retrieval.py','--output_dir',str(root/'checkpoints'),'--fixed_output_dir',
        '--do_train','1' if mode=='train' else '0','--base_encoder','ViT-B/32','--layer_list','0,5,11',
        '--anno_path',str(source/'datasets/MSR-VTT/raw_data'),'--video_path','/data2/hxj/data/MSRVTT/videos/all',
        '--batch_size','32','--batch_size_val','32','--epochs','5','--seed','42','--max_words','24','--max_frames','12',
        '--lr','1e-4','--coef_lr','1e-3','--weight_decay','0.2','--warmup_proportion','0.1',
        '--workers','4','--n_display','10','--split_batch','8','--clip_dtype','fp16' if mode=='train' else 'fp32']
    cfg=entry.get_args()
    # Pin the actual loader path, rather than trusting mutable ignored symlinks.
    clip_module._PT_NAME['ViT-B/32']=m['input_checkpoints'][0]['path']
    if mode=='train':
        for filename in ('MSRVTT_train.9000.csv','MSRVTT_test.1000.csv','MSRVTT_data.json'):
            resolved=str((Path(cfg.anno_path)/filename).resolve())
            if resolved not in references:raise ValueError('annotation alias outside bound inputs')
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.cuda.set_per_process_memory_fraction(60*1024**3/torch.cuda.get_device_properties(local).total_memory,local)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if mode=='train':
        entry.get_args=lambda:cfg
        print('TRAIN START corrected HVP-Net B32, 9k train, author test-selected, global batch32, epochs5',flush=True)
        entry.main()
        if dist.get_rank()==0:
            last=torch.load(root/'checkpoints/checkpoint_epoch_4.pth',map_location='cpu')
            publish_json(root/'provenance/result.json',dict(completed=True,manifest_sha256=manifest_sha256(m),
                epochs=5,selection='test_T2V_R1',best_test_R1=float(last['best_score']),source_files=ev['source_files'],test_accessed=True))
    else:
        rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
        dist.init_process_group('nccl');cfg.local_rank=local;cfg.world_size=2;cfg.device=torch.device('cuda',local)
        entry.logger=logging.getLogger('qualify');entry.meters=MetricLogger(delimiter='')
        torch.set_num_threads(1)
        with np.load(RANDOM,allow_pickle=False) as archive:
            state={name:torch.from_numpy(archive[name].copy()) for name in archive.files}
        random_pt=root/'checkpoints'/f'random-clip-rank{rank}.pt'
        torch.save(state,random_pt);del state
        clip_module._PT_NAME['ViT-B/32']=str(random_pt)
        reports=[]
        for dtype in ('fp32','fp16'):
            random.seed(42);np.random.seed(42);torch.manual_seed(42);cfg.clip_dtype=dtype
            model=modeling.MyModel(cfg).to(cfg.device)
            network=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],find_unused_parameters=True)
            opt,_,_=entry.prep_optimizer(cfg,model,4,local)
            generator=torch.Generator().manual_seed(42+rank)
            ids=torch.randint(1,49000,(16,24),generator=generator,device='cpu').to(cfg.device)
            ids[:,0]=49406;ids[:,-1]=49407;mask=torch.ones_like(ids)
            video=torch.randn(16,12,3,224,224,generator=generator,device='cpu').to(cfg.device)
            frames=torch.ones(16,12,device=cfg.device,dtype=torch.long)
            loss=network(ids,mask,video,frames,torch.arange(16,device=cfg.device),0)
            if not torch.isfinite(loss):raise ValueError('nonfinite production loss '+dtype)
            opt.zero_grad();loss.backward()
            groups={prefix:False for prefix in ('clip.','PatchListProcessing.','w_feat_w.')}
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise ValueError('nonfinite gradient '+dtype+' '+name)
                    for prefix in groups:
                        if name.startswith(prefix) and p.grad.abs().max()>0:groups[prefix]=True
            if not all(groups.values()):raise ValueError('missing production gradient path')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad()
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite updated weights')
            digest=model_sha256(model)
            cp=root/'checkpoints'/f'qualify-{dtype}-rank{rank}.pth'
            torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),epoch=1,global_step=1,best_score=0.),cp)
            with torch.no_grad():next(model.parameters()).add_(1)
            load_checkpoint(model,cp,resume=True,optimizer=opt)
            if model_sha256(model)!=digest:raise ValueError('checkpoint roundtrip changed model')
            # Full real encoder, MPP, Gloo exchange, row-shard scoring on synthetic data.
            localids=torch.arange(rank*3,rank*3+3)
            class Loader:
                dataset=list(range(6))
                def __iter__(self):
                    yield ids[:3],mask[:3],video[:3],frames[:3],localids.to(cfg.device),localids.to(cfg.device)
            entry.eval_epoch(cfg,model,Loader(),cfg.device)
            reports.append(dict(dtype=dtype,loss=float(loss.detach()),gradients=groups,checkpoint=True,distributed_eval=True))
            del network,opt,model,video;torch.cuda.empty_cache()
            print('QUALIFY PASS '+dtype,flush=True)
        if rank==0:publish_json(root/'provenance/result.json',dict(passed=True,completed=True,manifest_sha256=manifest_sha256(m),source_files=ev['source_files'],checks=reports,test_accessed=False))
    if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
