"""Two-rank random-capacity inference, real RGB/cache, and killed-process recovery."""
from pathlib import Path


def random_judge(root,device):
    import json,torch
    from transformers import AutoProcessor
    from laya.vlm import VLMAgent,ImagePrep,build_vlm_model
    from .judge import LayaJudge
    a=object.__new__(VLMAgent)
    a.device=device;a.cfg=json.loads((root/'vlm_agent_config.json').read_text())
    a.cfg['dtype']='fp32'
    a.processor=AutoProcessor.from_pretrained(str(root/'processor'),local_files_only=True)
    a.prep=ImagePrep.from_config(a.cfg,default_backend='processor');a.prep.apply(a.processor)
    a.model=build_vlm_model(a.cfg,str(root/'backbone'),dtype=torch.float32,prep=a.prep).to(device).eval().requires_grad_(False)
    a.cfg['readout']=a.processor.laya_readout=a.model.readout
    a.processor.laya_max_len=a.cfg['max_len'];a.prep.check(a.processor)
    a.temperature=a.cfg['temperature'];a.temperature_by_options=a.cfg.get('temperature_by_options',{})
    j=object.__new__(LayaJudge);j.agent=a
    return j


def worker(repo,root,mode):
    import datetime,json,os,sys
    import numpy as np
    import torch
    import torch.distributed as dist
    from experiment_artifacts import manifest_sha256
    from semantic_retrieval.runtime.distributed import configure_fp32
    from .store import digest,atomic,canonical
    from .protocol import CONTROL,committed
    m=json.loads((root/'manifest.json').read_text());cfg=m['config']['evaluation']
    from .campaign import validate_worker_launch
    validate_worker_launch(repo,root,m,mode)
    if json.loads((root/'status.json').read_text())['state']!='running':raise ValueError('qualification worker needs active run')
    for item in json.loads(committed(repo,repo/CONTROL/'qualification.input-index.json'))['files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('qualification runtime input changed')
    sys.dont_write_bytecode=True;sys.path.insert(0,cfg['external_code'])
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('qualification requires two CUDA ranks')
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    configure_fp32();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
    torch.manual_seed(713);np.random.seed(713)
    judge=random_judge(Path(cfg['weights_architecture']),device)
    parameter_count=sum(p.numel() for p in judge.agent.model.parameters())
    if parameter_count<200_000_000:raise ValueError('qualification substituted a reduced Laya model')
    from .evidence import decode_evidence
    from .visual_cache import VisualCache
    from .evaluator import evaluate
    from .core import rerank
    import cv2
    data=root/'arrays'/f'synthetic-rank{rank}';data.mkdir(exist_ok=True)
    rows=[]
    for i in range(12):
        path=data/f'video-{i}.avi'
        if not path.exists():
            out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),2.,(48,32))
            if not out.isOpened():raise ValueError('synthetic RGB encoder unavailable')
            for t in range(8):out.write(np.full((32,48,3),(i*17+t*5)%255,np.uint8))
            out.release()
        rows.append(dict(video_id=str(i),path=str(path),sha256=digest(path),split='val',source_component=str(i),frame_count=8,fps=2.))
    # Full-capacity random original UATVR path; no scientific checkpoint is read.
    from semantic_retrieval.models.clip import ClipBackbone,ClipConfig
    from semantic_retrieval.full_data.models import BaselineModel
    from semantic_retrieval.full_data.loader import CaptionDataset
    clip=ClipConfig(**cfg['uatvr_clip'])
    initial=ClipBackbone(clip).state_dict()
    retriever=BaselineModel('uatvr',clip,initial,temporal_layers=4).eval().requires_grad_(False).to(device)
    captions=[dict(query_id=f'q{i}',query_text=f'A synthetic scene number {i}.',official_video_id=str(i)) for i in range(4)]
    dataset=CaptionDataset({v['video_id']:v for v in rows},captions)
    with torch.inference_mode():
        text=retriever.encode_text(dataset.tokens.to(device),dataset.text_masks.to(device))
        parts=[]
        for i in range(0,12,2):
            decoded=[dataset.decode(str(v)) for v in range(i,i+2)]
            parts.append(retriever.encode_video(torch.stack([v[0] for v in decoded]).to(device),torch.stack([v[1] for v in decoded]).to(device)))
        video={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
        scores=retriever.score(text,video).cpu().numpy()
    if scores.shape!=(4,12) or not np.isfinite(scores).all():raise ValueError('random UATVR inference invalid')
    order=np.argsort(-scores,axis=1,kind='stable')
    retriever.cpu();del retriever,initial,text,video,parts;torch.cuda.empty_cache()
    queries=[dict(query_id=q['query_id'],query_text=q['query_text'],video_id=q['official_video_id'],split='val') for q in captions]
    visual=VisualCache(root/'arrays'/f'visual-rank{rank}',{'model':'random713','config':cfg})
    memory={}
    for row in rows:
        memory[row['video_id']]=visual.get({'sha256':row['sha256'],'frames':[0,2,4,6]},lambda row=row:decode_evidence(row)[0],judge)
    ids=list(range(rank,4,2));calls=0
    def score(texts,frames):
        nonlocal calls
        calls+=1
        if mode=='interrupted' and calls==3:
            atomic(root/f'provenance/interrupted-rank{rank}.json',canonical({'mode':mode,'source_sha':m['git_sha']}))
            dist.barrier();os._exit(23)
        return judge.score(texts,frames)
    target=root/'arrays'/('continuous' if mode=='continuous' else 'resume')/f'rank{rank}'
    result=evaluate([queries[i] for i in ids],rows,order[ids],score,lambda row:memory[row['video_id']],target,{'config':cfg})
    if mode=='interrupted':raise ValueError('interruption not injected')
    def saved(path,kind='full_rankings'):
        marker=next(p for p in path.glob('*.json') if p.name!='identity.json' and json.loads(p.read_text())['key']=={'kind':kind})
        meta=json.loads(marker.read_text())
        if digest(marker.with_suffix('.npy'))!=meta['sha256']:raise ValueError('completed block hash mismatch')
        return np.load(marker.with_suffix('.npy'),allow_pickle=False)
    local=saved(target)
    gathered=[None,None];dist.all_gather_object(gathered,(ids,local,saved(target,'query_logits_probability')))
    checks=dict(real_inference=True,visual_cache=True,shard_merge=True,batch_1_8=True,finite_fp32=True,
                exact_resume=True,cached_uncached_1e5=True,original_uatvr_inference=True)
    if mode=='continuous':
        if not np.array_equal(local,saved(root/'arrays/resume'/f'rank{rank}')):raise ValueError('killed-process recovery ranking changed')
        if not np.array_equal(saved(target,'query_logits_probability'),saved(root/'arrays/resume'/f'rank{rank}','query_logits_probability')):
            raise ValueError('killed-process recovery logits/probabilities changed')
        # Raw pixels versus cached visual features and batched request reordering.
        cache=memory['0'];texts=[q['query_text'] for q in queries]*2
        z8,p8=judge.score(texts,[cache]*8)
        singles=[judge.score([q],[cache]) for q in texts]
        z1=np.concatenate([z for z,p in singles]);p1=np.concatenate([p for z,p in singles])
        if np.max(np.abs(z8-z1))>1e-5 or not np.array_equal(z8.argmax(1),z1.argmax(1)):
            raise ValueError('batch1/8 logits disagree beyond 1e-5 or argmax changes')
        if not np.array_equal(rerank(np.arange(12),np.r_[p8,p8[:2]]),rerank(np.arange(12),np.r_[p1,p1[:2]])):
            raise ValueError('batch1/8 final ranking changed')
        permutation=[7,0,5,2,3,6,1,4]
        zr,_=judge.score([texts[i] for i in permutation],[cache]*8)
        if np.max(np.abs(zr-z8[permutation]))>1e-5:raise ValueError('request permutation changed query mapping')
        from laya.vlm import build_vlm_inputs,collate_vlm,QTYPES
        from .judge import QUESTION,CRITERIA,STATE
        a=judge.agent;q=a._to_internal({'type':'choice','instructions':QUESTION.format(texts[0]),'criteria':CRITERIA})
        row=build_vlm_inputs(a.processor,{**STATE,'images':decode_evidence(rows[0])[0]},q,a.cfg['max_len'],a.cfg['head_max_len'],option_order=[0,1]);row['qtype']=QTYPES['choice']
        b=collate_vlm([row],a.processor.tokenizer.pad_token_id,with_pixels=True)
        with torch.inference_mode():
            raw,_=a.model(*[b[k].to(device) for k in ('input_ids','attention_mask','marker_pos','marker_mask','qtype')],
                         pixel_values=b['pixel_values'].to(device),pixel_attention_mask=b['pixel_attention_mask'].to(device),option_span=b['option_span'].to(device))
        error=float(np.max(np.abs(raw.float().cpu().numpy()-z1[:1])))
        if error>1e-5:raise ValueError('cached/raw-pixel logits disagree beyond 1e-5')
        # Compare both shard outputs to a single-process all-query pass.
        all_root=root/'arrays'/f'single-rank{rank}'
        evaluate(queries,rows,order,judge.score,lambda row:memory[row['video_id']],all_root,{'config':cfg})
        merged=np.empty((4,12),dtype=np.int64)
        merged_numeric=np.empty((4,10,3),dtype=np.float64)
        for indices,values,numeric in gathered:
            merged[indices]=values;merged_numeric[indices]=numeric
        single_numeric=saved(all_root,'query_logits_probability')
        if (np.max(np.abs(merged_numeric-single_numeric))>1e-5
                or not np.array_equal(merged_numeric[:,:,:2].argmax(2),single_numeric[:,:,:2].argmax(2))):
            raise ValueError('two-rank numerical outputs differ from single-process beyond 1e-5 or argmax changes')
        if not np.array_equal(merged,saved(all_root)):raise ValueError('two-rank merge differs from single-process result')
        report=dict(schema_version='laya-qualification-v1',world_size=2,device_type='cuda',production_capacity=True,
                    random_initialization=True,synthetic_only=True,test_authorized=False,source_sha=m['git_sha'],
                    manifest_sha256=manifest_sha256(m),checks=checks,laya_parameters=parameter_count,
                    cached_uncached_max_abs_error=error,batch_max_abs_error=float(np.max(np.abs(z8-z1))))
        atomic(root/f'provenance/qualified-rank{rank}.json',canonical(report))
        dist.barrier()
        if rank==0:
            other=json.loads((root/'provenance/qualified-rank1.json').read_text())
            if other['checks']!=checks:raise ValueError('rank checks incomplete')
            atomic(root/'provenance/laya-qualification.json',canonical(report))
    atomic(root/f'provenance/{mode}-rank{rank}.json',canonical({'checks':checks,'metrics':result['metrics']}))
    dist.barrier();dist.destroy_process_group()


def main(repo,argv=None):
    import argparse,os
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--worker',choices=('interrupted','resumed','continuous'))
    args=parser.parse_args(argv);repo=Path(repo).resolve();root=args.run_dir.resolve()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    if args.worker:return worker(repo,root,args.worker) or 0
    from .campaign import run_controller
    return run_controller(repo,root)
