#!/usr/bin/env python3
"""Read-only, pinned MSRVTT train/validation provenance audit; stdout only."""
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.source_audit import training_source_metadata, summarize_sources


def main():
    annotation=Path('/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json')
    allowlist=Path('/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv')
    split=ROOT/'dataloaders/splits/msrvtt_trusted_v1_seed0.json'
    pinned={annotation:'8342df0ed6f8212ec8f4363e7415ce9a98d6ddf89e25ed2fb72b364f97c65d33',
            allowlist:'0a7ece2418e1f7ac5680555a04ddfef8950293ac3c4bdd6584fb0fa6cc0ff7d3',
            split:'a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882'}
    if any(file_sha256(p)!=h for p,h in pinned.items()):
        raise ValueError('source audit pinned input changed')
    with allowlist.open() as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['video_id']:
            raise ValueError('unexpected training allowlist schema')
        ids=[r['video_id'] for r in reader]
    if len(ids)!=9000 or len(set(ids))!=9000:
        raise ValueError('training allowlist cardinality mismatch')
    payload=json.loads(split.read_bytes())
    train,val=set(payload['train_video_ids']),set(payload['val_video_ids'])
    if len(train)!=8500 or len(val)!=500 or train & val or train|val!=set(ids):
        raise ValueError('trusted split membership mismatch')
    rows,privacy=training_source_metadata(annotation.read_bytes(),set(ids))
    video_root=Path('/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps')
    def inspect(i):
        path=video_root/(i+'.mp4')
        if path.is_symlink() or path.resolve()!=path:
            raise ValueError('noncanonical source video path')
        return i,file_sha256(path),path.stat().st_size
    with ThreadPoolExecutor(max_workers=8) as pool:
        files=list(pool.map(inspect,sorted(ids)))
    content={i:h for i,h,_ in files}
    result=dict(schema_version='evidence-source-audit-readonly-v1',input_sha256={str(p):h for p,h in pinned.items()},
                privacy=privacy,hash_verified_video_count=len(files),total_bytes_hashed=sum(n for _,_,n in files),
                inventory_sha256=hashlib.sha256(json.dumps(files,separators=(',',':')).encode()).hexdigest(),
                **summarize_sources(rows,train,val,content))
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
