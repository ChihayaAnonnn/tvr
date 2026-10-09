"""Shared thin entry point; every operation passes the protected runtime."""
import argparse
import datetime
import os
from pathlib import Path

import torch
import torch.distributed as dist

from semantic_retrieval.evidence.config import load_evidence_config
from semantic_retrieval.evidence.runtime import prepare_evidence_run, run_evidence_stage
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, publish_metrics, start_run
from semantic_retrieval.runtime.distributed import configure_fp32


def main(argv, *, allowed_stages, repo_root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--source-sha')
    args = parser.parse_args(argv)
    configure_fp32()
    cfg = load_evidence_config(args.config)
    if cfg.runtime is None or cfg.stage not in allowed_stages:
        raise ValueError('wrong evidence entry stage or missing runtime')
    for key in ('run_id','manifest_sha256','source_sha'):
        if getattr(args,key) is not None and getattr(args,key) != getattr(cfg.runtime,key):
            raise ValueError('CLI/config runtime identity mismatch')
    world = int(os.environ.get('WORLD_SIZE','1')); rank = int(os.environ.get('RANK','0'))
    if world > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo',timeout=datetime.timedelta(minutes=5))
    context = None
    try:
        context = prepare_evidence_run(cfg,repo_root=repo_root)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            start_run(context)
        if dist.is_initialized():
            dist.barrier()
        result = run_evidence_stage(context,cfg)
        if rank == 0:
            metrics = result.get('summary_metrics',{})
            publish_metrics(context,dict(split='validation',summary_metrics=metrics,array_references=[]))
            complete_run(context)
        if dist.is_initialized():
            dist.barrier()
        return 0
    except BaseException:
        if rank == 0 and context is not None:
            fail_run(context,exit_code=1)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
