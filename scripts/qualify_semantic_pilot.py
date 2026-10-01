#!/usr/bin/env python3
"""Archived new-study v2 fixture preparation and formal four-GPU qualification."""

import argparse
import datetime
import os
import signal
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from experiment_artifacts import strict_json_loads, validate_metrics_document
from scripts.prepare_semantic_pilot_candidates import runtime_from_run
from scripts.qualify_semantic_retrieval import local_device_uuid
from semantic_retrieval.config import config_payload, load_system_config
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, publish_metrics, start_run
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.gpu_qualification import require_four_gpu
from semantic_retrieval.runtime.pilot_candidates import file_reference, write_json
from semantic_retrieval.runtime.pilot_protocol import PHYSICAL_INDICES, Q0_PROTOCOL, _marker, validate_q0_report, verify_phase_evidence
from semantic_retrieval.runtime.pilot_qualification import build_pilot_fixture_inputs, qualify_pilot_chain


def validate_pilot_qualification_inputs(context, cfg):
    """Authenticate a separately completed Q0P and its entire immutable output inventory."""
    marker = _marker(context.manifest)
    evidence = verify_phase_evidence(context.repo_root, marker['evidence'].get('Q0P'), 'Q0P')
    if len(evidence['runs']) != 1:
        raise ValueError('new qualification requires exactly one synthetic v2 preparation')
    parent = evidence['runs'][0]
    if (parent['manifest']['variant'] != 'pilot-q0-prep' or parent['manifest']['environment']['gpu_count'] != 0
            or parent['manifest']['run_id'] not in context.manifest['parent_run_ids']):
        raise ValueError('new Q0 cannot borrow an old fixture or undeclared parent')
    inventory = strict_json_loads((parent['paths'].provenance/'pilot-input-inventory.json').read_text())
    if (inventory.get('protocol') != Q0_PROTOCOL or inventory.get('software_only') is not True
            or inventory.get('random_weights') is not True):
        raise ValueError('new Q0 synthetic v2 inventory identity mismatch')
    declared = {r['path']:r['sha256'] for f in ('input_checkpoints','dataset_splits') for r in context.manifest[f]}
    if declared != {r['path']:r['sha256'] for r in inventory['inputs']}:
        raise ValueError('new qualification must reserve the complete Q0P input inventory')
    for ref in inventory['inputs']:
        path = Path(ref['path'])
        if (not path.is_relative_to(parent['paths'].root)
                or parent['status']['output_hashes'].get(path.relative_to(parent['paths'].root).as_posix()) != ref['sha256']):
            raise ValueError('new Q0 input differs from completed synthetic v2 preparation')
    template = strict_json_loads((parent['paths'].provenance/'pilot-system-template.json').read_text())
    effective = config_payload(cfg)
    template.pop('runtime'); effective.pop('runtime')
    if template != effective:
        raise ValueError('new qualification must consume its exact P1 v2 system template')


def selected_hardware(study):
    if study.get('physical_indices') != list(PHYSICAL_INDICES):
        raise ValueError('qualification study topology differs from registered physical GPUs')
    output = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True).stdout
    rows = {int(r.split(',')[0]):tuple(x.strip() for x in r.split(',')[1:]) for r in output.splitlines()}
    selected = [rows[i] for i in PHYSICAL_INDICES]
    uuids = [r[0] for r in selected]
    if len(set(uuids)) != 4 or any('A800' not in r[1] for r in selected):
        raise ValueError('new qualification requires four unique registered A800 devices')
    if os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') != uuids:
        raise ValueError('qualification visibility must use selected physical GPU UUIDs in frozen order')
    return dict(physical_indices=list(PHYSICAL_INDICES),gpu_uuids=uuids,nvidia_smi=output,
                precision='fp32',tf32=False,rank_mapping='local rank i maps to physical_indices[i]')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare-production','four-gpu'),required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    args = parser.parse_args(argv)
    configure_fp32()
    repo = Path(__file__).resolve().parents[1]
    runtime = runtime_from_run(repo,args.run_dir)
    manifest = strict_json_loads((args.run_dir/'manifest.json').read_text())
    world,rank = int(os.environ.get('WORLD_SIZE','1')),int(os.environ.get('RANK','0'))
    context = None
    try:
        if args.mode == 'prepare-production':
            if args.config is not None or world != 1 or torch.cuda.device_count() != 0:
                raise ValueError('new Q0P must run once with CUDA visibility disabled')
            if (manifest['variant'] != 'pilot-q0-prep' or runtime.eval_split != 'val'
                    or manifest['config']['evaluation'] != dict(eval_split='val',qualification_scenario='pilot_q0_prep_v2')):
                raise ValueError('new Q0P scenario differs from committed reservation')
            context = prepare_run(runtime,repo_root=repo)
        else:
            if args.config is None or world != 4 or rank != int(os.environ.get('LOCAL_RANK','-1')) or rank not in range(4):
                raise ValueError('new Q0 requires a bound config and four unique local ranks')
            hardware = selected_hardware(yaml.safe_load(runtime.study_path.read_text()))
            if local_device_uuid(rank) != hardware['gpu_uuids'][rank]:
                raise ValueError('actual CUDA ordinal UUID differs from frozen physical topology')
            torch.cuda.set_device(rank)
            dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=8)); require_four_gpu()
            cfg = load_system_config(args.config)
            if cfg.runtime != runtime or cfg.model.clip.fixture or manifest['variant'] != 'pilot-q0-v2':
                raise ValueError('new Q0 must use bound production ViT-B/32 capacity')
            context = prepare_run(runtime,repo_root=repo,system_config=cfg)
            validate_pilot_qualification_inputs(context,cfg)
        def interrupted(signum,frame): raise InterruptedError(f'qualification signal {signum}')
        signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
        if world > 1: dist.barrier()
        if rank == 0: start_run(context)
        if world > 1: dist.barrier()
        if args.mode == 'prepare-production':
            _,refs = build_pilot_fixture_inputs(context)
            metrics = dict(software_only=True,input_count=len(refs))
        else:
            if rank == 0: write_json(context.paths.provenance/'hardware.json',hardware)
            report = qualify_pilot_chain(context,cfg,device=torch.device('cuda',rank))
            validate_q0_report(report,source_sha=manifest['git_sha'])
            metrics = dict(software_only=True,summary_metrics={k:float(report[k]) for k in (
                'resume_parameter_max_abs_delta','resume_optimizer_max_abs_delta','seconds_per_update','elapsed_seconds')})
        if rank == 0:
            publish_metrics(context,metrics)
            summary = metrics.get('summary_metrics',dict(software_input_count=metrics.get('input_count',0)))
            write_json(context.paths.metrics/'selected_validation.json',validate_metrics_document(
                dict(split='validation',summary_metrics=summary,array_references=[])))
            complete_run(context)
        if world > 1: dist.barrier()
    except BaseException:
        if context is not None and rank == 0: fail_run(context,exit_code=1)
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
