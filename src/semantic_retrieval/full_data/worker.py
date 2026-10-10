"""One sealed worker entry shared by synthetic qualification and full baselines."""
import argparse
import datetime
import os
from pathlib import Path
import random
import shlex

import numpy as np
import torch
import torch.distributed as dist

from experiment_artifacts import strict_json_load, manifest_sha256, load_validated_run_lifecycle
from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.runtime.artifacts import prepare_run, start_run, complete_run, fail_run, publish_metrics
from semantic_retrieval.runtime.distributed import configure_fp32
from .authorization import assert_full_data_authorized
from .runtime import FullDataSystemConfig, settings_from_manifest


def _collective_call(call):
    result, failure = None, None
    try:
        result = call()
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
        local_exception = exc
    errors = [failure]
    if dist.is_initialized():
        errors = [None]*dist.get_world_size(); dist.all_gather_object(errors, failure)
    if any(error is not None for error in errors):
        if failure is not None: raise RuntimeError('collective worker preparation failed: '+str(errors)) from local_exception
        raise RuntimeError('remote worker preparation failed: '+str(errors))
    return result


def main(repo, mode, argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    with (args.run_dir/'manifest.json').open() as stream:
        manifest = strict_json_load(stream)
    settings = settings_from_manifest(manifest)
    if (mode == 'qualification' and settings.stage not in {'cpu_fixture', 'gpu_qualification'}
            or mode == 'training' and settings.stage != 'baseline'):
        raise ValueError('worker entry does not match sealed stage')
    repo = Path(repo).resolve(); registry = repo/manifest['registry_path']
    assert_full_data_authorized(repo_root=repo, registry_path=registry, manifest=manifest, operation='start',
                               command=shlex.split((args.run_dir/'command.sh').read_text()))
    if settings.stage != 'cpu_fixture':
        from .controller import validate_controller_launch
        validate_controller_launch(repo, args.run_dir, manifest)
    configure_fp32(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    world, rank, local_rank = (int(os.environ.get(name, default)) for name, default in
                                (('WORLD_SIZE', '1'), ('RANK', '0'), ('LOCAL_RANK', '0')))
    device = torch.device('cpu') if settings.stage == 'cpu_fixture' else torch.device('cuda', local_rank)
    if device.type == 'cuda':
        if world != 4 or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('CUDA worker requires four ranks and deterministic cuBLAS workspace')
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=datetime.timedelta(seconds=180))
    runtime = RuntimeConfig(registry.with_name('study.yaml'), registry, args.run_dir, manifest['run_id'],
                            manifest_sha256(manifest), manifest['git_sha'], 'val')
    context = None
    try:
        system = FullDataSystemConfig(runtime, settings)
        context = _collective_call(lambda: prepare_run(runtime, repo_root=repo, system_config=system))
        _collective_call(lambda: start_run(context) if rank == 0 else None)
        random.seed(settings.seed+rank); np.random.seed(settings.seed+rank); torch.manual_seed(settings.seed+rank)
        if mode == 'qualification':
            from .qualification import qualify
            report = qualify(context, settings, device)
            def finish():
                if rank == 0:
                    publish_json(context.paths.provenance/'full-data-qualification.json', report)
                    publish_metrics(context, dict(split='validation', summary_metrics=dict(qualification_passed=1.0), array_references=report['array_references']))
                    complete_run(context)
            _collective_call(finish)
        else:
            from .baseline import train_baseline
            train_baseline(context, settings, device)
            _collective_call(lambda: complete_run(context) if rank == 0 else None)
    except BaseException:
        if rank == 0 and context is not None:
            _, status, _ = load_validated_run_lifecycle(context.paths)
            if status['state'] in {'running', 'validating'}:
                fail_run(context, exit_code=1)
        raise
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
    return 0
