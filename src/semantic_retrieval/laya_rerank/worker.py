"""Manifest-only frozen validation and measurement entry; parent owns lifecycle/budget."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LayaSystemConfig:
    runtime: object
    config: dict


def main(repo,argv=None):
    import argparse,json,os,sys
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--worker-mode')
    args=parser.parse_args(argv);repo=Path(repo).resolve();root=args.run_dir.resolve()
    if not args.worker_mode:
        from .science_controller import run_controller
        return run_controller(repo,root)
    import datetime,torch,torch.distributed as dist
    from semantic_retrieval.runtime.distributed import configure_fp32
    from semantic_retrieval.runtime.artifacts import assert_runtime_environment
    from .campaign import validate_worker_launch
    from .protocol import CONTROL,committed,assert_laya_authorized
    from .store import digest
    from .science import cohort,fixed_config,accuracy_worker
    from .measurement import prepare_original_cache,measure_arm
    import shlex
    m=json.loads((root/'manifest.json').read_text());mode=args.worker_mode
    validate_worker_launch(repo,root,m,mode)
    assert_laya_authorized(repo_root=repo,registry_path=repo/CONTROL/'registry.jsonl',manifest=m,operation='start',
                           command=shlex.split((root/'command.sh').read_text()))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.dont_write_bytecode=True
    configure_fp32();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    assert_runtime_environment(m)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic cuBLAS required')
    fixed=fixed_config(repo)
    for item in json.loads(committed(repo,repo/CONTROL/'external.input-index.json'))['files']:
        if digest(item['path'])!=item['sha256']:raise ValueError('external runtime/weights changed')
    sys.path.insert(0,fixed['external_code'])
    operation=m['config']['evaluation']['operation']
    if operation=='accuracy' and mode!='accuracy':raise ValueError('worker mode differs from sealed operation')
    if operation=='performance' and mode not in {'prepare','baseline-r0','enhanced-r0','baseline-r1','enhanced-r1','baseline-r2','enhanced-r2'}:
        raise ValueError('invalid performance worker mode')
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK','0')));torch.cuda.set_device(device)
    if operation=='accuracy':
        if int(os.environ.get('WORLD_SIZE','0'))!=2:raise ValueError('accuracy requires two qualified ranks')
        dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    torch.manual_seed(713)
    data=cohort(repo,fixed)
    try:
        if mode=='accuracy':accuracy_worker(repo,root,m,data,device)
        elif mode=='prepare':prepare_original_cache(root,data,device)
        else:measure_arm(repo,root,m,data,device,mode)
    finally:
        if dist.is_initialized():dist.destroy_process_group()
    return 0
