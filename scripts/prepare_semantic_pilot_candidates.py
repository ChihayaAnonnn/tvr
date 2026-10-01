#!/usr/bin/env python3
"""Run an archived fixed-CLIP candidate producer for the registered pilot phase."""

import argparse
from pathlib import Path

import torch

from experiment_artifacts import manifest_sha256, strict_json_loads
from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.runtime.artifacts import complete_run, fail_run, prepare_run, start_run
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.pilot_candidates import _bound, produce_candidate_artifacts
from semantic_retrieval.runtime.pilot_protocol import CONTROL, PILOT


def runtime_from_run(repo_root, run_dir):
    run = Path(run_dir).resolve(strict=True)
    manifest = strict_json_loads((run / 'manifest.json').read_text())
    if manifest['study'] != PILOT:
        raise ValueError('candidate CLI only accepts the new pilot study')
    return RuntimeConfig(Path(repo_root)/CONTROL/'study.yaml', Path(repo_root)/CONTROL/'registry.jsonl',
        run, manifest['run_id'], manifest_sha256(manifest), manifest['git_sha'], manifest['config']['evaluation']['eval_split'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    configure_fp32()
    runtime = runtime_from_run(repo, args.run_dir)
    context = prepare_run(runtime, repo_root=repo)
    _bound(context, dict(path=str(args.config.resolve()), sha256=__import__('hashlib').sha256(args.config.read_bytes()).hexdigest()))
    config = strict_json_loads(args.config.read_text())
    if torch.cuda.device_count() != 4:
        raise ValueError('candidate worker requires all four qualified GPUs visible')
    start_run(context)
    try:
        produce_candidate_artifacts(context, config, device=torch.device('cuda:0'))
        complete_run(context)
    except BaseException:
        fail_run(context, exit_code=1)
        raise


if __name__ == '__main__':
    main()
