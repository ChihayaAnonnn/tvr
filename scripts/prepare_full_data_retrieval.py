#!/usr/bin/env python3
"""Reserved CPU complete-cohort compilation; scientific GPU execution is gated."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

from experiment_artifacts import strict_json_load, manifest_sha256
from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.full_data.preparation import prepare_complete
from semantic_retrieval.runtime.artifacts import prepare_run, start_run, complete_run, fail_run, publish_metrics
from semantic_retrieval.runtime.distributed import configure_fp32


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    configure_fp32()
    with (args.run_dir/'manifest.json').open() as stream:
        manifest = strict_json_load(stream)
    registry = ROOT/manifest['registry_path']
    runtime = RuntimeConfig(registry.with_name('study.yaml'), registry, args.run_dir, manifest['run_id'],
                            manifest_sha256(manifest), manifest['git_sha'], 'val')
    context = prepare_run(runtime, repo_root=ROOT)
    start_run(context)
    try:
        result = prepare_complete(context)
        publish_metrics(context, dict(split='validation', summary_metrics={
            k: float(result[k]) for k in ('decoded_video_count', 'train_video_count', 'validation_video_count',
                                         'excluded_train_count', 'train_caption_count', 'validation_caption_count')},
                                      array_references=[]))
        complete_run(context)
    except BaseException:
        fail_run(context, exit_code=1)
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
