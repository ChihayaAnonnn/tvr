#!/usr/bin/env python3
"""Build or validate the versioned MSRVTT trusted-v1 split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.msrvtt_protocol import (  # noqa: E402
    build_trusted_manifest,
    load_trusted_manifest,
    validate_trusted_manifest,
    write_generated_split_files,
    write_trusted_manifest,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build or validate the deterministic MSRVTT trusted-v1 split."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate an existing manifest without writing any files.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = load_trusted_manifest(manifest_path)
        validate_trusted_manifest(
            manifest, args.train_csv, args.annotation_json, args.test_csv
        )
    elif args.check_only:
        raise FileNotFoundError(f"trusted manifest not found: {manifest_path}")
    else:
        manifest = build_trusted_manifest(
            args.train_csv, args.annotation_json, args.test_csv
        )
        write_trusted_manifest(manifest_path, manifest)

    if not args.check_only:
        write_generated_split_files(manifest, args.annotation_json, args.output_dir)
    print(
        f"trusted-v1 validated: train={len(manifest['train_video_ids'])} "
        f"val={len(manifest['val_video_ids'])} "
        f"test={manifest['counts']['test_videos']}"
    )


if __name__ == "__main__":
    main()
