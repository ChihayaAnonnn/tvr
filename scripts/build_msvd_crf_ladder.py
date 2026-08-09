"""MSVD CRF ladder builder (zero-shot cross-domain audit only).

Differences from the MSRVTT builder:

* Reads ``test_list.txt`` order; sources are ``.avi`` at the source root.
* Builds a *fixed* subset (default 300 videos) recorded in
  ``<output-root>/subset.json``. On rerun the existing subset.json is REUSED
  and RE-VALIDATED — never regenerated. This is what "zero training / zero
  fine-tuning" means at the data layer: the subset does not drift across
  reruns.
* Only builds CRF 34 and CRF 42 (the two levels the CRF evaluator scores for
  MSVD cross-domain).
* Encoder uses per-file ``probe_video(source)`` geometry / fps. There is no
  ``msrvtt_224_12fps``-style hardcoded rescale on this path — MSVD videos
  keep their native resolution and frame rate.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av

from crf_ladder import codec, crf_output_path

MSVD_CRFS: tuple[int, int] = (34, 42)


@dataclass(frozen=True)
class MSVDBuildJob:
    video_id: str
    crf: int


def _source_path_for(source_root: Path, video_id: str) -> Path:
    return Path(source_root) / f"{video_id}.avi"


def _load_or_freeze_subset(
    output_root: Path,
    test_list: Path,
    source_root: Path,
    subset_size: int,
) -> list[str]:
    subset_json = output_root / "subset.json"
    if subset_json.exists():
        payload = json.loads(subset_json.read_text())
        subset_ids = list(payload["video_ids"])
        for vid in subset_ids:
            if not _source_path_for(source_root, vid).exists():
                raise FileNotFoundError(
                    f"MSVD subset references missing source: {_source_path_for(source_root, vid)}"
                )
        return subset_ids

    ids = [line.strip() for line in test_list.read_text().splitlines() if line.strip()]
    picked: list[str] = []
    missing: list[str] = []
    for vid in ids:
        if len(picked) >= subset_size:
            break
        source = _source_path_for(source_root, vid)
        if source.exists():
            picked.append(vid)
        else:
            missing.append(vid)
    if len(picked) < subset_size:
        raise FileNotFoundError(
            f"MSVD: only {len(picked)} sources present under {source_root}; "
            f"need {subset_size}. First few missing: {missing[:5]}"
        )
    subset_json.parent.mkdir(parents=True, exist_ok=True)
    subset_payload = {
        "selection_rule": "first_n_from_test_list_txt",
        "subset_size": subset_size,
        "video_ids": picked,
        "test_list_source": str(test_list),
    }
    subset_json.write_text(json.dumps(subset_payload, sort_keys=True, indent=2) + "\n")
    return picked


def _manifest_path(output_root: Path, crf: int) -> Path:
    return Path(output_root) / f"crf{crf}" / "manifest.jsonl"


def _load_existing_manifest(output_root: Path) -> dict[tuple[str, int], "codec.ManifestRow"]:
    rows: dict[tuple[str, int], codec.ManifestRow] = {}
    root = Path(output_root)
    for level_dir in sorted(root.glob("crf*")):
        manifest_file = level_dir / "manifest.jsonl"
        if not manifest_file.exists():
            continue
        for line in manifest_file.read_text().splitlines():
            if not line:
                continue
            payload = json.loads(line)
            key = (payload["video_id"], int(payload["crf"]))
            rows[key] = codec.ManifestRow(
                **{k: payload.get(k) for k in codec.ManifestRow.__dataclass_fields__}
            )
    return rows


def _process_job(
    job: MSVDBuildJob,
    source_root: str,
    output_root: str,
    source_sha256: str,
    versions: dict[str, str],
) -> tuple[MSVDBuildJob, "codec.ManifestRow"]:
    source = _source_path_for(Path(source_root), job.video_id)
    dest = crf_output_path(output_root, job.crf, job.video_id)
    try:
        codec.transcode_atomic(source, dest, job.crf)
        row = codec.build_manifest_row(
            job=job,
            source_path=source,
            source_sha256=source_sha256,
            output_path=dest,
            versions=versions,
            success=True,
        )
    except BaseException as exc:  # noqa: BLE001
        row = codec.build_manifest_row(
            job=job,
            source_path=source,
            source_sha256=source_sha256,
            output_path=dest,
            versions=versions,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return job, row


def _versions() -> dict[str, str]:
    return {
        "pyav": av.__version__,
        "av_libraries": json.dumps(
            {k: list(v) for k, v in av.library_versions.items()}, sort_keys=True
        ),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--test-list", required=True,
                        help="Path to desc_files/test_list.txt")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subset-size", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dev-skip-preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    test_list = Path(args.test_list)
    output_root.mkdir(parents=True, exist_ok=True)

    subset_ids = _load_or_freeze_subset(
        output_root, test_list, source_root, args.subset_size
    )
    jobs = [
        MSVDBuildJob(video_id=vid, crf=int(crf))
        for vid in subset_ids
        for crf in MSVD_CRFS
    ]
    existing = _load_existing_manifest(output_root)

    if args.check_only:
        sha_cache: dict[str, str] = {}
        problems: list[str] = []
        for job in jobs:
            source = _source_path_for(source_root, job.video_id)
            source_sha = sha_cache.setdefault(job.video_id, codec.sha256_file(source))
            row = existing.get((job.video_id, job.crf))
            output = crf_output_path(output_root, job.crf, job.video_id)
            if codec.should_rebuild(output, row, source_sha):
                problems.append(f"needs rebuild: {job.video_id}/{job.crf}")
        if problems:
            print(f"[check-only] {len(problems)} problem(s)", file=sys.stderr)
            for issue in problems[:20]:
                print(f"  - {issue}", file=sys.stderr)
            return 2
        print(f"[check-only] verified {len(jobs)} jobs.")
        return 0

    if not args.dev_skip_preflight and jobs:
        first_source = _source_path_for(source_root, jobs[0].video_id)
        preflight_dst = output_root / "preflight.mp4"
        codec.run_preflight(first_source, preflight_dst, frame_limit=16)
        preflight_dst.unlink(missing_ok=True)

    # Compute source SHAs and pending jobs.
    sha_cache: dict[str, str] = {}
    pending: list[MSVDBuildJob] = []
    for job in jobs:
        source = _source_path_for(source_root, job.video_id)
        source_sha = sha_cache.setdefault(job.video_id, codec.sha256_file(source))
        row = existing.get((job.video_id, job.crf))
        output = crf_output_path(output_root, job.crf, job.video_id)
        if codec.should_rebuild(output, row, source_sha):
            pending.append(job)

    print(f"total={len(jobs)} pending={len(pending)} skipped={len(jobs)-len(pending)} "
          f"workers={args.workers}")

    versions = _versions()
    ctx = mp.get_context("fork")

    def _iter_results():
        if args.workers <= 1:
            for job in pending:
                yield _process_job(job, str(source_root), str(output_root),
                                   sha_cache[job.video_id], versions)
            return
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(_process_job, job, str(source_root), str(output_root),
                            sha_cache[job.video_id], versions)
                for job in pending
            ]
            for future in futures:
                yield future.result()

    completed = 0
    started = time.time()
    for job, row in _iter_results():
        codec.append_manifest_row(_manifest_path(output_root, job.crf), row)
        completed += 1
        if completed % 100 == 0 or completed == len(pending):
            elapsed = max(1e-6, time.time() - started)
            print(f"[{completed}/{len(pending)}] {completed/elapsed:.2f} files/s",
                  flush=True)

    print(f"done: built={completed} skipped={len(jobs)-len(pending)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
