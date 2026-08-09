"""MSRVTT CRF ladder builder.

Layout::

    <output-root>/
        crf30/
            manifest.jsonl   # one row per (video_id, crf), sorted-key JSON per line
            video1.mp4
            ...
        crf38/
        crf46/
        crf34/               # eval-only levels for val videos
        crf36/
        ...
        preflight.mp4        # discarded after the run

Concurrency model (Task 3 of the CRF plan):
* The parent process is the SOLE writer of every per-level ``manifest.jsonl``.
* Workers receive a :class:`BuildJob`, call :func:`codec.transcode_atomic` +
  :func:`codec.build_manifest_row`, and return the :class:`ManifestRow` value.
  Workers must NEVER call :func:`codec.append_manifest_row`.
* Resume runs entirely in the parent. On startup the parent loads every
  existing per-level manifest into a ``(video_id, crf) -> ManifestRow`` map
  (last row per key wins), then decides ``should_rebuild`` per job with a
  freshly-computed source SHA-256.
* Iteration in submission order guarantees byte-identical on-disk manifests
  across reruns with the same worker count and job list.
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

from crf_ladder import (
    EVAL_CRFS,
    TRAIN_CRFS,
    codec,
    crf_output_path,
)

_DEFAULT_SOURCE_ROOT = "/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps"
_DEFAULT_MANIFEST = "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
_DEFAULT_OUTPUT_ROOT = "data/generated/msrvtt_crf"
_DEFAULT_WORKERS = 8


@dataclass(frozen=True)
class BuildJob:
    video_id: str
    crf: int


def build_jobs(manifest: dict) -> list[BuildJob]:
    """Enumerate the full MSRVTT ladder jobs from a trusted-v1 manifest.

    Train videos get every ``TRAIN_CRFS`` level; val videos get every
    ``EVAL_CRFS`` level; test videos are never touched.
    """

    train_ids = list(manifest.get("train_video_ids", []))
    val_ids = list(manifest.get("val_video_ids", []))
    jobs: list[BuildJob] = []
    for vid in train_ids:
        for crf in TRAIN_CRFS:
            jobs.append(BuildJob(video_id=vid, crf=int(crf)))
    for vid in val_ids:
        for crf in EVAL_CRFS:
            jobs.append(BuildJob(video_id=vid, crf=int(crf)))
    return jobs


def _manifest_path(output_root: Path, crf: int) -> Path:
    return Path(output_root) / f"crf{crf}" / "manifest.jsonl"


def _load_existing_manifest(output_root: Path) -> dict[tuple[str, int], "codec.ManifestRow"]:
    """Load every per-level manifest into a ``(video_id, crf) -> ManifestRow`` map."""

    rows: dict[tuple[str, int], codec.ManifestRow] = {}
    root = Path(output_root)
    if not root.exists():
        return rows
    for level_dir in sorted(root.glob("crf*")):
        manifest_file = level_dir / "manifest.jsonl"
        if not manifest_file.exists():
            continue
        for line in manifest_file.read_text().splitlines():
            if not line:
                continue
            payload = json.loads(line)
            key = (payload["video_id"], int(payload["crf"]))
            rows[key] = _row_from_payload(payload)
    return rows


def _row_from_payload(payload: dict) -> "codec.ManifestRow":
    # Accept both old and new schemas defensively; the write path always emits
    # every field, but a manual edit could drop optional keys.
    return codec.ManifestRow(**{key: payload.get(key) for key in codec.ManifestRow.__dataclass_fields__})


def _source_path_for(source_root: Path, video_id: str) -> Path:
    return Path(source_root) / f"{video_id}.mp4"


def _process_job(
    job: BuildJob,
    source_root: str,
    output_root: str,
    source_sha256: str,
    versions: dict[str, str],
) -> tuple[BuildJob, "codec.ManifestRow"]:
    """Worker function: transcode + build manifest row.

    Never opens ``manifest.jsonl``. Any failure is captured in a
    ``success=False`` :class:`ManifestRow`; the exception is not re-raised so
    the parent can decide how to log and continue.
    """

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
    except BaseException as exc:  # noqa: BLE001 — worker returns the failure to parent
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


def _run_check_only(
    jobs: list[BuildJob],
    source_root: Path,
    output_root: Path,
    existing: dict[tuple[str, int], "codec.ManifestRow"],
) -> int:
    """Verify all jobs' outputs match their recorded manifest row. Never writes."""

    problems: list[str] = []
    source_sha_cache: dict[str, str] = {}
    for job in jobs:
        source = _source_path_for(source_root, job.video_id)
        if not source.exists():
            problems.append(f"missing source {source}")
            continue
        source_sha = source_sha_cache.get(job.video_id)
        if source_sha is None:
            source_sha = codec.sha256_file(source)
            source_sha_cache[job.video_id] = source_sha
        row = existing.get((job.video_id, job.crf))
        output = crf_output_path(output_root, job.crf, job.video_id)
        if codec.should_rebuild(output, row, source_sha):
            problems.append(f"needs rebuild: video_id={job.video_id} crf={job.crf}")
    if problems:
        print(f"[check-only] {len(problems)} problem(s):", file=sys.stderr)
        for issue in problems[:20]:
            print(f"  - {issue}", file=sys.stderr)
        return 2
    print(f"[check-only] verified {len(jobs)} jobs; all outputs match manifest.")
    return 0


def _pending_jobs(
    jobs: list[BuildJob],
    source_root: Path,
    output_root: Path,
    existing: dict[tuple[str, int], "codec.ManifestRow"],
) -> tuple[list[BuildJob], dict[str, str]]:
    """Return jobs that need (re)building plus the source-hash cache used to decide."""

    source_sha_cache: dict[str, str] = {}
    pending: list[BuildJob] = []
    for job in jobs:
        source = _source_path_for(source_root, job.video_id)
        if not source.exists():
            raise FileNotFoundError(f"MSRVTT source missing: {source}")
        source_sha = source_sha_cache.get(job.video_id)
        if source_sha is None:
            source_sha = codec.sha256_file(source)
            source_sha_cache[job.video_id] = source_sha
        row = existing.get((job.video_id, job.crf))
        output = crf_output_path(output_root, job.crf, job.video_id)
        if codec.should_rebuild(output, row, source_sha):
            pending.append(job)
    return pending, source_sha_cache


def _dispatch(
    pending: list[BuildJob],
    workers: int,
    source_root: Path,
    output_root: Path,
    source_sha_cache: dict[str, str],
    versions: dict[str, str],
):
    """Yield ``(job, row)`` pairs in submission order."""

    if not pending:
        return
    if workers <= 1:
        for job in pending:
            yield _process_job(
                job,
                str(source_root),
                str(output_root),
                source_sha_cache[job.video_id],
                versions,
            )
        return
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _process_job,
                job,
                str(source_root),
                str(output_root),
                source_sha_cache[job.video_id],
                versions,
            )
            for job in pending
        ]
        for future in futures:
            yield future.result()


def _versions() -> dict[str, str]:
    return {
        "pyav": av.__version__,
        "av_libraries": json.dumps(
            {k: list(v) for k, v in av.library_versions.items()}, sort_keys=True
        ),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=_DEFAULT_SOURCE_ROOT)
    parser.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N jobs (smoke).")
    parser.add_argument("--check-only", action="store_true",
                        help="Verify existing outputs against manifests; never writes.")
    parser.add_argument("--require-x264-provenance", choices=["none", "strict"],
                        default="none",
                        help="`strict` requires PyAV to expose an x264 version at preflight.")
    parser.add_argument("--dev-skip-preflight", action="store_true",
                        help="Test-only: skip the real libx264 preflight probe.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    jobs = build_jobs(manifest)
    if args.limit is not None:
        jobs = jobs[: args.limit]

    existing = _load_existing_manifest(output_root)

    if args.check_only:
        return _run_check_only(jobs, source_root, output_root, existing)

    if not jobs:
        print("no jobs to build.")
        return 0

    # Preflight against the first job's source before spawning any workers.
    if not args.dev_skip_preflight:
        first_source = _source_path_for(source_root, jobs[0].video_id)
        preflight_dst = output_root / "preflight.mp4"
        result = codec.run_preflight(first_source, preflight_dst, frame_limit=16)
        if result.encoder != "libx264":
            raise RuntimeError(
                f"preflight refused: got encoder={result.encoder!r} expected libx264"
            )
        if args.require_x264_provenance == "strict":
            version = _read_x264_version()
            if not version:
                raise RuntimeError(
                    "x264 provenance unavailable — refusing to build audit-grade ladder"
                )
            (output_root / "experiment_manifest.json").write_text(
                json.dumps(
                    {"x264_version": version[0], "x264_version_source": version[1]},
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            )
        preflight_dst.unlink(missing_ok=True)

    pending, source_sha_cache = _pending_jobs(jobs, source_root, output_root, existing)
    skipped = len(jobs) - len(pending)
    total = len(pending)
    print(f"total={len(jobs)} pending={total} skipped={skipped} workers={args.workers}")

    versions = _versions()

    completed = 0
    started = time.time()
    for job, row in _dispatch(
        pending, args.workers, source_root, output_root, source_sha_cache, versions
    ):
        codec.append_manifest_row(_manifest_path(output_root, job.crf), row)
        completed += 1
        if completed % 100 == 0 or completed == total:
            elapsed = max(1e-6, time.time() - started)
            rate = completed / elapsed
            remaining = total - completed
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"[{completed}/{total}] {rate:.2f} files/s ETA={eta:.1f}s",
                flush=True,
            )

    print(f"done: built={completed} skipped={skipped} rebuilt=0")
    return 0


def _read_x264_version() -> Optional[tuple[str, str]]:
    """Best-effort read of libx264's version through PyAV.

    Returns ``(hex_version, source_description)`` when a non-empty version is
    exposed; ``None`` otherwise. `av.codec.Codec.properties` is not a stable
    PyAV API contract, so this is intentionally best-effort — the caller uses
    the None return value to fail fast when audit-grade provenance is required.
    """

    try:
        codec_obj = av.codec.Codec("libx264", "w")
        raw = getattr(codec_obj, "long_name", "") or ""
    except Exception:  # pragma: no cover — environmental
        return None
    if not raw:
        return None
    return raw, "av.codec.Codec.long_name"


if __name__ == "__main__":
    sys.exit(main())
