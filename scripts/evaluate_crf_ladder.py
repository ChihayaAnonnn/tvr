"""Task 9 — single-pass CRF ladder evaluator + dev selection + audit / MSVD reports.

Public entry points:

* ``choose_checkpoint(candidates) -> dict`` — pure function used by the
  training loop after every CRF-dev evaluation to persist the current best
  ``best_crf_checkpoint.json``.
* ``build_audit_payload`` / ``build_msvd_payload`` — pure payload builders
  consumed by ``main`` and by tests.
* ``main(argv)`` — CLI evaluator that scores CRF-dev / CRF-audit / MSVD.
  Standard retrieval eval is single-pass; this evaluator too is single-pass
  per CRF level (no paired forward at eval time).

Report gates:

* MSRVTT ``--split audit`` writes exclusive-create (once per run) and stamps
  ``falsified = (u_video_accuracy < 0.90) OR (challenge_delta_ci_low <= 0)``.
* MSRVTT ``--split dev`` may be re-written freely (used every epoch).
* MSVD ``--split msvd_test`` writes ``msvd_report_<ckpt-sha12>.json``; a
  file that already exists for the SAME checkpoint requires ``--overwrite``,
  which stamps ``overwrote_previous_sha256`` and ``overwritten_at`` into
  the new report.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Pre-registered constants (locked at Task 9)
# ---------------------------------------------------------------------------

AUDIT_CRF_LEVELS: tuple = ("clean", 34, 36, 42, 44, 50)
AUDIT_PRIMARY_PAIRS: tuple = (("clean", 34), (34, 42), ("clean", 42))
AUDIT_CHALLENGE_PAIRS: tuple = ((34, 36), (42, 44))
AUDIT_EXTRAPOLATION_PAIR: tuple = (42, 50)

MSVD_CRF_LEVELS: tuple = ("clean", 34, 42)
MSVD_PRIMARY_PAIRS: tuple = (("clean", 34), (34, 42), ("clean", 42))

DEFAULT_AUDIT_SPLIT_JSON = "dataloaders/splits/crf_dev_audit_seed0.json"
DEFAULT_MSVD_SUBSET_JSON = "data/generated/msvd_crf/subset.json"
DEFAULT_MSRVTT_CRF_ROOT = "data/generated/msrvtt_crf"
DEFAULT_MSVD_CRF_ROOT = "data/generated/msvd_crf"
DEFAULT_MSRVTT_CLEAN_ROOT = "/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps"
DEFAULT_MSVD_CLEAN_ROOT = "/data2/hxj/data/MSVD/videos_2"


# ---------------------------------------------------------------------------
# Checkpoint selection (Task 9 core; used by training loop)
# ---------------------------------------------------------------------------


def choose_checkpoint(candidates: Iterable[dict]) -> dict:
    """Return the best CRF-dev checkpoint per the plan's frozen rule.

    Sort key: ``(-accuracy, epoch)``. Strict-> means the FIRST candidate with
    the max accuracy wins even if a later epoch matches exactly.

    ``candidates`` is a list of dicts each carrying at minimum
    ``{"epoch": int, "accuracy": float, "path": str}``. Extra fields (e.g.
    per-CRF breakdown) are preserved on the returned dict.
    """

    materialised = list(candidates)
    if not materialised:
        raise ValueError("choose_checkpoint requires at least one candidate")
    winner = max(materialised, key=lambda c: (float(c["accuracy"]), -int(c["epoch"])))
    return dict(winner)


# ---------------------------------------------------------------------------
# Audit / MSVD report writers
# ---------------------------------------------------------------------------


def write_audit_report(path: str | os.PathLike[str], payload: dict) -> Path:
    """Exclusive-create JSON write for the MSRVTT CRF-audit report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=2)
            fh.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"CRF-audit report already exists at {path}") from exc
    return path


def _checkpoint_sha12(checkpoint_path: str | os.PathLike[str]) -> str:
    return hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()[:12]


def _now_utc_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def write_msvd_report(
    *,
    output_dir: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    payload: dict,
    overwrite: bool = False,
) -> Path:
    """Write MSVD zero-shot report with checkpoint-scoped filename."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sha12 = _checkpoint_sha12(checkpoint_path)
    target = output_dir / f"msvd_report_{sha12}.json"

    if target.exists() and not overwrite:
        raise FileExistsError(
            f"MSVD report {target} already exists — pass --overwrite to replace"
        )

    body = dict(payload)
    if target.exists() and overwrite:
        prev_bytes = target.read_bytes()
        body["overwrote_previous_sha256"] = hashlib.sha256(prev_bytes).hexdigest()
        body["overwritten_at"] = _now_utc_iso()

    encoded = (json.dumps(body, sort_keys=True) + "\n").encode("utf-8")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# Audit gate classification (pure, testable)
# ---------------------------------------------------------------------------


def _classify_audit_gates(
    *,
    u_video_accuracy_three_pair: float,
    u_video_accuracy_50_vs_42: float,
    extrapolation_readable_flag: bool,
    challenge_delta_ci_low: float,
) -> dict:
    """Convert raw CRF-audit metrics into the pre-registered boolean payload.

    Per plan [G14, G15]:
      * in_domain_accuracy_pass    = u_video 3-pair accuracy >= 0.90
      * extrapolation_accuracy_pass= u_video acc CRF50 vs CRF42 >= 0.80
      * baseline_gate_pass         = challenge_delta_ci_low > 0
      * falsified                  = NOT in_domain_accuracy_pass OR NOT baseline_gate_pass
    """
    in_domain = float(u_video_accuracy_three_pair) >= 0.90
    extrapolation = float(u_video_accuracy_50_vs_42) >= 0.80
    baseline = float(challenge_delta_ci_low) > 0.0
    return {
        "in_domain_accuracy_pass": bool(in_domain),
        "extrapolation_accuracy_pass": bool(extrapolation),
        "extrapolation_readable": bool(extrapolation_readable_flag),
        "baseline_gate_pass": bool(baseline),
        "falsified": bool((not in_domain) or (not baseline)),
    }


# ---------------------------------------------------------------------------
# Payload builders (pure)
# ---------------------------------------------------------------------------


def _u_array(records: dict, video_ids: Sequence[str], level) -> np.ndarray:
    return np.asarray([records[vid][level]["u_video"] for vid in video_ids], dtype=float)


def _laplacian_array(records: dict, video_ids: Sequence[str], level) -> np.ndarray:
    return np.asarray([records[vid][level]["laplacian"] for vid in video_ids], dtype=float)


def _saturation_rate_over_records(records: dict, video_ids: Sequence[str], level) -> float:
    hits = [
        bool(records[vid][level]["saturated_low"]) or bool(records[vid][level]["saturated_high"])
        for vid in video_ids
    ]
    if not hits:
        return 0.0
    return float(sum(hits) / len(hits))


def build_audit_payload(
    *,
    video_ids: Sequence[str],
    per_video_records: dict,
    seed: int,
    samples: int,
    metadata: dict,
) -> dict:
    """Build the CRF-audit JSON payload from per-video decoded scores."""
    from crf_ladder.statistics import (
        bootstrap_paired_delta,
        bootstrap_video_accuracy,
        extrapolation_readable,
        paired_accuracy,
        saturation_rate,
    )

    u_by_level = {lvl: _u_array(per_video_records, video_ids, lvl) for lvl in AUDIT_CRF_LEVELS}
    lap_by_level = {
        lvl: _laplacian_array(per_video_records, video_ids, lvl) for lvl in AUDIT_CRF_LEVELS
    }

    # --- Primary 3-pair accuracy ---------------------------------------
    primary_correct = np.stack(
        [(u_by_level[a] < u_by_level[b]).astype(float) for (a, b) in AUDIT_PRIMARY_PAIRS],
        axis=1,
    )  # [V, 3]
    primary_boot = bootstrap_video_accuracy(primary_correct, seed=seed, samples=samples)
    primary_accuracy = float(primary_correct.mean())

    # --- Extrapolation accuracy (CRF50 vs CRF42) -----------------------
    extra_a, extra_b = AUDIT_EXTRAPOLATION_PAIR
    extra_accuracy = float(paired_accuracy(u_by_level[extra_a], u_by_level[extra_b]))

    # --- Saturation per level ------------------------------------------
    # We prefer the record-provided per-video flags (they respect the
    # AleatoricUncertaintyModule's own saturation heuristic if the scorer
    # populated them); a level with no flag information falls back to the
    # spec's 1%-of-range formula on the u_video array itself.
    sat_by_level: dict[str, float] = {}
    for lvl in AUDIT_CRF_LEVELS:
        flag_rate = _saturation_rate_over_records(per_video_records, video_ids, lvl)
        if flag_rate > 0.0:
            sat_by_level[str(lvl)] = flag_rate
        else:
            sat_by_level[str(lvl)] = float(saturation_rate(u_by_level[lvl]))

    readable = bool(
        extrapolation_readable(
            u_by_level[34],
            u_by_level[42],
            u_by_level[50],
            sat_rate=sat_by_level[str(50)],
        )
    )

    # --- Challenge pairs ------------------------------------------------
    challenge_reports: dict[str, dict] = {}
    min_challenge_ci_low = float("inf")
    for a, b in AUDIT_CHALLENGE_PAIRS:
        # For paired video-bootstrap the correctness matrix is [V, 1].
        u_hits = (u_by_level[a] < u_by_level[b]).astype(float).reshape(-1, 1)
        lap_hits = (lap_by_level[a] < lap_by_level[b]).astype(float).reshape(-1, 1)
        u_boot = bootstrap_video_accuracy(u_hits, seed=seed, samples=samples)
        lap_boot = bootstrap_video_accuracy(lap_hits, seed=seed, samples=samples)
        delta = bootstrap_paired_delta(u_hits, lap_hits, seed=seed, samples=samples)
        pair_key = f"({a},{b})"
        challenge_reports[pair_key] = {
            "u_accuracy": float(u_boot.point),
            "u_ci_low": float(u_boot.low),
            "u_ci_high": float(u_boot.high),
            "laplacian_accuracy": float(lap_boot.point),
            "laplacian_ci_low": float(lap_boot.low),
            "laplacian_ci_high": float(lap_boot.high),
            "delta_point": float(delta.point),
            "delta_ci_low": float(delta.low),
            "delta_ci_high": float(delta.high),
        }
        if delta.low < min_challenge_ci_low:
            min_challenge_ci_low = float(delta.low)

    gates = _classify_audit_gates(
        u_video_accuracy_three_pair=primary_accuracy,
        u_video_accuracy_50_vs_42=extra_accuracy,
        extrapolation_readable_flag=readable,
        challenge_delta_ci_low=min_challenge_ci_low,
    )

    payload = {
        "dataset": "msrvtt",
        "split": "audit",
        "n_videos": len(video_ids),
        "crf_levels": [str(lvl) for lvl in AUDIT_CRF_LEVELS],
        "u_video_accuracy_three_pair": primary_accuracy,
        "u_video_accuracy_three_pair_ci_low": float(primary_boot.low),
        "u_video_accuracy_three_pair_ci_high": float(primary_boot.high),
        "u_video_accuracy_50_vs_42": extra_accuracy,
        "saturation_rate_by_level": sat_by_level,
        "challenge_pairs": challenge_reports,
        "challenge_delta_ci_low": (
            float(min_challenge_ci_low)
            if np.isfinite(min_challenge_ci_low) else 0.0
        ),
        "metadata": {
            **metadata,
            "bootstrap_seed": int(seed),
            "bootstrap_samples": int(samples),
        },
    }
    payload.update(gates)
    return payload


def build_msvd_payload(
    *,
    video_ids: Sequence[str],
    per_video_records: dict,
    seed: int,
    samples: int,
    metadata: dict,
) -> dict:
    """Build MSVD zero-shot payload (three-pair accuracy + bootstrap + spearman)."""
    from crf_ladder.statistics import (
        bootstrap_video_accuracy,
        spearman_correlation,
    )

    u_by_level = {lvl: _u_array(per_video_records, video_ids, lvl) for lvl in MSVD_CRF_LEVELS}
    lap_by_level = {lvl: _laplacian_array(per_video_records, video_ids, lvl) for lvl in MSVD_CRF_LEVELS}

    primary_correct = np.stack(
        [(u_by_level[a] < u_by_level[b]).astype(float) for (a, b) in MSVD_PRIMARY_PAIRS],
        axis=1,
    )
    primary_boot = bootstrap_video_accuracy(primary_correct, seed=seed, samples=samples)
    accuracy = float(primary_correct.mean())

    # Exploratory: Spearman ρ between u and -Laplacian across all (video, level)
    # rows. Higher Laplacian = sharper video, so -Laplacian = more compressed
    # → should correlate positively with u if u tracks compression damage.
    all_u = np.concatenate([u_by_level[lvl] for lvl in MSVD_CRF_LEVELS])
    all_neg_lap = -np.concatenate([lap_by_level[lvl] for lvl in MSVD_CRF_LEVELS])
    if all_u.size >= 2 and np.unique(all_neg_lap).size >= 2:
        rho = float(spearman_correlation(all_u, all_neg_lap))
    else:
        rho = float("nan")

    cross_domain_pass = bool(accuracy >= 0.80 and primary_boot.low > 0.5)

    return {
        "dataset": "msvd",
        "split": "msvd_test",
        "n_videos": len(video_ids),
        "crf_levels": [str(lvl) for lvl in MSVD_CRF_LEVELS],
        "three_pair_accuracy": accuracy,
        "three_pair_ci_low": float(primary_boot.low),
        "three_pair_ci_high": float(primary_boot.high),
        "cross_domain_pass": cross_domain_pass,
        "spearman_u_vs_neg_bpp": {"value": rho, "exploratory": True},
        "metadata": {
            **metadata,
            "bootstrap_seed": int(seed),
            "bootstrap_samples": int(samples),
        },
    }


def build_per_video_tsv_rows(
    *,
    dataset: str,
    split: str,
    video_ids: Sequence[str],
    per_video_records: dict,
    levels: Sequence,
) -> list[dict]:
    """Flatten per-video-per-level records into TSV-ready rows."""
    rows: list[dict] = []
    for vid in video_ids:
        for lvl in levels:
            r = per_video_records[vid][lvl]
            rows.append({
                "dataset": dataset,
                "split": split,
                "video_id": vid,
                "level": str(lvl),
                "u_video": float(r["u_video"]),
                "laplacian": float(r["laplacian"]),
                "u_frame_mean": float(r["u_frame_mean"]),
                "u_frame_min": float(r["u_frame_min"]),
                "u_frame_max": float(r["u_frame_max"]),
                "saturated_low": bool(r["saturated_low"]),
                "saturated_high": bool(r["saturated_high"]),
            })
    return rows


def _write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "split", "video_id", "level", "u_video", "laplacian",
        "u_frame_mean", "u_frame_min", "u_frame_max",
        "saturated_low", "saturated_high",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Split loaders
# ---------------------------------------------------------------------------


def _load_audit_video_ids(path: str | os.PathLike[str]) -> list[str]:
    payload = json.loads(Path(path).read_text())
    return list(payload["crf_audit_video_ids"])


def _load_msvd_subset_ids(path: str | os.PathLike[str]) -> list[str]:
    payload = json.loads(Path(path).read_text())
    return list(payload["video_ids"])


def _load_selection(selection_json: Optional[str]) -> Optional[dict]:
    if not selection_json:
        return None
    return json.loads(Path(selection_json).read_text())


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Real scoring loop — decodes videos, runs the model, computes Laplacian
# ---------------------------------------------------------------------------


def _resolve_clean_path(clean_root: Path, dataset: str, video_id: str) -> Path:
    return clean_root / f"{video_id}.mp4"


def _resolve_crf_path(crf_root: Path, level, video_id: str) -> Path:
    from crf_ladder import crf_output_path
    return Path(crf_output_path(str(crf_root), int(level), video_id))


def _score_ladder(
    *,
    checkpoint_path: str,
    video_ids: Sequence[str],
    levels: Sequence,
    clean_paths: dict[str, Path],
    crf_paths: dict[tuple[str, int], Path],
    cli_args: argparse.Namespace,
) -> dict:
    """Decode each (video, level) and compute u_frame / u_video / Laplacian.

    Returns ``{video_id: {level: {u_video, laplacian, u_frame_*, saturated_*}}}``.
    """
    import torch

    from crf_ladder.statistics import laplacian_video_score
    from dataloaders.rawvideo_util import RawVideoExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_uatvr_for_eval(checkpoint_path, cli_args, device)
    model.eval()

    extractor = RawVideoExtractor(
        framerate=cli_args.feature_framerate,
        size=cli_args.image_resolution,
    )

    def _decode_and_score(path: Path) -> dict:
        raw = extractor.get_video_data(str(path))["video"]
        if len(raw.shape) <= 3:
            raise RuntimeError(f"decode returned no frames: {path}")
        proc = extractor.process_raw_data(raw)  # [T, 1, 3, H, W] float32 in [0,1]
        if cli_args.max_frames < proc.shape[0]:
            idx = np.linspace(0, proc.shape[0] - 1, num=cli_args.max_frames, dtype=int)
            proc = proc[idx]
        T = proc.shape[0]

        # Frames for Laplacian: uint8 [T, H, W, 3].
        # RawVideoExtractor normalises to CLIP mean/std then divides / 255 —
        # we recover approximate uint8 RGB from the extractor's `raw` output
        # (which is un-normalised pre-CLIP tensor of shape [T, 1, 3, H, W]
        # already resized to `size`). Fallback to the processed tensor if
        # `raw` has a different layout.
        try:
            if raw.ndim == 5:
                raw_frames = raw[:, 0].permute(0, 2, 3, 1).cpu().numpy()
            else:
                raw_frames = raw.permute(0, 2, 3, 1).cpu().numpy()
            frames_uint8 = np.clip(raw_frames * 255.0, 0, 255).astype(np.uint8)
        except Exception:
            frames_uint8 = None

        video_t = torch.as_tensor(proc).unsqueeze(0).unsqueeze(0).float().to(
            device, non_blocking=True
        )
        mask = torch.ones(1, T, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            u_frame, u_video = model.score_video_risk(video_t, mask)
        u_frame_np = u_frame.float().detach().cpu().numpy().reshape(-1)
        u_video_val = float(u_video.float().detach().cpu().item())

        if frames_uint8 is not None:
            laplacian = float(laplacian_video_score(frames_uint8))
        else:
            laplacian = float("nan")

        u_span = float(u_frame_np.max() - u_frame_np.min()) if u_frame_np.size else 0.0
        threshold = 0.01 * u_span if u_span > 0 else 0.0
        saturated_low = bool(threshold > 0 and float((u_frame_np - u_frame_np.min() < threshold).mean()) > 0.5)
        saturated_high = bool(threshold > 0 and float((u_frame_np.max() - u_frame_np < threshold).mean()) > 0.5)

        return {
            "u_video": u_video_val,
            "laplacian": laplacian,
            "u_frame_mean": float(u_frame_np.mean()),
            "u_frame_min": float(u_frame_np.min()),
            "u_frame_max": float(u_frame_np.max()),
            "saturated_low": saturated_low,
            "saturated_high": saturated_high,
        }

    records: dict = {}
    for i, vid in enumerate(video_ids):
        records[vid] = {}
        for lvl in levels:
            if lvl == "clean":
                path = clean_paths[vid]
            else:
                path = crf_paths[(vid, int(lvl))]
            records[vid][lvl] = _decode_and_score(path)
        if (i + 1) % 25 == 0:
            print(f"[evaluate_crf_ladder] scored {i + 1}/{len(video_ids)} videos", flush=True)
    return records


def _load_uatvr_for_eval(checkpoint_path: str, cli_args: argparse.Namespace, device):
    """Construct the UATVR model, load ``checkpoint_path`` state dict, move to device.

    Reads the training args from ``<ckpt_dir>/training_args.json`` when
    present so the model architecture matches the trained one. Falls back to
    a minimal namespace built from the CLI defaults otherwise.
    """
    import torch

    ckpt_dir = Path(checkpoint_path).parent
    training_args_path = ckpt_dir / "training_args.json"
    if training_args_path.exists():
        task_config = _restore_task_config(training_args_path, cli_args)
    else:
        raise FileNotFoundError(
            f"missing {training_args_path}: cannot reconstruct model architecture. "
            "Either re-run training with the updated main_task_retrieval.py that "
            "writes training_args.json, or hand-craft the file per the runner."
        )

    # Dispatch UATVR class based on training-time --use_uatvr_full (persisted
    # via _write_training_args_json). Default is HEAD arch for old ckpts.
    if getattr(task_config, "use_uatvr_full", False):
        from modules.modeling_mulit import UATVR
    else:
        from modules.modeling import UATVR

    task_config.init_model = str(checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cache_dir = getattr(task_config, "cache_dir", "") or os.path.join(
        os.environ.get("PYTORCH_PRETRAINED_BERT_CACHE", "/tmp"), "distributed"
    )
    model = UATVR.from_pretrained(
        task_config.cross_model,
        cache_dir=cache_dir,
        state_dict=state_dict,
        task_config=task_config,
    )
    model.to(device)
    return model


def _restore_task_config(training_args_path: Path, cli_args: argparse.Namespace) -> argparse.Namespace:
    """Load a persisted training args JSON into a Namespace usable by UATVR."""
    payload = json.loads(training_args_path.read_text())
    ns = argparse.Namespace(**payload)
    # Force single-rank eval mode; the audit CLI runs on one GPU.
    ns.local_rank = 0
    ns.world_size = 1
    ns.n_gpu = 1
    if not hasattr(ns, "cross_model"):
        ns.cross_model = "cross-base"
    # Prefer CLI-provided max_frames / feature_framerate when explicitly set.
    if hasattr(cli_args, "max_frames") and cli_args.max_frames:
        ns.max_frames = int(cli_args.max_frames)
    if hasattr(cli_args, "feature_framerate") and cli_args.feature_framerate:
        ns.feature_framerate = float(cli_args.feature_framerate)
    if not hasattr(ns, "image_resolution"):
        ns.image_resolution = int(getattr(cli_args, "image_resolution", 224))
    return ns


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["msrvtt", "msvd"], required=True)
    parser.add_argument(
        "--split",
        choices=["dev", "audit", "msvd_test"],
        required=True,
        help="MSRVTT: dev|audit. MSVD: msvd_test.",
    )
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Explicit list of checkpoints to score.")
    parser.add_argument("--selection-json", type=str, default=None,
                        help="Path to best_crf_checkpoint.json produced by --split dev.")
    parser.add_argument("--clean-root", type=str, default=DEFAULT_MSRVTT_CLEAN_ROOT)
    parser.add_argument("--msvd-clean-root", type=str, default=DEFAULT_MSVD_CLEAN_ROOT)
    parser.add_argument("--crf-root", type=str, default=DEFAULT_MSRVTT_CRF_ROOT)
    parser.add_argument("--audit-split-json", type=str, default=DEFAULT_AUDIT_SPLIT_JSON)
    parser.add_argument("--msvd-subset-json", type=str, default=DEFAULT_MSVD_SUBSET_JSON)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="MSVD-only: replace an existing report with the same checkpoint SHA.")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--feature-framerate", type=float, default=1.0)
    parser.add_argument("--image-resolution", type=int, default=224)
    return parser.parse_args(argv)


def _resolve_paths_msrvtt(
    clean_root: Path, crf_root: Path, video_ids: Sequence[str], levels: Sequence
) -> tuple[dict, dict]:
    clean = {vid: _resolve_clean_path(clean_root, "msrvtt", vid) for vid in video_ids}
    crf: dict = {}
    for vid in video_ids:
        for lvl in levels:
            if lvl == "clean":
                continue
            crf[(vid, int(lvl))] = _resolve_crf_path(crf_root, lvl, vid)
    return clean, crf


def _resolve_paths_msvd(
    clean_root: Path, crf_root: Path, video_ids: Sequence[str], levels: Sequence
) -> tuple[dict, dict]:
    """Resolve MSVD clean paths (auto-probing .mp4 → .avi) and CRF paths."""
    def _probe(vid: str) -> Path:
        for ext in (".mp4", ".avi", ".webm", ".mkv"):
            candidate = clean_root / f"{vid}{ext}"
            if candidate.exists():
                return candidate
        # Fall back to .mp4 so the downstream error names the expected layout.
        return clean_root / f"{vid}.mp4"

    clean = {vid: _probe(vid) for vid in video_ids}
    crf: dict = {}
    for vid in video_ids:
        for lvl in levels:
            if lvl == "clean":
                continue
            crf[(vid, int(lvl))] = _resolve_crf_path(crf_root, lvl, vid)
    return clean, crf


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.dataset == "msrvtt" and args.split not in ("dev", "audit"):
        raise ValueError("MSRVTT requires --split dev|audit")
    if args.dataset == "msvd" and args.split != "msvd_test":
        raise ValueError("MSVD requires --split msvd_test")

    selection = _load_selection(args.selection_json)
    checkpoints = list(args.checkpoints)
    if not checkpoints and selection is not None:
        checkpoints = [selection.get("path")]
    if not checkpoints:
        raise ValueError("no checkpoints supplied via --checkpoints or --selection-json")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "msrvtt" and args.split == "audit":
        # Pre-check exclusive-create BEFORE running the (expensive) scoring
        # loop, so a mistaken re-run fails fast without recomputing 250×6
        # decodes for nothing.
        target = output_dir / "crf_audit_report.json"
        if target.exists():
            raise FileExistsError(f"CRF-audit report already exists at {target}")

    metadata_base = {
        "checkpoint_path": str(checkpoints[0]),
        "checkpoint_sha12": _checkpoint_sha12(checkpoints[0]) if Path(checkpoints[0]).exists() else "unknown",
        "git_commit": _git_commit(),
        "timestamp_utc": _now_utc_iso(),
    }

    print(
        f"[evaluate_crf_ladder] dataset={args.dataset} split={args.split} "
        f"n_ckpt={len(checkpoints)} output_dir={args.output_dir}"
    )

    if args.dataset == "msrvtt" and args.split == "audit":
        video_ids = _load_audit_video_ids(args.audit_split_json)
        clean_paths, crf_paths = _resolve_paths_msrvtt(
            Path(args.clean_root), Path(args.crf_root), video_ids, AUDIT_CRF_LEVELS
        )
        records = _score_ladder(
            checkpoint_path=checkpoints[0],
            video_ids=video_ids,
            levels=AUDIT_CRF_LEVELS,
            clean_paths=clean_paths,
            crf_paths=crf_paths,
            cli_args=args,
        )
        payload = build_audit_payload(
            video_ids=video_ids,
            per_video_records=records,
            seed=args.bootstrap_seed,
            samples=args.bootstrap_samples,
            metadata=metadata_base,
        )
        write_audit_report(output_dir / "crf_audit_report.json", payload)
        _write_tsv(
            output_dir / "crf_audit_per_video.tsv",
            build_per_video_tsv_rows(
                dataset="msrvtt",
                split="audit",
                video_ids=video_ids,
                per_video_records=records,
                levels=AUDIT_CRF_LEVELS,
            ),
        )
        _print_audit_summary(payload)
        return 0

    if args.dataset == "msvd" and args.split == "msvd_test":
        video_ids = _load_msvd_subset_ids(args.msvd_subset_json)
        clean_paths, crf_paths = _resolve_paths_msvd(
            Path(args.msvd_clean_root), Path(args.crf_root), video_ids, MSVD_CRF_LEVELS
        )
        records = _score_ladder(
            checkpoint_path=checkpoints[0],
            video_ids=video_ids,
            levels=MSVD_CRF_LEVELS,
            clean_paths=clean_paths,
            crf_paths=crf_paths,
            cli_args=args,
        )
        payload = build_msvd_payload(
            video_ids=video_ids,
            per_video_records=records,
            seed=args.bootstrap_seed,
            samples=args.bootstrap_samples,
            metadata=metadata_base,
        )
        sha12 = _checkpoint_sha12(checkpoints[0])
        write_msvd_report(
            output_dir=output_dir,
            checkpoint_path=checkpoints[0],
            payload=payload,
            overwrite=args.overwrite,
        )
        _write_tsv(
            output_dir / f"msvd_per_video_{sha12}.tsv",
            build_per_video_tsv_rows(
                dataset="msvd",
                split="msvd_test",
                video_ids=video_ids,
                per_video_records=records,
                levels=MSVD_CRF_LEVELS,
            ),
        )
        _print_msvd_summary(payload)
        return 0

    if args.dataset == "msrvtt" and args.split == "dev":
        # CRF-dev evaluation is invoked from the training loop directly;
        # exposing a stand-alone CLI path is out of scope for Task 9 —
        # dev selection reuses main_task_retrieval.py's evaluator.
        raise NotImplementedError(
            "--dataset msrvtt --split dev is invoked from the training loop, not this CLI"
        )

    raise ValueError(f"unsupported dataset/split combination: {args.dataset}/{args.split}")


def _print_audit_summary(payload: dict) -> None:
    print(
        f"[audit] three_pair_accuracy={payload['u_video_accuracy_three_pair']:.4f} "
        f"crf50_vs_42={payload['u_video_accuracy_50_vs_42']:.4f} "
        f"challenge_delta_ci_low={payload['challenge_delta_ci_low']:.4f} "
        f"in_domain_pass={payload['in_domain_accuracy_pass']} "
        f"baseline_gate_pass={payload['baseline_gate_pass']} "
        f"extrapolation_readable={payload['extrapolation_readable']} "
        f"falsified={payload['falsified']}"
    )


def _print_msvd_summary(payload: dict) -> None:
    print(
        f"[msvd] three_pair_accuracy={payload['three_pair_accuracy']:.4f} "
        f"ci_low={payload['three_pair_ci_low']:.4f} "
        f"cross_domain_pass={payload['cross_domain_pass']}"
    )


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Public helpers used by main_task_retrieval.py during CRF-dev evaluation.
# ---------------------------------------------------------------------------


def write_dev_selection(path: str | os.PathLike[str], winner: dict) -> Path:
    """Overwrite-friendly write for `best_crf_checkpoint.json` produced every
    CRF-dev epoch. Uses atomic replace so the training loop cannot leave a
    half-written file behind on crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(winner, sort_keys=True, indent=2) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, path)
    return path
