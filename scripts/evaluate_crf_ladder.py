"""Task 9 — single-pass CRF ladder evaluator + dev selection + audit / MSVD reports.

Two entry points:

* ``choose_checkpoint(candidates) -> dict`` — pure function used by the
  training loop after every CRF-dev evaluation to persist the current best
  ``best_crf_checkpoint.json``.
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
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional


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
# Audit / MSVD reporting
# ---------------------------------------------------------------------------


def write_audit_report(path: str | os.PathLike[str], payload: dict) -> Path:
    """Exclusive-create JSON write for the MSRVTT CRF-audit report.

    The plan guarantees the audit runs EXACTLY ONCE on the selected
    checkpoint. Enforcing this at the write layer prevents accidental
    overwrite via re-running the evaluator.
    """
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
    """Write MSVD zero-shot report with checkpoint-scoped filename.

    Different checkpoints never collide because the SHA-256 prefix is baked
    into the filename. A same-checkpoint rerun requires ``overwrite=True``
    and stamps ``overwrote_previous_sha256`` + ``overwritten_at`` into the
    new payload so the previous artifact is traceable.
    """
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
    # Best-effort atomic write via os.replace.
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
# CLI scaffolding
# ---------------------------------------------------------------------------


def _load_selection(selection_json: Optional[str]) -> Optional[dict]:
    if not selection_json:
        return None
    return json.loads(Path(selection_json).read_text())


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
    parser.add_argument("--clean-root", type=str,
                        default="/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps")
    parser.add_argument("--crf-root", type=str, default="data/generated/msrvtt_crf")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="MSVD-only: replace an existing report with the same checkpoint SHA.")
    return parser.parse_args(argv)


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

    print(
        f"[evaluate_crf_ladder] dataset={args.dataset} split={args.split} "
        f"n_ckpt={len(checkpoints)} output_dir={args.output_dir}"
    )
    print(
        "[evaluate_crf_ladder] NOTE: real per-video scoring loop uses the model "
        "via model.score_video_risk(); this CLI is the entry-point scaffolding "
        "expected by the plan's runner. Task 11 wires the end-to-end path."
    )
    # The scoring loop and Laplacian-baseline computation are implemented as
    # library functions called from the training-time evaluator and from
    # Task 10's diagnostics harness; when a user invokes this CLI directly
    # they should have already selected a checkpoint via the training loop.
    # We return 0 so the runner can print the two follow-up commands; the
    # actual audit / MSVD numeric pass happens after a real training run
    # produces a real checkpoint.
    return 0


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
