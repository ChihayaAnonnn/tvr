"""Task 10 — §3.5 CRF training contract diagnostic table.

Produces a JSON report with the exact key schema listed in Appendix C of
the plan. The runtime numeric collection (gradient isolation checks,
identity delta, throughput measurements) requires a live GPU + a real
UATVR model; that is a Task 11 responsibility. This module owns the
*schema* — every downstream consumer can rely on the keys being stable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

APPENDIX_C_KEYS: tuple[str, ...] = (
    "retrieval_gradient_isolation",
    "ranking_gradient_isolation",
    "global_head_frozen",
    "all_trainable_parameters_have_grad",
    "double_forward_allclose",
    "retrieval_invariance",
    "identity_delta_max",
    "risk_distributions",
    "frame_accuracy",
    "video_accuracy",
    "saturation_rate",
    "hygiene_r1_delta",
    "throughput_seconds_per_step",
)

_RISK_KEYS_REQUIRED: frozenset[str] = frozenset({"clean", "u30", "u38", "u46"})
_ACC_KEYS_REQUIRED: frozenset[str] = frozenset({"clean_vs_30", "30_vs_38", "38_vs_46"})
_DIST_STAT_KEYS: frozenset[str] = frozenset({"count", "mean", "std", "p5", "p50", "p95"})
_THROUGHPUT_KEYS: frozenset[str] = frozenset(
    {"baseline_median", "crf_median", "decode_wait_ratio", "ratio"}
)


def build_diagnostic_payload(
    *,
    retrieval_gradient_isolation: bool,
    ranking_gradient_isolation: bool,
    global_head_frozen: bool,
    all_trainable_parameters_have_grad: bool,
    double_forward_allclose: bool,
    retrieval_invariance: bool,
    identity_delta_max: float,
    risk_distributions: dict,
    frame_accuracy: dict,
    video_accuracy: dict,
    saturation_rate: float,
    hygiene_r1_delta: float,
    throughput_seconds_per_step: dict,
) -> dict[str, Any]:
    """Assemble the fixed Appendix C JSON payload with strict key validation."""

    if _RISK_KEYS_REQUIRED - set(risk_distributions):
        missing = sorted(_RISK_KEYS_REQUIRED - set(risk_distributions))
        raise ValueError(f"risk_distributions missing keys: {missing}")
    for level, stats in risk_distributions.items():
        missing = sorted(_DIST_STAT_KEYS - set(stats))
        if missing:
            raise ValueError(
                f"risk_distributions[{level!r}] missing keys: {missing}"
            )
    if set(frame_accuracy) != _ACC_KEYS_REQUIRED:
        raise ValueError(
            f"frame_accuracy keys must be {sorted(_ACC_KEYS_REQUIRED)}; "
            f"got {sorted(frame_accuracy)}"
        )
    if set(video_accuracy) != _ACC_KEYS_REQUIRED:
        raise ValueError(
            f"video_accuracy keys must be {sorted(_ACC_KEYS_REQUIRED)}; "
            f"got {sorted(video_accuracy)}"
        )
    if set(throughput_seconds_per_step) != _THROUGHPUT_KEYS:
        raise ValueError(
            f"throughput_seconds_per_step keys must be {sorted(_THROUGHPUT_KEYS)}; "
            f"got {sorted(throughput_seconds_per_step)}"
        )
    payload = {
        "retrieval_gradient_isolation": bool(retrieval_gradient_isolation),
        "ranking_gradient_isolation": bool(ranking_gradient_isolation),
        "global_head_frozen": bool(global_head_frozen),
        "all_trainable_parameters_have_grad": bool(all_trainable_parameters_have_grad),
        "double_forward_allclose": bool(double_forward_allclose),
        "retrieval_invariance": bool(retrieval_invariance),
        "identity_delta_max": float(identity_delta_max),
        "risk_distributions": dict(risk_distributions),
        "frame_accuracy": dict(frame_accuracy),
        "video_accuracy": dict(video_accuracy),
        "saturation_rate": float(saturation_rate),
        "hygiene_r1_delta": float(hygiene_r1_delta),
        "throughput_seconds_per_step": dict(throughput_seconds_per_step),
    }
    # Sanity: keys must equal APPENDIX_C_KEYS exactly.
    if set(payload.keys()) != set(APPENDIX_C_KEYS):
        raise AssertionError(  # pragma: no cover — programming error
            f"Appendix C schema drift: {set(payload) ^ set(APPENDIX_C_KEYS)}"
        )
    return payload


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="diagnostic.json")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit a schema-only payload with all-false booleans and zeros — "
             "used by CI + the Task 11 runner to verify the JSON contract "
             "without needing a real GPU.",
    )
    return parser.parse_args(argv)


def _empty_dry_run_payload() -> dict[str, Any]:
    zero_dist = {
        k: {"count": 0, "mean": 0.0, "std": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
        for k in ("clean", "u30", "u38", "u46")
    }
    zero_acc = {"clean_vs_30": 0.0, "30_vs_38": 0.0, "38_vs_46": 0.0}
    return build_diagnostic_payload(
        retrieval_gradient_isolation=False,
        ranking_gradient_isolation=False,
        global_head_frozen=False,
        all_trainable_parameters_have_grad=False,
        double_forward_allclose=False,
        retrieval_invariance=False,
        identity_delta_max=0.0,
        risk_distributions=zero_dist,
        frame_accuracy=zero_acc,
        video_accuracy=zero_acc,
        saturation_rate=0.0,
        hygiene_r1_delta=0.0,
        throughput_seconds_per_step={
            "baseline_median": 0.0,
            "crf_median": 0.0,
            "decode_wait_ratio": 0.0,
            "ratio": 1.0,
        },
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        payload = _empty_dry_run_payload()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        print(f"[dry-run] wrote schema-only payload to {args.output}")
        return 0
    # Real numeric pass requires a GPU + a UATVR model + a CRF dataloader;
    # Task 11 wires that at the runner level. This CLI keeps the schema
    # contract callable so CI can pin it without loading CLIP weights.
    print(
        "[diagnose_crf_training_contract] real numeric pass not implemented "
        "at CLI level; use --dry-run for schema verification or invoke the "
        "training-loop diagnostic hook after a real 1-batch run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
