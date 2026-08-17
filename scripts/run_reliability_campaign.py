#!/usr/bin/env python3
"""Administrative entry point for the validation-gated reliability campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_artifacts import StrictJSONError, strict_json_dumps, strict_json_loads
from prob_models.reliability_campaign import (
    CampaignController,
    evaluate_stage1_gate,
    evaluate_stage2_gate,
    evaluate_stage3_gate,
    evaluate_stage4_gate,
    evaluate_stage5_gate,
    load_phase_spec,
)


def _controller(args: argparse.Namespace) -> CampaignController:
    return CampaignController(
        Path(args.repo_root),
        artifact_root=Path(args.artifact_root) if args.artifact_root else None,
    )


def prepare(args: argparse.Namespace) -> object:
    return _controller(args).prepare(Path(args.phase))


def launch(args: argparse.Namespace) -> object:
    controller = _controller(args)
    phase = Path(args.phase)
    # Test-authorized launches are enforced by re-executing Stage 5 completion,
    # study SHA, and frozen-configuration proofs from committed state inside
    # ``launch_reserved``; no CLI flag can bypass those proofs.
    return controller.launch_reserved(
        controller.reservations_for_phase(phase), phase=phase
    )


def monitor(args: argparse.Namespace) -> object:
    controller = _controller(args)
    if args.phase:
        run_dirs = [
            Path(str(reservation["artifact_path"]))
            for reservation in controller.reservations_for_phase(Path(args.phase))
        ]
    else:
        run_dirs = [Path(value) for value in args.run_dir]
    return controller.monitor(run_dirs)


def _path_is_test_sealed(path: Path) -> bool:
    """A path is sealed if any of its components contains the substring ``test``.

    ``Path.parts`` yields individual path components. Guarding by exact equality
    (``"test" in {parts}``) missed real artifact filenames such as
    ``final_test.json`` or ``stage7_test_metrics.json``. The substring form
    matches those while still allowing benign paths (no component contains
    ``test``).
    """

    return any("test" in part.lower() for part in path.parts)


def _read_validation_payload(path: Path, split: str) -> Mapping[str, object]:
    if split != "validation":
        raise ValueError("gate input must be declared as validation before any file access")
    if _path_is_test_sealed(path):
        raise ValueError("test paths remain sealed during gate evaluation")
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, StrictJSONError) as exc:
        raise ValueError("gate payload is unavailable or invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("gate payload must be an object")
    if payload.get("split", "validation") != "validation":
        raise ValueError("gate payload may contain validation scalars only")
    return payload


def evaluate_gate(args: argparse.Namespace) -> object:
    controller = _controller(args)
    phase = load_phase_spec(controller.repo_root, Path(args.phase))
    payload = _read_validation_payload(Path(args.input), args.split)
    common = {
        "split": "validation",
        "study_sha": controller.study_sha,
        "phase_sha": phase.sha256,
    }
    if args.stage == 1:
        decision = evaluate_stage1_gate(payload["a3"], payload["strongest"], **common)
    elif args.stage == 2:
        decision = evaluate_stage2_gate(
            payload["full"], payload["score_only"], payload["no_context"],
            changed_tensors=str(payload["changed_tensors"]), **common,
        )
    elif args.stage == 3:
        decision = evaluate_stage3_gate(payload["candidates"], **common)
    elif args.stage == 4:
        decision = evaluate_stage4_gate(payload["comparison"], **common)
    else:
        frozen_sim = payload.get("sim_matrix_sha256_frozen")
        candidate_sim = payload.get("sim_matrix_sha256_candidate")
        if not isinstance(frozen_sim, str) or not isinstance(candidate_sim, str):
            raise ValueError(
                "Stage 5 payload must supply sim_matrix_sha256_frozen and "
                "sim_matrix_sha256_candidate; operator-asserted rank booleans "
                "are no longer accepted"
            )
        decision = evaluate_stage5_gate(
            payload["comparison"],
            sim_matrix_sha256_frozen=frozen_sim,
            sim_matrix_sha256_candidate=candidate_sim,
            frozen_configuration_sha256=payload.get("frozen_configuration_sha256"),
            **common,
        )
    return decision.to_dict()


def finalize(args: argparse.Namespace) -> object:
    return _controller(args).finalize(Path(args.phase), Path(args.summary))


def authorize_test(args: argparse.Namespace) -> object:
    return _controller(args).authorize_test(
        Path(args.phase), Path(args.stage5_decision),
        args.frozen_configuration_sha256,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--artifact-root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("prepare", allow_abbrev=False)
    _common(command)
    command.add_argument("--phase", required=True)
    command.set_defaults(handler=prepare)

    command = commands.add_parser("launch", allow_abbrev=False)
    _common(command)
    command.add_argument("--phase", required=True)
    command.set_defaults(handler=launch)

    command = commands.add_parser("monitor", allow_abbrev=False)
    _common(command)
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--phase")
    source.add_argument("--run-dir", action="append")
    command.set_defaults(handler=monitor)

    command = commands.add_parser("evaluate-gate", allow_abbrev=False)
    _common(command)
    command.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    command.add_argument("--phase", required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--split", choices=("validation", "test"), required=True)
    command.set_defaults(handler=evaluate_gate)

    command = commands.add_parser("finalize", allow_abbrev=False)
    _common(command)
    command.add_argument("--phase", required=True)
    command.add_argument("--summary", required=True)
    command.set_defaults(handler=finalize)

    command = commands.add_parser("authorize-test", allow_abbrev=False)
    _common(command)
    command.add_argument("--phase", required=True)
    command.add_argument("--stage5-decision", required=True)
    command.add_argument("--frozen-configuration-sha256", required=True)
    command.set_defaults(handler=authorize_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(strict_json_dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
