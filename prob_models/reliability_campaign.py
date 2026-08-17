"""Pure stop gates and fail-closed orchestration for the reliability campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping, Sequence

from experiment_artifacts import StrictJSONError, strict_json_dumps, strict_json_loads


_DIRECTIONS = ("t2v", "v2t")
_SHA256_LENGTH = 64
_TERMINAL_EVENTS = frozenset({"completed", "failed", "cancelled"})
_A3_COMMAND_SUFFIX = ("bash", "run_train_msrvtt_contextual_risk_bg.sh")
_PHASE_RUN_FIELDS = frozenset(
    {
        "dataset", "stage", "variant", "seed", "scheduled_timestamp_utc",
        "environment", "input_checkpoints", "dataset_splits", "config",
        "parent_run_ids", "selection_lineage", "validation_authorized",
        "test_authorized", "expected_scalar_outputs", "array_references", "command",
    }
)


def _decimal(value: object, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a scalar number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return Decimal(str(value))


def _validate_scalar_tree(value: object, field: str = "input") -> None:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if not isinstance(name, str):
                raise ValueError(f"{field} keys must be strings")
            _validate_scalar_tree(child, f"{field}.{name}")
        return
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
        return
    raise ValueError(f"{field} must contain scalar values only")


def _require_mapping_value(payload: Mapping[str, object], name: str) -> object:
    if name not in payload:
        raise ValueError(f"missing required gate input: {name}")
    return payload[name]


def _require_direction(payload: Mapping[str, object], direction: str) -> Mapping[str, object]:
    value = _require_mapping_value(payload, direction)
    if not isinstance(value, Mapping):
        raise ValueError(f"{direction} must be an object")
    return value


def _require_validation(split: str) -> None:
    if split != "validation":
        raise ValueError("campaign gates may use validation only")


def _binding_sha(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class GateDecision:
    stage: int
    passed: bool
    inputs: object
    comparisons: dict[str, object]
    failed_checks: tuple[str, ...]
    study_sha: str | None
    phase_sha: str | None
    next_authorized_stage: int | None
    frozen_configuration_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["failed_checks"] = list(self.failed_checks)
        return payload


def _decision(
    stage: int,
    inputs: object,
    comparisons: dict[str, object],
    checks: Mapping[str, bool],
    *,
    study_sha: str | None,
    phase_sha: str | None,
    frozen_configuration_sha256: str | None = None,
) -> GateDecision:
    failed = tuple(name for name, passed in checks.items() if not passed)
    passed = not failed
    return GateDecision(
        stage=stage,
        passed=passed,
        inputs=inputs,
        comparisons=comparisons,
        failed_checks=failed,
        study_sha=_binding_sha(study_sha, "study_sha"),
        phase_sha=_binding_sha(phase_sha, "phase_sha"),
        next_authorized_stage=stage + 1 if passed else None,
        frozen_configuration_sha256=_binding_sha(
            frozen_configuration_sha256, "frozen_configuration_sha256"
        ),
    )


def evaluate_stage1_gate(
    a3: Mapping[str, object],
    strongest: Mapping[str, object],
    *,
    split: str,
    study_sha: str | None = None,
    phase_sha: str | None = None,
) -> GateDecision:
    _require_validation(split)
    _validate_scalar_tree(a3, "a3")
    _validate_scalar_tree(strongest, "strongest")
    auroc_delta = _decimal(
        _require_mapping_value(a3, "mean_auroc"), "a3.mean_auroc"
    ) - _decimal(
        _require_mapping_value(strongest, "mean_auroc"), "strongest.mean_auroc"
    )
    aurc_delta = _decimal(
        _require_mapping_value(a3, "mean_aurc"), "a3.mean_aurc"
    ) - _decimal(
        _require_mapping_value(strongest, "mean_aurc"), "strongest.mean_aurc"
    )
    directional_auroc = [
        _decimal(
            _require_mapping_value(_require_direction(a3, direction), "auroc"),
            f"a3.{direction}.auroc",
        )
        for direction in _DIRECTIONS
    ]
    directional_rho = [
        _decimal(
            _require_mapping_value(_require_direction(a3, direction), "rho"),
            f"a3.{direction}.rho",
        )
        for direction in _DIRECTIONS
    ]
    comparisons = {
        "mean_auroc_delta": float(auroc_delta),
        "mean_aurc_delta": float(aurc_delta),
    }
    checks = {
        "a3_improves_strongest": (
            auroc_delta >= Decimal("0.01") or aurc_delta <= Decimal("-0.01")
        ),
        "directional_auroc_gt_0_5": all(
            value > Decimal("0.5") for value in directional_auroc
        ),
        "directional_rho_gte_0_20": all(
            value >= Decimal("0.20") for value in directional_rho
        ),
    }
    return _decision(
        1, {"a3": dict(a3), "strongest": dict(strongest), "split": split},
        comparisons, checks, study_sha=study_sha, phase_sha=phase_sha,
    )


def evaluate_stage2_gate(
    full: Mapping[str, object],
    score_only: Mapping[str, object],
    no_context: Mapping[str, object],
    *,
    changed_tensors: str,
    split: str,
    study_sha: str | None = None,
    phase_sha: str | None = None,
) -> GateDecision:
    _require_validation(split)
    for name, payload in (
        ("full", full), ("score_only", score_only), ("no_context", no_context)
    ):
        _validate_scalar_tree(payload, name)
    comparisons: dict[str, object] = {}
    improvement_checks = []
    directional_checks = []
    for name, baseline in (("score_only", score_only), ("no_context", no_context)):
        auroc_delta = _decimal(full["mean_auroc"], "full.mean_auroc") - _decimal(
            _require_mapping_value(baseline, "mean_auroc"), f"{name}.mean_auroc"
        )
        aurc_delta = _decimal(full["mean_aurc"], "full.mean_aurc") - _decimal(
            _require_mapping_value(baseline, "mean_aurc"), f"{name}.mean_aurc"
        )
        comparisons[name] = {
            "mean_auroc_delta": float(auroc_delta),
            "mean_aurc_delta": float(aurc_delta),
        }
        improvement_checks.append(
            auroc_delta >= Decimal("0.005") or aurc_delta <= Decimal("-0.005")
        )
        for direction in _DIRECTIONS:
            full_value = _decimal(
                _require_mapping_value(_require_direction(full, direction), "auroc"),
                f"full.{direction}.auroc",
            )
            baseline_value = _decimal(
                _require_mapping_value(_require_direction(baseline, direction), "auroc"),
                f"{name}.{direction}.auroc",
            )
            directional_checks.append(full_value - baseline_value >= Decimal("-0.01"))
    checks = {
        "full_improves_both_controls": all(improvement_checks),
        "max_directional_auroc_loss_lte_0_01": all(directional_checks),
        "only_declared_sidecar_tensors_changed": changed_tensors == "a3_sidecar_only",
    }
    return _decision(
        2,
        {
            "full": dict(full), "score_only": dict(score_only),
            "no_context": dict(no_context), "changed_tensors": changed_tensors,
            "split": split,
        },
        comparisons, checks, study_sha=study_sha, phase_sha=phase_sha,
    )


def evaluate_stage3_gate(
    candidates: Sequence[Mapping[str, object]],
    *,
    split: str,
    study_sha: str | None = None,
    phase_sha: str | None = None,
) -> GateDecision:
    _require_validation(split)
    if not candidates:
        raise ValueError("stage 3 candidates must not be empty")
    normalized = []
    for index, candidate in enumerate(candidates):
        _validate_scalar_tree(candidate, f"candidates[{index}]")
        for field in ("name", "mean_auroc", "mean_aurc", "k", "active_loss_terms", "epoch"):
            _require_mapping_value(candidate, field)
        rho = [
            _decimal(
                _require_mapping_value(_require_direction(candidate, direction), "rho"),
                f"candidates[{index}].{direction}.rho",
            )
            for direction in _DIRECTIONS
        ]
        if all(value >= Decimal("0.20") for value in rho):
            normalized.append(candidate)
    retained: list[Mapping[str, object]] = []
    winner = None
    if normalized:
        best_auroc = max(_decimal(item["mean_auroc"], "mean_auroc") for item in normalized)
        best_aurc = min(_decimal(item["mean_aurc"], "mean_aurc") for item in normalized)
        retained = [
            item for item in normalized
            if best_auroc - _decimal(item["mean_auroc"], "mean_auroc") <= Decimal("0.005")
            and _decimal(item["mean_aurc"], "mean_aurc") - best_aurc <= Decimal("0.005")
        ]
        winner = min(
            retained,
            key=lambda item: (
                int(item["k"]), int(item["active_loss_terms"]), int(item["epoch"])
            ),
        )
    comparisons = {
        "eligible_count": len(normalized),
        "retained": [dict(item) for item in retained],
        "winner": dict(winner) if winner is not None else None,
    }
    frozen_configuration_sha256 = None
    if winner is not None:
        configuration = winner.get("configuration")
        if configuration is None:
            configuration = {
                name: winner[name]
                for name in ("name", "k", "active_loss_terms", "epoch")
            }
        if not isinstance(configuration, Mapping):
            raise ValueError("winner configuration must be an object")
        frozen_configuration_sha256 = hashlib.sha256(
            strict_json_dumps(
                dict(configuration), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
    return _decision(
        3, {"candidates": [dict(item) for item in candidates], "split": split},
        comparisons, {"eligible_candidate": winner is not None},
        study_sha=study_sha, phase_sha=phase_sha,
        frozen_configuration_sha256=frozen_configuration_sha256,
    )


def _confirmation_values(payload: Mapping[str, object]) -> dict[str, object]:
    samples = _require_mapping_value(payload, "bootstrap_samples")
    seed = _require_mapping_value(payload, "bootstrap_seed")
    if type(samples) is not int or type(seed) is not int:
        raise ValueError("bootstrap_samples and bootstrap_seed must be integers")
    if samples != 10_000 or seed != 0:
        raise ValueError("confirmation gates require 10,000 samples and seed 0")
    for field in ("directional_auroc_delta", "directional_aurc_delta"):
        value = _require_mapping_value(payload, field)
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be an object")
        for direction in _DIRECTIONS:
            _decimal(_require_mapping_value(value, direction), f"{field}.{direction}")
    result: dict[str, object] = {}
    for field in ("mean_auroc_delta_ci95", "mean_aurc_delta_ci95"):
        value = _require_mapping_value(payload, field)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{field} must contain exactly two scalar bounds")
        low = _decimal(value[0], f"{field}[0]")
        high = _decimal(value[1], f"{field}[1]")
        if low > high:
            raise ValueError(f"{field} bounds are reversed")
        result[field] = (low, high)
    result["directional_auroc_delta"] = payload["directional_auroc_delta"]
    result["directional_aurc_delta"] = payload["directional_aurc_delta"]
    return result


def _evaluate_confirmation_gate(
    stage: int,
    payload: Mapping[str, object],
    *,
    split: str,
    study_sha: str | None,
    phase_sha: str | None,
    retrieval_match: bool | None = None,
    frozen_configuration_sha256: str | None = None,
) -> GateDecision:
    _require_validation(split)
    values = _confirmation_values(payload)
    auroc_directional = all(
        _decimal(payload["directional_auroc_delta"][direction], "auroc delta") > 0  # type: ignore[index]
        for direction in _DIRECTIONS
    )
    aurc_directional = all(
        _decimal(payload["directional_aurc_delta"][direction], "aurc delta") < 0  # type: ignore[index]
        for direction in _DIRECTIONS
    )
    auroc_ci = values["mean_auroc_delta_ci95"][0] > 0  # type: ignore[index]
    aurc_ci = values["mean_aurc_delta_ci95"][1] < 0  # type: ignore[index]
    checks = {
        "directional_point_estimates_and_ci": (
            (auroc_directional and auroc_ci) or (aurc_directional and aurc_ci)
        )
    }
    if retrieval_match is not None:
        checks["frozen_retrieval_exact_match"] = retrieval_match
    comparisons = {
        "auroc_directional_improves": auroc_directional,
        "aurc_directional_improves": aurc_directional,
        "auroc_ci_strictly_positive": auroc_ci,
        "aurc_ci_strictly_negative": aurc_ci,
    }
    return _decision(
        stage, {**dict(payload), "split": split}, comparisons, checks,
        study_sha=study_sha, phase_sha=phase_sha,
        frozen_configuration_sha256=frozen_configuration_sha256,
    )


def evaluate_stage4_gate(
    payload: Mapping[str, object], *, split: str,
    study_sha: str | None = None, phase_sha: str | None = None,
) -> GateDecision:
    return _evaluate_confirmation_gate(
        4, payload, split=split, study_sha=study_sha, phase_sha=phase_sha
    )


def evaluate_stage5_gate(
    payload: Mapping[str, object], *, split: str,
    sim_matrix_sha256_frozen: str,
    sim_matrix_sha256_candidate: str,
    study_sha: str | None = None, phase_sha: str | None = None,
    frozen_configuration_sha256: str | None = None,
) -> GateDecision:
    """Evaluate the Stage 5 confirmation gate with a computed retrieval-identity check.

    The frozen-retriever contract is verified by comparing the SHA-256 of the
    pre-DSL similarity matrix produced under a canonical evaluator (see
    ``scripts/verify_frozen_retrieval_equivalence.py``). Byte-equal similarity
    matrices imply byte-equal ranks and byte-equal Recall@k on the same
    dataset, so the gate no longer accepts operator-asserted booleans.
    """

    if (
        not isinstance(sim_matrix_sha256_frozen, str)
        or not isinstance(sim_matrix_sha256_candidate, str)
        or len(sim_matrix_sha256_frozen) != _SHA256_LENGTH
        or len(sim_matrix_sha256_candidate) != _SHA256_LENGTH
        or any(c not in "0123456789abcdef" for c in sim_matrix_sha256_frozen.lower())
        or any(c not in "0123456789abcdef" for c in sim_matrix_sha256_candidate.lower())
    ):
        raise ValueError(
            "Stage 5 retrieval identity requires two 64-hex SHA-256 hashes: "
            "frozen retriever and candidate must both be provided"
        )
    identity_match = sim_matrix_sha256_frozen.lower() == sim_matrix_sha256_candidate.lower()
    return _evaluate_confirmation_gate(
        5, payload, split=split, study_sha=study_sha, phase_sha=phase_sha,
        retrieval_match=identity_match,
        frozen_configuration_sha256=frozen_configuration_sha256,
    )


@dataclass(frozen=True)
class CommittedPhase:
    path: Path
    payload: dict[str, object]
    sha256: str


def _repository_relative(path: Path, field: str) -> Path:
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"{field} must be repository-relative and traversal-free")
    return path


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _committed_bytes(repo: Path, relative: Path, description: str) -> bytes:
    relative = _repository_relative(relative, description)
    try:
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=repo,
            check=True, capture_output=True,
        ).stdout
        disk = (repo / relative).read_bytes()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"{description} must exist in committed HEAD") from exc
    if disk != committed:
        raise ValueError(f"{description} must exactly match its committed HEAD bytes")
    return committed


def load_phase_spec(repo: Path, path: Path) -> CommittedPhase:
    relative = _repository_relative(path, "phase path")
    content = _committed_bytes(repo.resolve(), relative, "phase")
    try:
        payload = strict_json_loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError) as exc:
        raise ValueError("phase must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"phase", "study", "runs"}:
        raise ValueError("phase must contain exactly phase, study, and runs")
    if not isinstance(payload["phase"], str) or not payload["phase"]:
        raise ValueError("phase name must be non-empty")
    if payload["study"] != "contextual-risk-v2":
        raise ValueError("phase study must be contextual-risk-v2")
    runs = payload["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("phase runs must be a non-empty list")
    identities = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or set(run) != _PHASE_RUN_FIELDS:
            raise ValueError(f"phase runs[{index}] must contain the complete run contract")
        identity = tuple(
            run[name]
            for name in ("dataset", "stage", "variant", "seed", "scheduled_timestamp_utc")
        )
        identities.append(identity)
        if run["test_authorized"] and run["stage"] != "final":
            raise ValueError("test authorization is allowed only for final runs")
        if not run["validation_authorized"]:
            raise ValueError("scientific phase runs must authorize validation")
    if len(identities) != len(set(identities)):
        raise ValueError("phase contains duplicate canonical run identity")
    return CommittedPhase(relative, payload, hashlib.sha256(content).hexdigest())


def _phase_run_identity(run: Mapping[str, object]) -> tuple:
    """Canonical identity tuple for a phase run.

    Matches :func:`load_phase_spec`'s uniqueness key so that phases with
    repeated ``variant`` values (e.g. Stage 7's dual ``frozen_test`` entries
    for MSRVTT and MSVD) do not collide.
    """

    return tuple(
        run[name]
        for name in ("dataset", "stage", "variant", "seed", "scheduled_timestamp_utc")
    )


def phase_run_results(
    phase: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    run_ids_by_identity: Mapping[tuple, str],
) -> dict[tuple, Mapping[str, object]]:
    """Collect terminal records for every run in a phase, keyed by identity.

    The result dictionary is keyed by the same 5-tuple identity that
    :func:`load_phase_spec` uses to guarantee uniqueness. Variant strings
    alone are insufficient: Stage 7 legitimately has two ``frozen_test``
    runs (MSRVTT + MSVD) that would silently collide under variant keying.
    """

    result: dict[tuple, Mapping[str, object]] = {}
    for run in phase.get("runs", []):  # type: ignore[union-attr]
        if not isinstance(run, Mapping):
            raise ValueError("phase run must be an object")
        identity = _phase_run_identity(run)
        if identity not in run_ids_by_identity:
            raise ValueError(f"missing run identity for phase entry {identity}")
        run_id = run_ids_by_identity[identity]
        matching = [record for record in records if record.get("run_id") == run_id]
        reservations = [record for record in matching if record.get("event") == "reserved"]
        terminals = [record for record in matching if record.get("event") in _TERMINAL_EVENTS]
        if len(reservations) != 1:
            qualifier = "duplicate" if len(reservations) > 1 else "missing"
            raise ValueError(f"{qualifier} reservation for {run_id}")
        if len(terminals) != 1:
            qualifier = "duplicate" if len(terminals) > 1 else "missing terminal"
            raise ValueError(f"{qualifier} record for {run_id}")
        terminal = terminals[0]
        if terminal.get("event") != "completed":
            raise ValueError(f"phase worker {run_id} failed or was cancelled")
        if terminal.get("manifest_sha256") != reservations[0].get("manifest_sha256"):
            raise ValueError(f"manifest changed between reservation and terminal for {run_id}")
        result[identity] = terminal
    return result


class TrainingLease:
    """A non-blocking cross-process lease for the single GPU training slot."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def __enter__(self) -> "TrainingLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError("a four-GPU scientific training is already active") from exc
        self._handle = handle
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._handle is not None
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


Runner = Callable[..., subprocess.CompletedProcess]


class CampaignController:
    STUDY_PATH = Path("experiments/contextual-risk-v2/study.yaml")
    REGISTRY_PATH = Path("experiments/contextual-risk-v2/registry.jsonl")

    def __init__(
        self,
        repo_root: Path,
        *,
        artifact_root: Path | None = None,
        runner: Runner = subprocess.run,
        expected_study_sha: str | None = None,
    ):
        self.repo_root = repo_root.resolve()
        common_dir = Path(
            _git(self.repo_root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
        )
        self.main_repo_root = common_dir.parent.resolve()
        canonical_artifact_root = self.main_repo_root / "artifacts"
        if artifact_root is not None and artifact_root.resolve() != canonical_artifact_root:
            raise ValueError("artifact root must be the canonical main-repository artifacts path")
        self.artifact_root = canonical_artifact_root
        self.runner = runner
        study_bytes = _committed_bytes(self.repo_root, self.STUDY_PATH, "study")
        self.study_sha = hashlib.sha256(study_bytes).hexdigest()
        self.protocol_frozen = b"protocol_version:" in study_bytes
        self.expected_study_sha = _binding_sha(
            expected_study_sha, "expected_study_sha"
        ) if expected_study_sha is not None else self.study_sha

    @property
    def manager(self) -> Path:
        return self.repo_root / "scripts/manage_experiment_run.py"

    def _clean(self) -> None:
        for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
            if subprocess.run(["git", *arguments], cwd=self.repo_root).returncode:
                raise ValueError("campaign operation requires clean tracked source")

    def _manager_call(self, command: str, *arguments: str) -> dict[str, object]:
        completed = self.runner(
            [sys.executable, str(self.manager), command, *arguments],
            cwd=self.repo_root, check=False, capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or f"archive {command} failed")
        try:
            payload = strict_json_loads(completed.stdout or "{}")
        except (json.JSONDecodeError, StrictJSONError) as exc:
            raise ValueError(f"archive {command} returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"archive {command} response must be an object")
        return payload

    def _assert_stage0_qualified(self) -> None:
        study_text = _committed_bytes(self.repo_root, self.STUDY_PATH, "study").decode("utf-8")
        if "protocol_version:" not in study_text:
            return
        summary_path = Path(
            "experiments/contextual-risk-v2/summaries/stage0-production-archive-qualification.json"
        )
        summary_bytes = _committed_bytes(self.repo_root, summary_path, "Stage 0 summary")
        summary = strict_json_loads(summary_bytes.decode("utf-8"))
        runs = summary.get("runs", []) if isinstance(summary, Mapping) else []
        if not any(
            isinstance(run, Mapping)
            and run.get("state") == "completed"
            and run.get("summary_metrics", {}).get("qualification_loss") is not None
            for run in runs
        ):
            raise ValueError("Stage 0 production archive qualification has not passed")

    def prepare(self, phase_path: Path, *, allow_test: bool = False) -> dict[str, object]:
        self._clean()
        self._assert_stage0_qualified()
        phase = load_phase_spec(self.repo_root, phase_path)
        self.assert_phase_authorized(phase)
        self.assert_retry_lineage(phase)
        has_test = any(run["test_authorized"] for run in phase.payload["runs"])
        if has_test and not allow_test:
            raise ValueError("test phase must be reserved through authorize-test")
        return self._manager_call(
            "reserve-phase",
            "--repo-root", str(self.repo_root),
            "--phase", phase.path.as_posix(),
            "--registry", self.REGISTRY_PATH.as_posix(),
            "--artifact-root", str(self.artifact_root),
        )

    def assert_phase_authorized(self, phase: CommittedPhase) -> None:
        """Require the exact committed predecessor gate for Stages 2 through 6."""

        if not self.protocol_frozen:
            return
        phase_name = str(phase.payload["phase"])
        try:
            stage = int(phase_name.removeprefix("stage").split("-", 1)[0])
        except ValueError as exc:
            raise ValueError("campaign phase name must begin with stage1 through stage7") from exc
        if stage <= 1 or stage == 7:
            return
        previous_paths = {
            1: Path("experiments/contextual-risk-v2/phases/stage1-baselines.json"),
            2: Path("experiments/contextual-risk-v2/phases/stage2-features.json"),
            3: Path("experiments/contextual-risk-v2/phases/stage3-loss-topk.json"),
            4: Path("experiments/contextual-risk-v2/phases/stage4-multiseed.json"),
            5: Path("experiments/contextual-risk-v2/phases/stage5-msvd.json"),
        }
        previous_stage = stage - 1
        gate_path = Path(
            f"experiments/contextual-risk-v2/summaries/stage{previous_stage}-gate.json"
        )
        try:
            previous = load_phase_spec(self.repo_root, previous_paths[previous_stage])
            decision = strict_json_loads(
                _committed_bytes(
                    self.repo_root, gate_path, f"Stage {previous_stage} gate"
                ).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError, ValueError) as exc:
            raise ValueError(
                f"Stage {previous_stage} gate is not committed and valid"
            ) from exc
        if not isinstance(decision, Mapping) or (
            decision.get("stage") != previous_stage
            or decision.get("passed") is not True
            or decision.get("next_authorized_stage") != stage
            or decision.get("study_sha") != self.study_sha
            or decision.get("phase_sha") != previous.sha256
        ):
            raise ValueError(
                f"Stage {previous_stage} gate does not authorize exact Stage {stage} preparation"
            )

    def assert_retry_lineage(self, phase: CommittedPhase) -> None:
        """Require tracked human authorization for successors of terminal failures."""

        try:
            registry_text = _committed_bytes(
                self.repo_root, self.REGISTRY_PATH, "registry"
            ).decode("utf-8")
            records = [
                strict_json_loads(line)
                for line in registry_text.splitlines() if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError) as exc:
            raise ValueError("committed registry is invalid") from exc
        terminal_by_run = {
            str(record["run_id"]): str(record["event"])
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("run_id"), str)
            and record.get("event") in _TERMINAL_EVENTS
        }
        for run in phase.payload["runs"]:
            failed_parents = [
                parent for parent in run["parent_run_ids"]
                if terminal_by_run.get(parent) in {"failed", "cancelled"}
            ]
            if not failed_parents:
                continue
            authorized = False
            for item in run["selection_lineage"]:
                if not isinstance(item, Mapping) or not item.get("authorization"):
                    continue
                report = item.get("report")
                commit = item.get("commit")
                if isinstance(report, str):
                    path = _repository_relative(Path(report), "retry report")
                    if "sdd" in path.parts:
                        raise ValueError("retry lineage cannot reference worktree sdd state")
                    exists = subprocess.run(
                        ["git", "cat-file", "-e", f"HEAD:{path.as_posix()}"],
                        cwd=self.repo_root, capture_output=True,
                    ).returncode == 0
                    authorized = exists
                elif isinstance(commit, str):
                    authorized = subprocess.run(
                        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                        cwd=self.repo_root, capture_output=True,
                    ).returncode == 0
                if authorized:
                    break
            if not authorized:
                raise ValueError(
                    "failed/cancelled successor requires tracked human authorization "
                    f"for parent(s): {', '.join(failed_parents)}"
                )

    def _verify_reservation(self, reservation: Mapping[str, object]) -> dict[str, object]:
        required = ("run_id", "artifact_path", "manifest_sha256")
        if any(name not in reservation for name in required):
            raise ValueError("launch reservation identity is incomplete")
        result = self._manager_call(
            "verify-run",
            "--repo-root", str(self.repo_root),
            "--registry", self.REGISTRY_PATH.as_posix(),
            "--run-dir", str(reservation["artifact_path"]),
            "--run-id", str(reservation["run_id"]),
            "--manifest-sha256", str(reservation["manifest_sha256"]),
        )
        for field, response_name in (
            ("run_id", "run_id"), ("artifact_path", "run_dir"),
            ("manifest_sha256", "manifest_sha256"),
        ):
            if str(reservation[field]) != str(result.get(response_name)):
                raise ValueError(f"verified {response_name} differs from exact reservation manifest identity")
        return result

    def reservations_for_phase(self, phase_path: Path) -> list[dict[str, object]]:
        phase = load_phase_spec(self.repo_root, phase_path)
        try:
            records_payload = _committed_bytes(
                self.repo_root, self.REGISTRY_PATH, "registry"
            ).decode("utf-8")
            records = [
                strict_json_loads(line)
                for line in records_payload.splitlines() if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError) as exc:
            raise ValueError("committed registry is invalid") from exc
        reservations = []
        for spec in phase.payload["runs"]:
            prefix = (
                f"{spec['dataset']}-{spec['stage']}-{spec['variant']}-"
                f"seed{int(spec['seed']):03d}-{spec['scheduled_timestamp_utc']}-"
            )
            matches = [
                record for record in records
                if isinstance(record, Mapping)
                and record.get("event") == "reserved"
                and str(record.get("run_id", "")).startswith(prefix)
            ]
            if len(matches) != 1:
                qualifier = "duplicate" if len(matches) > 1 else "missing"
                raise ValueError(f"{qualifier} committed reservation for {prefix}")
            reservation = dict(matches[0])
            reservation["dataset"] = spec["dataset"]
            reservation["environment"] = spec["environment"]
            reservations.append(reservation)
        return reservations

    @staticmethod
    def _validate_training_capability(
        reservation: Mapping[str, object], verified: Mapping[str, object]
    ) -> list[str]:
        if verified.get("state") != "reserved":
            raise ValueError("scientific launch requires an exact reserved run")
        # Test-authorization is enforced upstream by _assert_test_launch_authorized.
        # This function verifies training-capability invariants that hold for every
        # reservation regardless of test authorization status.
        if reservation.get("dataset") == "msvd":
            raise ValueError(
                "artifact-aware MSVD launcher is not qualified; Stage 5/7 must remain sealed"
            )
        environment = reservation.get("environment")
        if not isinstance(environment, Mapping) or (
            environment.get("gpu_count") != 4
            or environment.get("precision") != "fp32"
            or environment.get("distributed_topology") != "torchrun-4"
            or "A800" not in str(environment.get("gpu_model", ""))
        ):
            raise ValueError("scientific training requires the frozen four-GPU A800 fp32 topology")
        command = verified.get("command_argv")
        if (
            not isinstance(command, list)
            or len(command) != 4
            or tuple(command[-2:]) != _A3_COMMAND_SUFFIX
            or command[0] != "env"
            or command[1] != f"ARTIFACT_RUN_ID={verified.get('run_id')}"
        ):
            raise ValueError("scientific training command is not the qualified sealed launcher")
        return [str(item) for item in command]

    @staticmethod
    def _launch_environment() -> dict[str, str]:
        protected_prefixes = ("CUDA_", "NCCL_", "TORCH_NCCL_", "CUBLAS_")
        protected_names = {"NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"}
        return {
            name: value for name, value in os.environ.items()
            if name not in protected_names
            and not any(name.startswith(prefix) for prefix in protected_prefixes)
        }

    def launch_reserved(
        self,
        reservations: Sequence[Mapping[str, object]],
        *,
        phase: "CommittedPhase | Path | None" = None,
    ) -> list[dict[str, object]]:
        self._clean()
        if not reservations:
            raise ValueError("launch requires at least one committed reservation")
        verified_reservations = [
            (reservation, self._verify_reservation(reservation))
            for reservation in reservations
        ]
        if any(verified.get("test_authorized") for _, verified in verified_reservations):
            if phase is None:
                raise ValueError(
                    "test-authorized launch requires the committed phase and prior "
                    "authorize-test evidence"
                )
            if isinstance(phase, Path):
                phase = load_phase_spec(self.repo_root, phase)
            self._assert_test_launch_authorized(phase)
        prepared = [
            (reservation, verified, self._validate_training_capability(reservation, verified))
            for reservation, verified in verified_reservations
        ]
        results = []
        lock = self.artifact_root / ".locks/contextual-risk-v2-training.lock"
        with TrainingLease(lock):
            for _reservation, verified, command in prepared:
                completed = self.runner(
                    command, cwd=self.repo_root, env=self._launch_environment(),
                    check=False, capture_output=True, text=True,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"scientific run {verified['run_id']} exited {completed.returncode}; "
                        "evidence is preserved and the terminal run cannot be retried in place"
                    )
                results.append({"run_id": verified["run_id"], "returncode": 0})
        return results

    def monitor(self, run_dirs: Sequence[Path]) -> list[dict[str, object]]:
        results = []
        for run_dir in run_dirs:
            path = run_dir / "status.json"
            try:
                payload = strict_json_loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, StrictJSONError) as exc:
                raise ValueError(f"run status is unavailable or invalid: {path}") from exc
            if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), str):
                raise ValueError(f"run status is incomplete: {path}")
            result = {"run_dir": str(run_dir), "state": payload["state"]}
            if "updated_at" in payload:
                result["updated_at"] = payload["updated_at"]
            results.append(result)
        return results

    def finalize(self, phase_path: Path, summary_path: Path) -> dict[str, object]:
        self._clean()
        phase = load_phase_spec(self.repo_root, phase_path)
        summary = _repository_relative(summary_path, "summary path")
        return self._manager_call(
            "finalize-phase", "--repo-root", str(self.repo_root),
            "--phase", phase.path.as_posix(),
            "--registry", self.REGISTRY_PATH.as_posix(),
            "--artifact-root", str(self.artifact_root),
            "--summary", summary.as_posix(),
        )

    def _assert_test_launch_authorized(self, phase: CommittedPhase) -> str:
        """Enforce Stage 5 completion and frozen-configuration binding at launch.

        Derives the frozen configuration hash from the committed Stage 7 phase
        specification, then re-executes the same proofs :meth:`authorize_test`
        executes. This makes ``launch`` self-authorizing for
        ``test_authorized`` reservations rather than relying on an unverified
        CLI flag.
        """

        runs = phase.payload["runs"]
        if not all(run["stage"] == "final" and run["test_authorized"] is True for run in runs):
            raise ValueError(
                "test-authorized launch requires the committed phase to declare "
                "every run as an explicit final test reservation"
            )
        lineage_hashes: set[str] = set()
        for run in runs:
            for item in run["selection_lineage"]:
                if not isinstance(item, Mapping):
                    continue
                value = item.get("frozen_configuration_sha256")
                if isinstance(value, str):
                    lineage_hashes.add(value)
        if not lineage_hashes:
            raise ValueError(
                "test-authorized launch requires committed frozen-configuration "
                "lineage on every Stage 7 run"
            )
        if len(lineage_hashes) != 1:
            raise ValueError(
                "test-authorized launch requires a single frozen-configuration "
                "hash across all Stage 7 runs"
            )
        frozen_hash = lineage_hashes.pop()
        if not (isinstance(frozen_hash, str) and len(frozen_hash) == _SHA256_LENGTH):
            raise ValueError(
                "test-authorized launch requires a committed 64-hex frozen "
                "configuration hash (found: 'pending-stage5-freeze' or "
                "malformed hash); rerun authorize-test after Stage 5 freezes"
            )
        decision_path = Path(
            "experiments/contextual-risk-v2/summaries/stage5-gate.json"
        )
        try:
            decision_bytes = _committed_bytes(
                self.repo_root, decision_path, "Stage 5 decision"
            )
            decision = strict_json_loads(decision_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError, ValueError) as exc:
            raise ValueError(
                "test-authorized launch requires the committed passing Stage 5 gate"
            ) from exc
        if (
            not isinstance(decision, Mapping)
            or decision.get("stage") != 5
            or decision.get("passed") is not True
        ):
            raise ValueError(
                "test-authorized launch requires the committed passing Stage 5 gate"
            )
        if decision.get("study_sha") != self.expected_study_sha:
            raise ValueError(
                "test-authorized launch: Stage 5 gate study SHA differs from the frozen study"
            )
        if decision.get("frozen_configuration_sha256") != frozen_hash:
            raise ValueError(
                "test-authorized launch: Stage 5 frozen configuration hash differs from the phase lineage"
            )
        self._assert_stage5_completed()
        return frozen_hash

    def authorize_test(
        self,
        phase_path: Path,
        stage5_decision_path: Path,
        frozen_configuration_sha256: str,
    ) -> dict[str, object]:
        self._clean()
        frozen = _binding_sha(frozen_configuration_sha256, "frozen configuration")
        decision_bytes = _committed_bytes(
            self.repo_root,
            _repository_relative(stage5_decision_path, "Stage 5 decision path"),
            "Stage 5 decision",
        )
        decision = strict_json_loads(decision_bytes.decode("utf-8"))
        if not isinstance(decision, Mapping) or decision.get("stage") != 5 or decision.get("passed") is not True:
            raise ValueError("authorize-test requires the committed passing Stage 5 gate")
        self._assert_stage5_completed()
        if decision.get("study_sha") != self.expected_study_sha:
            raise ValueError("Stage 5 gate study SHA differs from the frozen study")
        if decision.get("frozen_configuration_sha256") != frozen:
            raise ValueError("Stage 5 frozen configuration hash differs")
        phase = load_phase_spec(self.repo_root, phase_path)
        runs = phase.payload["runs"]
        if not all(run["stage"] == "final" and run["test_authorized"] is True for run in runs):
            raise ValueError("authorize-test requires an explicit Stage 7 final reservation")
        for run in runs:
            lineage_hashes = {
                item.get("frozen_configuration_sha256")
                for item in run["selection_lineage"] if isinstance(item, Mapping)
            }
            if frozen not in lineage_hashes:
                raise ValueError("Stage 7 reservation lacks frozen configuration lineage")
        return self.prepare(phase_path, allow_test=True)

    def _assert_stage5_completed(self) -> None:
        summary_path = Path("experiments/contextual-risk-v2/summaries/stage5-msvd.json")
        try:
            summary = strict_json_loads(
                _committed_bytes(
                    self.repo_root, summary_path, "Stage 5 summary"
                ).decode("utf-8")
            )
            registry_text = _committed_bytes(
                self.repo_root, self.REGISTRY_PATH, "registry"
            ).decode("utf-8")
            records = [
                strict_json_loads(line)
                for line in registry_text.splitlines() if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError, StrictJSONError, ValueError) as exc:
            raise ValueError("authorize-test requires a completed Stage 5") from exc
        if not isinstance(summary, Mapping):
            raise ValueError("authorize-test requires a completed Stage 5")
        runs = summary.get("runs")
        counts = summary.get("terminal_counts")
        if (
            summary.get("phase") != "stage5-msvd"
            or summary.get("study") != "contextual-risk-v2"
            or not isinstance(runs, list)
            or not runs
            or not isinstance(counts, Mapping)
            or counts.get("completed") != len(runs)
            or any(name != "completed" and value for name, value in counts.items())
        ):
            raise ValueError("authorize-test requires a completed Stage 5")
        for run in runs:
            if (
                not isinstance(run, Mapping)
                or run.get("state") != "completed"
                or not isinstance(run.get("run_id"), str)
                or not isinstance(run.get("manifest_sha256"), str)
            ):
                raise ValueError("authorize-test requires a completed Stage 5")
            matching = [
                record for record in records
                if isinstance(record, Mapping)
                and record.get("run_id") == run["run_id"]
                and record.get("manifest_sha256") == run["manifest_sha256"]
            ]
            reservations = [item for item in matching if item.get("event") == "reserved"]
            completed = [item for item in matching if item.get("event") == "completed"]
            other_terminals = [
                item for item in matching if item.get("event") in {"failed", "cancelled"}
            ]
            if len(reservations) != 1 or len(completed) != 1 or other_terminals:
                raise ValueError("authorize-test requires a completed Stage 5")
