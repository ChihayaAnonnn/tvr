from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from prob_models.reliability_campaign import (
    CampaignController,
    GateDecision,
    TrainingLease,
    evaluate_stage1_gate,
    evaluate_stage2_gate,
    evaluate_stage3_gate,
    evaluate_stage4_gate,
    evaluate_stage5_gate,
    load_phase_spec,
    phase_run_results,
)


DIGEST = "a" * 64
DIRECTIONS = ("t2v", "v2t")


def _risk(mean_auroc: float, mean_aurc: float, *, auroc=0.8, rho=0.3):
    return {
        "mean_auroc": mean_auroc,
        "mean_aurc": mean_aurc,
        "t2v": {"auroc": auroc, "rho": rho},
        "v2t": {"auroc": auroc, "rho": rho},
    }


def test_stage1_gate_uses_validation_only_and_exact_threshold():
    result = evaluate_stage1_gate(
        a3=_risk(0.82, 0.25, auroc=0.81, rho=0.3),
        strongest={"mean_auroc": 0.81, "mean_aurc": 0.24},
        split="validation",
        study_sha=DIGEST,
        phase_sha="b" * 64,
    )

    assert isinstance(result, GateDecision)
    assert result.passed
    assert result.study_sha == DIGEST
    assert result.phase_sha == "b" * 64
    assert result.next_authorized_stage == 2
    assert result.to_dict()["comparisons"]["mean_auroc_delta"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="validation"):
        evaluate_stage1_gate(_risk(1.0, 0.0), {}, split="test")


@pytest.mark.parametrize(
    "a3,failed_check",
    [
        (_risk(0.8199, 0.2301), "a3_improves_strongest"),
        (_risk(0.82, 0.25, auroc=0.5), "directional_auroc_gt_0_5"),
        (_risk(0.82, 0.25, rho=0.1999), "directional_rho_gte_0_20"),
    ],
)
def test_stage1_gate_fails_closed_at_strict_or_missed_boundaries(a3, failed_check):
    decision = evaluate_stage1_gate(
        a3, {"mean_auroc": 0.81, "mean_aurc": 0.24}, split="validation"
    )
    assert not decision.passed
    assert failed_check in decision.failed_checks


def test_stage1_aurc_boundary_is_inclusive():
    decision = evaluate_stage1_gate(
        _risk(0.80, 0.23),
        {"mean_auroc": 0.81, "mean_aurc": 0.24},
        split="validation",
    )
    assert decision.passed


def test_stage2_requires_both_comparators_directional_floor_and_freeze_audit():
    full = _risk(0.806, 0.295, auroc=0.79)
    score_only = _risk(0.800, 0.300, auroc=0.80)
    no_context = _risk(0.801, 0.300, auroc=0.80)

    assert evaluate_stage2_gate(
        full, score_only, no_context,
        changed_tensors="a3_sidecar_only", split="validation",
    ).passed
    assert not evaluate_stage2_gate(
        full, score_only, _risk(0.802, 0.299, auroc=0.80),
        changed_tensors="a3_sidecar_only", split="validation",
    ).passed
    assert not evaluate_stage2_gate(
        _risk(0.806, 0.295, auroc=0.7899), score_only, no_context,
        changed_tensors="a3_sidecar_only", split="validation",
    ).passed
    assert not evaluate_stage2_gate(
        full, score_only, no_context,
        changed_tensors="retriever", split="validation",
    ).passed


def test_stage3_retains_pareto_band_then_uses_frozen_tie_break_order():
    candidates = [
        {"name": "large", "mean_auroc": 0.820, "mean_aurc": 0.200,
         "t2v": {"rho": 0.2}, "v2t": {"rho": 0.3},
         "k": 16, "active_loss_terms": 3, "epoch": 2},
        {"name": "small_late", "mean_auroc": 0.815, "mean_aurc": 0.205,
         "t2v": {"rho": 0.2}, "v2t": {"rho": 0.3},
         "k": 4, "active_loss_terms": 1, "epoch": 3},
        {"name": "small_early", "mean_auroc": 0.815, "mean_aurc": 0.205,
         "t2v": {"rho": 0.2}, "v2t": {"rho": 0.3},
         "k": 4, "active_loss_terms": 1, "epoch": 1},
        {"name": "bad_rho", "mean_auroc": 0.9, "mean_aurc": 0.1,
         "t2v": {"rho": 0.199}, "v2t": {"rho": 0.4},
         "k": 1, "active_loss_terms": 0, "epoch": 0},
    ]

    decision = evaluate_stage3_gate(candidates, split="validation")

    assert decision.passed
    assert decision.comparisons["winner"]["name"] == "small_early"
    assert len(decision.frozen_configuration_sha256) == 64
    assert decision.to_dict()["frozen_configuration_sha256"] == decision.frozen_configuration_sha256
    assert [item["name"] for item in decision.comparisons["retained"]] == [
        "large", "small_late", "small_early"
    ]


def test_stage3_fails_if_no_candidate_meets_directional_rho_floor():
    candidate = {
        "name": "bad", "mean_auroc": 0.9, "mean_aurc": 0.1,
        "t2v": {"rho": 0.19}, "v2t": {"rho": 0.3},
        "k": 1, "active_loss_terms": 0, "epoch": 0,
    }
    decision = evaluate_stage3_gate([candidate], split="validation")
    assert not decision.passed
    assert "eligible_candidate" in decision.failed_checks


def _confirmation(*, auroc=(0.01, 0.02), aurc=(-0.01, -0.02),
                  auroc_ci=(0.001, 0.02), aurc_ci=(-0.02, -0.001)):
    return {
        "directional_auroc_delta": dict(zip(DIRECTIONS, auroc)),
        "directional_aurc_delta": dict(zip(DIRECTIONS, aurc)),
        "mean_auroc_delta_ci95": list(auroc_ci),
        "mean_aurc_delta_ci95": list(aurc_ci),
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 0,
    }


def test_stage4_requires_directional_point_improvement_and_strict_ci():
    assert evaluate_stage4_gate(_confirmation(), split="validation").passed
    assert not evaluate_stage4_gate(
        _confirmation(auroc_ci=(0.0, 0.02), aurc_ci=(-0.02, 0.0)),
        split="validation",
    ).passed
    assert not evaluate_stage4_gate(
        _confirmation(auroc=(0.0, 0.02), aurc=(0.0, -0.02)),
        split="validation",
    ).passed


def test_stage4_accepts_one_complete_metric_pathway():
    auroc_only = _confirmation(aurc=(0.01, -0.02), aurc_ci=(-0.02, 0.0))
    aurc_only = _confirmation(auroc=(-0.01, 0.02), auroc_ci=(0.0, 0.02))
    assert evaluate_stage4_gate(auroc_only, split="validation").passed
    assert evaluate_stage4_gate(aurc_only, split="validation").passed


def test_stage4_rejects_missing_or_changed_frozen_bootstrap_protocol():
    missing = _confirmation()
    del missing["bootstrap_samples"]
    with pytest.raises(ValueError, match="bootstrap_samples"):
        evaluate_stage4_gate(missing, split="validation")
    changed = {**_confirmation(), "bootstrap_seed": 1}
    with pytest.raises(ValueError, match="10,000 samples and seed 0"):
        evaluate_stage4_gate(changed, split="validation")


def test_stage5_adds_frozen_retrieval_identity_requirement():
    payload = _confirmation()
    passed = evaluate_stage5_gate(
        payload, retrieval_ranks_match=True, recall_at_k_match=True,
        frozen_configuration_sha256=DIGEST, split="validation",
    )
    assert passed.passed
    assert passed.frozen_configuration_sha256 == DIGEST
    failed = evaluate_stage5_gate(
        payload, retrieval_ranks_match=True, recall_at_k_match=False,
        split="validation",
    )
    assert not failed.passed
    assert "frozen_retrieval_exact_match" in failed.failed_checks


def test_gate_inputs_reject_missing_nonfinite_and_nonscalar_values():
    with pytest.raises(ValueError, match="missing"):
        evaluate_stage1_gate({}, {}, split="validation")
    with pytest.raises(ValueError, match="finite"):
        evaluate_stage1_gate(
            _risk(float("nan"), 0.2),
            {"mean_auroc": 0.5, "mean_aurc": 0.3},
            split="validation",
        )
    with pytest.raises(ValueError, match="scalar"):
        evaluate_stage1_gate(
            {**_risk(0.8, 0.2), "raw": [1, 2]},
            {"mean_auroc": 0.5, "mean_aurc": 0.3},
            split="validation",
        )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def controller_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "experiments/contextual-risk-v2/phases").mkdir(parents=True)
    (repo / "experiments/contextual-risk-v2/summaries").mkdir(parents=True)
    (repo / "scripts").mkdir()
    study = repo / "experiments/contextual-risk-v2/study.yaml"
    registry = repo / "experiments/contextual-risk-v2/registry.jsonl"
    phase = repo / "experiments/contextual-risk-v2/phases/stage2.json"
    study.write_text("study: contextual-risk-v2\n", encoding="utf-8")
    registry.write_text("", encoding="utf-8")
    phase.write_text(json.dumps({
        "phase": "stage2-features", "study": "contextual-risk-v2",
        "runs": [_run_spec("score_only", "feature_ablation")],
    }), encoding="utf-8")
    (repo / "scripts/manage_experiment_run.py").write_text("# fixture\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    return repo


def _run_spec(variant="full", stage="feature_ablation", *, dataset="msrvtt",
              timestamp="20260815T120000Z", test_authorized=False):
    return {
        "dataset": dataset,
        "stage": stage,
        "variant": variant,
        "seed": 0,
        "scheduled_timestamp_utc": timestamp,
        "environment": {
            "python": "3.10.0", "pytorch": "2.1.2+cu121", "cuda": "12.1",
            "gpu_model": "NVIDIA A800 80GB PCIe", "gpu_count": 4,
            "precision": "fp32", "distributed_topology": "torchrun-4",
        },
        "input_checkpoints": [{"path": "/checkpoint", "sha256": DIGEST}],
        "dataset_splits": [{"path": "/split", "sha256": "b" * 64}],
        "config": {
            "model": {"contextual_risk_feature_set": variant},
            "loss": {"w_contextual_risk_fit": 0.25, "w_contextual_risk_rank": 0.25},
            "optimizer": {"seed": 0},
            "scheduler": {"epochs": 3},
            "evaluation": {"eval_split": "val", "run_final_test": test_authorized},
        },
        "parent_run_ids": [],
        "selection_lineage": [{"continuation_rule": "fresh_from_frozen_parent"}],
        "validation_authorized": True,
        "test_authorized": test_authorized,
        "expected_scalar_outputs": ["mean_auroc", "mean_aurc"],
        "array_references": [],
        "command": [
            "env", "ARTIFACT_RUN_ID={run_id}", "bash",
            "run_train_msrvtt_contextual_risk_bg.sh",
        ],
    }


def test_load_phase_requires_exact_committed_bytes_and_hashes(controller_repo):
    relative = Path("experiments/contextual-risk-v2/phases/stage2.json")
    loaded = load_phase_spec(controller_repo, relative)
    expected = hashlib.sha256((controller_repo / relative).read_bytes()).hexdigest()
    assert loaded.sha256 == expected
    assert loaded.payload["phase"] == "stage2-features"

    (controller_repo / relative).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="committed"):
        load_phase_spec(controller_repo, relative)


def test_load_phase_rejects_absolute_traversal_and_duplicate_identity(controller_repo):
    with pytest.raises(ValueError, match="repository-relative"):
        load_phase_spec(controller_repo, Path("/tmp/phase.json"))

    relative = Path("experiments/contextual-risk-v2/phases/stage2.json")
    payload = json.loads((controller_repo / relative).read_text())
    payload["runs"].append(dict(payload["runs"][0]))
    (controller_repo / relative).write_text(json.dumps(payload), encoding="utf-8")
    subprocess.run(["git", "add", relative], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "duplicate"], cwd=controller_repo, check=True)
    with pytest.raises(ValueError, match="duplicate"):
        load_phase_spec(controller_repo, relative)


def test_phase_run_results_rejects_missing_duplicate_failed_and_partial_phase():
    phase = {"runs": [_run_spec("one"), _run_spec("two", timestamp="20260815T120100Z")]}
    reservations = [
        {"run_id": "one", "event": "reserved", "manifest_sha256": DIGEST},
        {"run_id": "two", "event": "reserved", "manifest_sha256": "b" * 64},
    ]
    with pytest.raises(ValueError, match="missing terminal"):
        phase_run_results(phase, reservations, {"one": "one", "two": "two"})
    with pytest.raises(ValueError, match="failed"):
        phase_run_results(
            phase,
            reservations + [
                {"run_id": "one", "event": "failed", "manifest_sha256": DIGEST},
                {"run_id": "two", "event": "completed", "manifest_sha256": "b" * 64},
            ],
            {"one": "one", "two": "two"},
        )
    with pytest.raises(ValueError, match="duplicate"):
        phase_run_results(
            phase,
            reservations + [
                {"run_id": "one", "event": "completed", "manifest_sha256": DIGEST},
                {"run_id": "one", "event": "completed", "manifest_sha256": DIGEST},
                {"run_id": "two", "event": "completed", "manifest_sha256": "b" * 64},
            ],
            {"one": "one", "two": "two"},
        )


class RecordingRunner:
    def __init__(self, outputs=()):
        self.calls = []
        self.outputs = list(outputs)

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        stdout = self.outputs.pop(0) if self.outputs else "{}"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")


def test_prepare_delegates_to_archive_manager_with_main_repo_artifact_root(controller_repo):
    runner = RecordingRunner([json.dumps({"runs": []})])
    controller = CampaignController(controller_repo, runner=runner)
    phase = Path("experiments/contextual-risk-v2/phases/stage2.json")

    controller.prepare(phase)

    argv = runner.calls[0][0]
    assert argv[1].endswith("scripts/manage_experiment_run.py")
    assert argv[2] == "reserve-phase"
    assert argv[argv.index("--artifact-root") + 1] == str(controller_repo / "artifacts")
    assert argv[argv.index("--phase") + 1] == phase.as_posix()


def test_prepare_rejects_dirty_source_test_phase_and_wrong_artifact_root(controller_repo):
    phase = Path("experiments/contextual-risk-v2/phases/stage2.json")
    controller = CampaignController(controller_repo, runner=RecordingRunner())
    (controller_repo / "dirty.txt").write_text("dirty", encoding="utf-8")
    subprocess.run(["git", "add", "dirty.txt"], cwd=controller_repo, check=True)
    with pytest.raises(ValueError, match="clean"):
        controller.prepare(phase)
    subprocess.run(["git", "reset", "-q", "HEAD", "dirty.txt"], cwd=controller_repo, check=True)
    (controller_repo / "dirty.txt").unlink()

    with pytest.raises(ValueError, match="canonical main-repository"):
        CampaignController(controller_repo, artifact_root=controller_repo / "elsewhere")

    payload = json.loads((controller_repo / phase).read_text())
    payload["runs"][0]["stage"] = "final"
    payload["runs"][0]["test_authorized"] = True
    payload["runs"][0]["config"]["evaluation"]["run_final_test"] = True
    (controller_repo / phase).write_text(json.dumps(payload), encoding="utf-8")
    subprocess.run(["git", "add", phase], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test phase"], cwd=controller_repo, check=True)
    with pytest.raises(ValueError, match="authorize-test"):
        controller.prepare(phase)


def test_frozen_protocol_requires_exact_committed_preceding_gate(controller_repo):
    study = controller_repo / "experiments/contextual-risk-v2/study.yaml"
    study.write_text("study: contextual-risk-v2\nprotocol_version: 1\n", encoding="utf-8")
    subprocess.run(["git", "add", study], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze protocol"], cwd=controller_repo, check=True)
    controller = CampaignController(controller_repo, runner=RecordingRunner())
    stage2_path = Path("experiments/contextual-risk-v2/phases/stage2.json")
    stage2 = load_phase_spec(controller_repo, stage2_path)

    with pytest.raises(ValueError, match="Stage 1 gate"):
        controller.assert_phase_authorized(stage2)

    stage1_path = Path("experiments/contextual-risk-v2/phases/stage1-baselines.json")
    stage1_payload = json.loads((controller_repo / stage2_path).read_text())
    stage1_payload["phase"] = "stage1-baselines"
    (controller_repo / stage1_path).write_text(json.dumps(stage1_payload), encoding="utf-8")
    subprocess.run(["git", "add", stage1_path], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "stage1 phase"], cwd=controller_repo, check=True)
    stage1 = load_phase_spec(controller_repo, stage1_path)
    gate = Path("experiments/contextual-risk-v2/summaries/stage1-gate.json")
    (controller_repo / gate).write_text(json.dumps({
        "stage": 1, "passed": True, "next_authorized_stage": 2,
        "study_sha": controller.study_sha, "phase_sha": stage1.sha256,
    }), encoding="utf-8")
    subprocess.run(["git", "add", gate], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "pass stage1"], cwd=controller_repo, check=True)

    controller.assert_phase_authorized(load_phase_spec(controller_repo, stage2_path))


def test_cancelled_retry_requires_unique_successor_and_tracked_human_authorization(controller_repo):
    phase_path = Path("experiments/contextual-risk-v2/phases/stage2.json")
    payload = json.loads((controller_repo / phase_path).read_text())
    payload["runs"][0]["parent_run_ids"] = ["cancelled-parent"]
    payload["runs"][0]["selection_lineage"] = [{"continuation_rule": "retry"}]
    (controller_repo / phase_path).write_text(json.dumps(payload), encoding="utf-8")
    registry = controller_repo / "experiments/contextual-risk-v2/registry.jsonl"
    registry.write_text(json.dumps({
        "run_id": "cancelled-parent", "event": "cancelled",
        "manifest_sha256": DIGEST,
    }) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", phase_path, registry], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "unauthorized retry"], cwd=controller_repo, check=True)
    controller = CampaignController(controller_repo, runner=RecordingRunner())

    with pytest.raises(ValueError, match="human authorization"):
        controller.assert_retry_lineage(load_phase_spec(controller_repo, phase_path))

    report = Path("experiments/contextual-risk-v2/summaries/retry-review.md")
    (controller_repo / report).write_text("Human-approved infrastructure retry.\n", encoding="utf-8")
    payload["runs"][0]["selection_lineage"] = [{
        "continuation_rule": "retry", "authorization": "human approved",
        "report": report.as_posix(),
    }]
    (controller_repo / phase_path).write_text(json.dumps(payload), encoding="utf-8")
    subprocess.run(["git", "add", phase_path, report], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "authorize retry"], cwd=controller_repo, check=True)

    controller.assert_retry_lineage(load_phase_spec(controller_repo, phase_path))


def test_launch_composes_exact_sealed_command_and_rejects_changed_manifest(controller_repo):
    manifest_sha = "c" * 64
    run_id = "msrvtt-feature_ablation-score_only-seed000-20260815T120000Z-123456789abc"
    run_dir = controller_repo / "artifacts/contextual-risk-v2/msrvtt/feature_ablation" / run_id
    verify = json.dumps({
        "run_id": run_id, "run_dir": str(run_dir), "manifest_sha256": manifest_sha,
        "state": "reserved", "stage": "feature_ablation", "test_authorized": False,
        "command_argv": ["env", f"ARTIFACT_RUN_ID={run_id}", "bash",
                         "run_train_msrvtt_contextual_risk_bg.sh"],
        "registered_a3_variant_argv": ["--contextual_risk_feature_set", "score_only"],
    })
    runner = RecordingRunner([verify, "{}"])
    controller = CampaignController(controller_repo, runner=runner)
    reservation = {
        "run_id": run_id, "artifact_path": str(run_dir),
        "manifest_sha256": manifest_sha, "dataset": "msrvtt",
        "environment": _run_spec()["environment"],
    }

    result = controller.launch_reserved([reservation])

    assert result[0]["run_id"] == run_id
    assert runner.calls[1][0] == [
        "env", f"ARTIFACT_RUN_ID={run_id}", "bash",
        "run_train_msrvtt_contextual_risk_bg.sh",
    ]

    bad = RecordingRunner([verify.replace(manifest_sha, "d" * 64)])
    with pytest.raises(ValueError, match="manifest"):
        CampaignController(controller_repo, runner=bad).launch_reserved([reservation])


def test_launch_rejects_test_access_failed_state_wrong_gpu_shape_and_unqualified_msvd(controller_repo):
    run_id = "msrvtt-feature_ablation-full-seed000-20260815T120000Z-123456789abc"
    base = {
        "run_id": run_id, "run_dir": str(controller_repo / "artifacts/run"),
        "manifest_sha256": DIGEST, "state": "reserved",
        "stage": "feature_ablation", "test_authorized": False,
        "command_argv": ["env", f"ARTIFACT_RUN_ID={run_id}", "bash",
                         "run_train_msrvtt_contextual_risk_bg.sh"],
        "registered_a3_variant_argv": [],
    }
    reservation = {
        "run_id": run_id, "artifact_path": base["run_dir"],
        "manifest_sha256": DIGEST, "dataset": "msrvtt",
        "environment": _run_spec()["environment"],
    }
    for changed, message in [
        ({"state": "failed"}, "reserved"),
        ({"test_authorized": True, "stage": "final"}, "authorize-test"),
    ]:
        payload = {**base, **changed}
        with pytest.raises(ValueError, match=message):
            CampaignController(
                controller_repo, runner=RecordingRunner([json.dumps(payload)])
            ).launch_reserved([reservation])

    bad_hardware = {**reservation, "environment": {**reservation["environment"], "gpu_count": 3}}
    with pytest.raises(ValueError, match="four-GPU"):
        CampaignController(
            controller_repo, runner=RecordingRunner([json.dumps(base)])
        ).launch_reserved([bad_hardware])

    msvd = {**reservation, "dataset": "msvd"}
    with pytest.raises(ValueError, match="MSVD launcher"):
        CampaignController(
            controller_repo, runner=RecordingRunner([json.dumps(base)])
        ).launch_reserved([msvd])


def test_training_lease_allows_only_one_gpu_training(tmp_path):
    lock = tmp_path / "artifacts/.locks/contextual-risk-v2-training.lock"
    with TrainingLease(lock):
        with pytest.raises(RuntimeError, match="already active"):
            with TrainingLease(lock):
                pass
    with TrainingLease(lock):
        pass


def test_monitor_is_read_only(controller_repo, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status = run_dir / "status.json"
    status.write_text(json.dumps({"state": "running", "updated_at": "now"}), encoding="utf-8")
    before = status.stat().st_mtime_ns

    result = CampaignController(controller_repo).monitor([run_dir])

    assert result == [{"run_dir": str(run_dir), "state": "running", "updated_at": "now"}]
    assert status.stat().st_mtime_ns == before


def test_authorize_test_requires_completed_stage5_frozen_hash_and_explicit_stage7(controller_repo):
    phase = Path("experiments/contextual-risk-v2/phases/stage2.json")
    payload = json.loads((controller_repo / phase).read_text())
    payload["phase"] = "stage7-final"
    payload["runs"] = [_run_spec("final", "final", test_authorized=True)]
    payload["runs"][0]["selection_lineage"] = [{"frozen_configuration_sha256": DIGEST}]
    (controller_repo / phase).write_text(json.dumps(payload), encoding="utf-8")
    decision_path = Path("experiments/contextual-risk-v2/summaries/stage5-gate.json")
    decision = {
        "stage": 5, "passed": True, "next_authorized_stage": 6,
        "study_sha": DIGEST, "phase_sha": "b" * 64,
        "frozen_configuration_sha256": DIGEST,
    }
    (controller_repo / decision_path).write_text(json.dumps(decision), encoding="utf-8")
    subprocess.run(["git", "add", phase, decision_path], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "stage7 declaration"], cwd=controller_repo, check=True)
    controller = CampaignController(controller_repo, runner=RecordingRunner(["{}"]),
                                    expected_study_sha=DIGEST)

    with pytest.raises(ValueError, match="completed Stage 5"):
        controller.authorize_test(phase, decision_path, "b" * 64)
    stage5_summary = Path("experiments/contextual-risk-v2/summaries/stage5-msvd.json")
    stage5_run = "msvd-multiseed-frozen-seed000-20260815T124000Z-123456789abc"
    stage5_manifest = "e" * 64
    (controller_repo / stage5_summary).write_text(json.dumps({
        "phase": "stage5-msvd", "study": "contextual-risk-v2",
        "phase_path": "experiments/contextual-risk-v2/phases/stage5-msvd.json",
        "terminal_counts": {"completed": 1},
        "runs": [{"run_id": stage5_run, "state": "completed",
                  "manifest_sha256": stage5_manifest,
                  "output_hashes": {}, "summary_metrics": {}}],
    }), encoding="utf-8")
    registry = controller_repo / "experiments/contextual-risk-v2/registry.jsonl"
    registry.write_text("\n".join(json.dumps(item) for item in [
        {"run_id": stage5_run, "event": "reserved", "manifest_sha256": stage5_manifest},
        {"run_id": stage5_run, "event": "completed", "manifest_sha256": stage5_manifest},
    ]) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", stage5_summary, registry], cwd=controller_repo, check=True)
    subprocess.run(["git", "commit", "-qm", "complete stage5"], cwd=controller_repo, check=True)
    with pytest.raises(ValueError, match="frozen configuration"):
        controller.authorize_test(phase, decision_path, "c" * 64)

    result = controller.authorize_test(phase, decision_path, DIGEST)
    assert result == {}


def test_cli_entrypoint_resolves_repository_imports_without_pythonpath():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["/home/xujie/.conda/envs/tvr/bin/python", "scripts/run_reliability_campaign.py", "--help"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    for command in ("prepare", "launch", "monitor", "evaluate-gate", "finalize", "authorize-test"):
        assert command in result.stdout


def test_committed_phase_templates_are_complete_and_preserve_canonical_duplicate_semantics():
    root = Path(__file__).resolve().parents[1]
    phase_dir = root / "experiments/contextual-risk-v2/phases"
    paths = [phase_dir / f"stage{stage}-{name}.json" for stage, name in [
        (1, "baselines"), (2, "features"), (3, "loss-topk"),
        (4, "multiseed"), (5, "msvd"), (6, "analysis"), (7, "final"),
    ]]
    phases = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    required = {
        "dataset", "stage", "variant", "seed", "scheduled_timestamp_utc",
        "environment", "input_checkpoints", "dataset_splits", "config",
        "parent_run_ids", "selection_lineage", "validation_authorized",
        "test_authorized", "expected_scalar_outputs", "array_references", "command",
    }
    for phase in phases:
        assert phase["study"] == "contextual-risk-v2"
        assert phase["runs"]
        for run in phase["runs"]:
            assert set(run) == required
            assert run["environment"]["precision"] == "fp32"
            assert run["environment"]["gpu_model"] == "NVIDIA A800 80GB PCIe"
            assert run["selection_lineage"][0]["continuation_rule"]
            if run["stage"] != "baseline":
                assert run["parent_run_ids"]
            assert run["config"]["model"]["contextual_risk_feature_set"]
            assert run["config"]["model"]["contextual_risk_topk"] in [1, 4, 8, 16, 32]
            assert isinstance(run["config"]["loss"]["w_contextual_risk_fit"], float)
            assert isinstance(run["config"]["loss"]["w_contextual_risk_rank"], float)
            assert run["config"]["scheduler"]["epochs"] in [3, 5]

    assert [run["variant"] for run in phases[1]["runs"]] == [
        "score_only", "semantic_only", "no_context", "no_score_geometry", "full"
    ]
    stage3 = phases[2]["runs"]
    full_k16 = [
        run for run in stage3
        if run["config"]["model"]["contextual_risk_topk"] == 16
        and run["config"]["loss"]["w_contextual_risk_fit"] == 0.25
        and run["config"]["loss"]["w_contextual_risk_rank"] == 0.25
    ]
    assert len(full_k16) == 1
    assert {run["seed"] for run in phases[3]["runs"]} == {0, 1, 2}
