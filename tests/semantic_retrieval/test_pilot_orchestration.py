"""Run real CPU workers through whole-batch reservation and finalization."""

import json
import sys
from pathlib import Path

import pytest

from experiment_artifacts import artifact_json_bytes, parse_registry_text
from test_pilot_protocol import CONTROL, NAME, pilot_repo, template
from test_runtime_artifacts import git


def cpu_phase(repo, *, exit_code=0):
    from semantic_retrieval.runtime.pilot_protocol import build_phase
    spec = template("pilot-q0-prep")
    spec["dataset"] = "synthetic"
    root = Path(__file__).resolve().parents[2]
    program = (
        "import sys,json; from pathlib import Path; from semantic_retrieval.config import RuntimeConfig; "
        "from semantic_retrieval.runtime.artifacts import prepare_run,start_run,publish_metrics,complete_run; "
        "from semantic_retrieval.runtime.distributed import configure_fp32; "
        "configure_fp32(); repo=Path(sys.argv[1]); run=Path(sys.argv[2]); "
        "m=json.loads((run/'manifest.json').read_text()); "
        "from experiment_artifacts import manifest_sha256; "
        "r=RuntimeConfig(repo/'experiments/semantic-answerability-pilot-v1/study.yaml',"
        "repo/m['registry_path'],run,m['run_id'],manifest_sha256(m),m['git_sha'],'val'); "
        "c=prepare_run(r,repo_root=repo); start_run(c); publish_metrics(c,{'cpu_fixture':True}); "
        + ("complete_run(c)" if not exit_code else f"raise SystemExit({exit_code})")
    )
    spec["command"] = ["env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1",
                       f"PYTHONPATH={root / 'src'}:{root}", sys.executable, "-c", program, str(repo), "{artifact_run_dir}"]
    path = f"{CONTROL}/phases/Q0P.json"
    phase = build_phase("Q0P", phase_path=path, scheduled_timestamp_utc="20261001T120000Z", run_templates=[spec], evidence={})
    (repo / path).parent.mkdir(exist_ok=True)
    (repo / path).write_bytes(artifact_json_bytes(phase))
    git(repo, "add", path)
    git(repo, "commit", "-qm", "CPU phase")
    return path


def test_whole_batch_commit_worker_terminal_and_tracked_summary(pilot_repo):
    from scripts.orchestrate_semantic_pilot import reserve_and_commit, execute_phase
    path = cpu_phase(pilot_repo)
    result = reserve_and_commit(pilot_repo, path, pilot_repo / "artifacts")
    run = Path(result["runs"][0]["artifact_path"])
    assert json.loads((run / "status.json").read_text())["state"] == "reserved"
    assert len(parse_registry_text(git(pilot_repo, "show", f"HEAD:{CONTROL}/registry.jsonl"))) == 1
    summary = execute_phase(pilot_repo, path, pilot_repo / "artifacts")
    assert summary["terminal_counts"] == {"completed": 1}
    assert (run / "logs/worker.log").is_file()
    assert json.loads(git(pilot_repo, "show", f"HEAD:{CONTROL}/summaries/{summary['phase']}.json"))["runs"][0]["state"] == "completed"
    with pytest.raises(ValueError, match="reserved|retry|terminal"):
        execute_phase(pilot_repo, path, pilot_repo / "artifacts")


def test_worker_failure_is_archived_once_and_requires_manual_successor(pilot_repo):
    from scripts.orchestrate_semantic_pilot import reserve_and_commit, execute_phase
    path = cpu_phase(pilot_repo, exit_code=7)
    reserved = reserve_and_commit(pilot_repo, path, pilot_repo / "artifacts")
    run = Path(reserved["runs"][0]["artifact_path"])
    with pytest.raises(RuntimeError, match="manual"):
        execute_phase(pilot_repo, path, pilot_repo / "artifacts")
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "failed" and status["exit_code"] == 7
    before = (run / "status.json").read_bytes(), (run / "provenance/audit.jsonl").read_bytes()
    with pytest.raises(ValueError):
        execute_phase(pilot_repo, path, pilot_repo / "artifacts")
    assert before == ((run / "status.json").read_bytes(), (run / "provenance/audit.jsonl").read_bytes())
    records = parse_registry_text(git(pilot_repo, "show", f"HEAD:{CONTROL}/registry.jsonl"))
    assert [r["state"] for r in records] == ["reserved", "failed"]


def test_run_config_provisioning_injects_exact_reservation_and_rejects_changed_template(pilot_repo, model_config):
    from dataclasses import replace
    from scripts.orchestrate_semantic_pilot import provision_run_config, reserve_and_commit
    from semantic_retrieval.config import PolicyConfig, RuntimeConfig, SystemConfig, TrainConfig, config_payload, effective_manifest_config
    rt = RuntimeConfig(pilot_repo / CONTROL / "study.yaml", pilot_repo / CONTROL / "registry.jsonl",
        pilot_repo / "placeholder", "fixture", "a" * 64, "b" * 40, "train")
    cfg = SystemConfig(rt, model_config, TrainConfig(1e-4, 0, 0, "constant", 1, ("matcher.",),
        dict(coarse=0.,match=0.,pair=1.,exist=0.,align=0.,group=0.)), PolicyConfig(1.,1.,(1,),1))
    path = cpu_phase(pilot_repo)
    result = reserve_and_commit(pilot_repo, path, pilot_repo / "artifacts")
    run = Path(result["runs"][0]["artifact_path"])
    # A mismatched effective template cannot be smuggled into run-local config.
    with pytest.raises(ValueError, match="configuration|config"):
        provision_run_config(pilot_repo, run, config_payload(cfg))
    assert not (run / "provenance/system-config.json").exists()
