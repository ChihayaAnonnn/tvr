"""End-to-end qualification fixture for the run-centric artifact archive."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import numpy as np
import torch
import torch.distributed as dist

from experiment_artifacts import (
    ArtifactArrayStore,
    artifact_json_bytes,
    validate_artifact_json_payload,
    verify_array_reference,
)


MANAGER = REPOSITORY / "scripts" / "manage_experiment_run.py"
REGISTRY = Path("experiments/contextual-risk-v2/registry.jsonl")
PHASE = Path("experiments/contextual-risk-v2/phases/stage0-e2e-fixture.json")
SUMMARY = Path("experiments/contextual-risk-v2/summaries/stage0-e2e-fixture.json")
ONE_MIB = 1024 * 1024


def _run_manager(repo: Path, *arguments: object) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY)])
    completed = subprocess.run(
        [sys.executable, str(MANAGER), *map(str, arguments)],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _commit(repo: Path, message: str, *paths: Path) -> None:
    subprocess.run(["git", "add", *map(str, paths)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=repo, check=True)


def _tracked_status(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fixture_worker_source() -> str:
    return """\
import argparse
import json
from pathlib import Path
from semantic_retrieval.config import RuntimeConfig
from semantic_retrieval.runtime.artifacts import prepare_run, start_run, complete_run
from semantic_retrieval.runtime.qualification import run_cpu_qualification

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
args = parser.parse_args()
root = Path(args.run_dir)
repo = Path.cwd()
manifest = json.loads((root/"manifest.json").read_text())
from experiment_artifacts import manifest_sha256
config = RuntimeConfig(repo/"experiments/contextual-risk-v2/study.yaml",repo/manifest["registry_path"],root,manifest["run_id"],manifest_sha256(manifest),manifest["git_sha"],"val")
context = prepare_run(config,repo_root=repo)
start_run(context)
run_cpu_qualification(context,scenario="success")
(context.paths.metrics/"val_eval_000.json").rename(context.paths.metrics/"selected_validation.json")
complete_run(context)
"""


def test_four_rank_worker_entry_resolves_repository_modules_without_pythonpath():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--help"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--four-rank-worker" in completed.stdout


def test_archive_phase_round_trip_externalizes_arrays_without_legacy_outputs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / REGISTRY.parent).mkdir(parents=True)
    (repo / PHASE.parent).mkdir(parents=True)
    (repo / REGISTRY).write_text("", encoding="utf-8")
    (repo / REGISTRY.parent / "study.yaml").write_text("study: contextual-risk-v2\n")
    worker = repo / "fixture_worker.py"
    worker.write_text(_fixture_worker_source(), encoding="utf-8")

    for legacy_root in (repo / "logs", repo / "ckpts"):
        legacy_root.mkdir()
        (legacy_root / "sentinel.txt").write_text("legacy-read-only\n", encoding="utf-8")
    legacy_before = {name: _tree_snapshot(repo / name) for name in ("logs", "ckpts")}

    phase = {
        "phase": "stage0-e2e-fixture",
        "study": "contextual-risk-v2",
        "runs": [
            {
                "dataset": "msrvtt",
                "stage": "baseline",
                "variant": "cpu_e2e_fixture",
                "seed": 0,
                "timestamp_utc": "20260814T000000Z",
                "environment": {
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "cuda": str(torch.version.cuda or "none"),
                    "gpu_model": "none",
                    "gpu_count": 0,
                    "precision": "fp32",
                    "distributed_topology": "single-process",
                },
                "input_checkpoints": [],
                "dataset_splits": [],
                "config": {
                    "model": {},
                    "loss": {},
                    "optimizer": {},
                    "scheduler": {},
                    "evaluation": {"eval_split": "val", "qualification_scenario": "success"},
                },
                "parent_run_ids": [],
                "selection_lineage": ["Stage 0 archive qualification only"],
                "validation_authorized": True,
                "test_authorized": False,
                "expected_scalar_outputs": ["qualification_loss"],
                "array_references": [],
                "command": [
                    sys.executable,
                    "fixture_worker.py",
                    "--run-dir",
                    "{artifact_run_dir}",
                ],
            }
        ],
    }
    (repo / PHASE).write_text(json.dumps(phase, sort_keys=True), encoding="utf-8")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    _commit(repo, "define fixture phase", Path("."))

    artifact_root = (tmp_path / "artifacts").resolve()
    reservation = _run_manager(
        repo,
        "reserve-phase",
        "--repo-root",
        repo,
        "--phase",
        PHASE,
        "--registry",
        REGISTRY,
        "--artifact-root",
        artifact_root,
    )
    run = reservation["runs"][0]
    run_dir = Path(run["artifact_path"])
    assert _tracked_status(repo) == f" M {REGISTRY.as_posix()}\n"
    _commit(repo, "reserve fixture phase", REGISTRY)
    assert _tracked_status(repo) == ""

    identity = (
        "--repo-root",
        repo,
        "--registry",
        REGISTRY,
        "--run-dir",
        run_dir,
        "--run-id",
        run["run_id"],
        "--manifest-sha256",
        run["manifest_sha256"],
    )
    verified = _run_manager(repo, "verify-run", *identity)
    assert verified["state"] == "reserved"

    command = shlex.split((run_dir / "command.sh").read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY)])
    subprocess.run(command, cwd=repo, env=environment, check=True)
    assert _tracked_status(repo) == ""

    metric_path = run_dir / "metrics" / "selected_validation.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    assert np.isfinite(metrics["summary_metrics"]["qualification_loss"])
    reference = metrics["array_references"][0]
    assert reference["byte_size"] == 2 * ONE_MIB
    restored = verify_array_reference(run_dir, reference)
    assert restored.shape == (524_288,)
    assert not any(isinstance(value, list) for value in metrics["summary_metrics"].values())

    terminal = _run_manager(
        repo,
        "transition-run",
        *identity,
        "--expected",
        "validating",
        "--next",
        "completed",
        "--exit-code",
        0,
        "--output",
        "metrics/selected_validation.json",
        "--output",
        reference["path"],
    )
    assert terminal["state"] == "completed"
    assert _tracked_status(repo) == ""

    finalized = _run_manager(
        repo,
        "finalize-phase",
        "--repo-root",
        repo,
        "--phase",
        PHASE,
        "--registry",
        REGISTRY,
        "--artifact-root",
        artifact_root,
        "--summary",
        SUMMARY,
    )
    assert finalized["terminal_counts"] == {"completed": 1}
    assert _tracked_status(repo) == f" M {REGISTRY.as_posix()}\n"
    assert (repo / SUMMARY).is_file()
    _commit(repo, "finalize fixture phase", REGISTRY, SUMMARY)
    assert _tracked_status(repo) == ""

    records = [json.loads(line) for line in (repo / REGISTRY).read_text().splitlines()]
    assert [(record["event"], record["state"]) for record in records] == [
        ("reserved", "reserved"),
        ("completed", "completed"),
    ]
    for json_path in run_dir.rglob("*.json"):
        assert json_path.stat().st_size < ONE_MIB
    assert {name: _tree_snapshot(repo / name) for name in ("logs", "ckpts")} == legacy_before


def _write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    """Publish a small marker once so rank ownership is directly auditable."""

    validate_artifact_json_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(artifact_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _run_four_rank_worker(mode: str, run_dir: Path) -> None:
    """One real CUDA step plus bounded fault modes for Stage 0 qualification."""

    if not torch.cuda.is_available():
        raise RuntimeError("four-rank qualification requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"qualification requires exactly four ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=20))
    marker = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "mode": mode,
        "cuda_device": torch.cuda.get_device_name(local_rank),
    }
    _write_exclusive_json(run_dir / "provenance" / f"rank-{rank:02d}.json", marker)

    if mode == "rank-failure":
        if rank == 2:
            raise RuntimeError("intentional nonzero-rank qualification failure")
        dist.barrier()
        return
    if mode == "collective-fault":
        if rank == 3:
            os._exit(17)
        time.sleep(0.5)
        value = torch.tensor([float(rank)], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        return
    if mode == "sigterm-drain":
        while True:
            time.sleep(0.2)
    if mode != "success":
        raise ValueError(f"unknown qualification worker mode: {mode}")

    torch.manual_seed(1000 + rank)
    model = torch.nn.Linear(4, 2).cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    features = torch.arange(8, dtype=torch.float32, device=f"cuda:{local_rank}").reshape(2, 4)
    targets = torch.zeros((2, 2), dtype=torch.float32, device=f"cuda:{local_rank}")
    loss = torch.nn.functional.mse_loss(model(features), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("four-rank fixture loss is not finite")
    optimizer.zero_grad()
    loss.backward()
    for parameter in model.parameters():
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)
    optimizer.step()
    torch.cuda.synchronize(local_rank)

    reduced_loss = loss.detach().clone()
    dist.all_reduce(reduced_loss)
    reduced_loss.div_(world_size)
    if rank == 0:
        _write_exclusive_json(
            run_dir / "provenance" / "lifecycle-owner.json",
            {"rank": rank, "pid": os.getpid(), "role": "lifecycle-owner"},
        )
        values = np.arange(524_288, dtype=np.float32)
        reference = ArtifactArrayStore(run_dir / "arrays").write("four_rank_fixture_values", values)
        metrics = {
            "summary_metrics": {"qualification_loss": float(reduced_loss.cpu())},
            "split": "validation",
            "world_size": world_size,
            "array_references": [reference.as_dict()],
        }
        validate_artifact_json_payload(metrics)
        (run_dir / "metrics" / "selected_validation.json").write_bytes(artifact_json_bytes(metrics))
    dist.barrier()
    dist.destroy_process_group()


def _worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--four-rank-worker", action="store_true", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("success", "rank-failure", "collective-fault", "sigterm-drain"),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    _run_four_rank_worker(arguments.mode, arguments.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1:]))
