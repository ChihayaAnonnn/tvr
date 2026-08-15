import os
import json
import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_contextual_risk_bg.sh"


def _fake_controller(tmp_path, *, wait_for_output_release=False, exit_code=0):
    marker = tmp_path / "controller-called"
    release = tmp_path / "release-controller"
    controller = tmp_path / "fake-controller.sh"
    wait = (
        """
while [[ ! -e "${A3_TEST_CONTROLLER_RELEASE}" ]]; do sleep 0.01; done
mkdir -p "${OUTPUT_DIR}"
if [[ -n "${LOG_FILE}" ]]; then printf 'controller-final-line\n' >> "${LOG_FILE}"; fi
sleep 0.2
""" + (f"exit {exit_code}\n" if exit_code else "")
        if wait_for_output_release
        else ""
    )
    controller.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "exec >/dev/null 2>&1\n"
        "printf '%s\\n' \"$$\" >> \"${A3_TEST_CONTROLLER_MARKER}\"\n"
        "printf '%s\\n' \"$$\" > \"${TRAIN_PID_FILE}\"\n"
        ": > \"${A3_TEST_HANDOFF_MARKER}\"\n"
        "if [[ -n \"${A3_TEST_CONTROLLER_ENV_MARKER:-}\" ]]; then\n"
        "  printf '%s\\n' \"${NCCL_ASYNC_ERROR_HANDLING-<absent>}\" "
        "\"${TORCH_NCCL_ASYNC_ERROR_HANDLING-<absent>}\" "
        "> \"${A3_TEST_CONTROLLER_ENV_MARKER}\"\n"
        "fi\n"
        "if [[ -n \"${A3_TEST_CAPTURE_FILE:-}\" ]]; then\n"
        "  printf '%s\\n' \"${OUTPUT_DIR}\" \"${LOG_FILE}\" > \"${A3_TEST_CAPTURE_FILE}\"\n"
        "  printf '%s\\n' \"$@\" >> \"${A3_TEST_CAPTURE_FILE}\"\n"
        "fi\n"
        f"{wait}"
    )
    controller.chmod(0o755)
    return controller, marker, release


def _launcher_env(controller, marker, *, output_dir=None):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("NCCL_", "TORCH_NCCL_", "CUDA_", "CUBLAS_")) or name in {
            "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        }:
            env.pop(name)
    env.update({
        "A3_LAUNCHER_TEST_MODE": "1",
        "LEGACY_A3_LAYOUT": "1",
        "A3_BG_CONTROLLER": str(controller),
        "A3_TEST_CONTROLLER_MARKER": str(marker),
        "A3_TEST_HANDOFF_MARKER": str(marker.with_name("worker-pid-written")),
    })
    if output_dir is not None:
        env["A3_TEST_OUTPUT_DIR"] = str(output_dir)
    return env


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reserve_committed_registered_a3_run(
    tmp_path, *, config_mutation=None, timestamp="20260815T120000Z"
):
    import main_task_retrieval as retrieval

    repo = tmp_path / "control-repo"
    registry = Path("experiments/contextual-risk-v2/registry.jsonl")
    phase = Path("experiments/contextual-risk-v2/phases/e2e-feature.json")
    (repo / registry.parent).mkdir(parents=True)
    (repo / registry).write_text("", encoding="utf-8")
    (repo / phase.parent).mkdir(parents=True)
    generated = tmp_path / "generated-split"
    generated.mkdir()
    train_csv = generated / "train.csv"
    val_csv = generated / "val.csv"
    train_csv.write_text("video_id,sentence\nvideo0,train\n", encoding="utf-8")
    val_csv.write_text("video_id,sentence\nvideo1,val\n", encoding="utf-8")

    common = Path(
        subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    ).parent
    data_root = Path("/data2/hxj/data/MSRVTT")
    checkpoint = (
        common
        / "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
    )
    clip_asset = common / ".cache/ViT-B-16.pt"
    split_manifest = REPOSITORY / "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
    source_train = data_root / "csv/MSRVTT_train.9k.csv"
    test_csv = data_root / "csv/MSRVTT_JSFUSION_test.csv"
    annotation = data_root / "annotation/MSRVTT_v2.json"

    fixed = retrieval._registered_a3_expected_config()
    for name in ("world_size", "n_gpu", "pin_memory"):
        fixed.pop(name)
    fixed.update(
        {
            "contextual_risk_feature_set": "score_only",
            "contextual_risk_topk": 4,
            "w_contextual_risk_fit": 0.25,
            "w_contextual_risk_rank": 0.0,
            "seed": 1,
            "epochs": 3,
            "init_model": str(checkpoint),
            "run_final_test": False,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "features_path": str(
                data_root / "videos/compressed_videos/msrvtt_224_12fps"
            )
            + "/",
        }
    )
    reserved_names = {
        "contextual_risk_feature_set",
        "contextual_risk_topk",
        "w_contextual_risk_fit",
        "w_contextual_risk_rank",
        "seed",
        "epochs",
        "run_final_test",
    }
    config = {
        "model": {
            name: value for name, value in fixed.items()
            if name not in reserved_names
        },
        "loss": {
            "w_contextual_risk_fit": fixed["w_contextual_risk_fit"],
            "w_contextual_risk_rank": fixed["w_contextual_risk_rank"],
        },
        "optimizer": {"seed": fixed["seed"]},
        "scheduler": {"epochs": fixed["epochs"]},
        "evaluation": {"run_final_test": fixed["run_final_test"]},
    }
    config["model"].update(
        {
            "contextual_risk_feature_set": fixed[
                "contextual_risk_feature_set"
            ],
            "contextual_risk_topk": fixed["contextual_risk_topk"],
        }
    )
    if config_mutation is not None:
        config_mutation(config)

    references = [
        checkpoint,
        clip_asset,
        split_manifest,
        source_train,
        train_csv,
        val_csv,
        test_csv,
        annotation,
    ]
    digests = {
        checkpoint: (
            "a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607"
        ),
        clip_asset: (
            "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
        ),
        **{
            path: _file_sha256(path)
            for path in references
            if path not in {checkpoint, clip_asset}
        },
    }
    phase_payload = {
        "phase": "e2e-feature",
        "study": "contextual-risk-v2",
        "runs": [
            {
                "dataset": "msrvtt",
                "stage": "feature_ablation",
                "variant": "score_only_e2e",
                "seed": 1,
                "scheduled_timestamp_utc": timestamp,
                "environment": {
                    "python": platform.python_version(),
                    "pytorch": str(torch.__version__),
                    "cuda": str(torch.version.cuda or "none"),
                    "gpu_model": "none",
                    "gpu_count": 0,
                    "precision": "fp32",
                    "distributed_topology": "torchrun-4",
                },
                "input_checkpoints": [
                    {"path": str(path), "sha256": digests[path]}
                    for path in (checkpoint, clip_asset)
                ],
                "dataset_splits": [
                    {"path": str(path), "sha256": digests[path]}
                    for path in (
                        split_manifest,
                        source_train,
                        train_csv,
                        val_csv,
                        test_csv,
                        annotation,
                    )
                ],
                "config": config,
                "parent_run_ids": ["historical-a3-validation"],
                "selection_lineage": [],
                "validation_authorized": True,
                "test_authorized": False,
                "expected_scalar_outputs": [],
                "array_references": [],
                "command": [
                    "env",
                    "ARTIFACT_RUN_ID={run_id}",
                    "bash",
                    "run_train_msrvtt_contextual_risk_bg.sh",
                ],
            }
        ],
    }
    (repo / phase).write_text(json.dumps(phase_payload), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "phase"], cwd=repo, check=True)
    artifact_root = tmp_path / "artifacts"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY)
    reserved = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/manage_experiment_run.py"),
            "reserve-phase",
            "--repo-root",
            str(repo),
            "--phase",
            str(phase),
            "--registry",
            str(registry),
            "--artifact-root",
            str(artifact_root),
        ],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    run = json.loads(reserved.stdout)["runs"][0]
    subprocess.run(["git", "add", str(registry)], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "reserve phase"], cwd=repo, check=True
    )
    return repo, artifact_root, generated, run


def _production_contract_test_processes(tmp_path):
    capture = tmp_path / "runtime-capture.json"
    fake_torchrun = tmp_path / "fake-torchrun.py"
    fake_torchrun.write_text(
        "#!/home/xujie/.conda/envs/tvr/bin/python\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0,{str(REPOSITORY)!r})\n"
        "import torch\n"
        "import main_task_retrieval as retrieval\n"
        "from experiment_artifacts import make_run_paths, strict_json_load\n"
        "worker=sys.argv[1:]\n"
        "entry=next(i for i,v in enumerate(worker) if v.endswith('main_task_retrieval.py'))\n"
        "sys.argv=['main_task_retrieval.py',*worker[entry+1:]]\n"
        "os.environ.update(WORLD_SIZE='4',RANK='0',LOCAL_RANK='0')\n"
        "torch.cuda.device_count=lambda: 0\n"
        "args=retrieval.get_args()\n"
        "args.local_rank=0\n"
        "manifest=strict_json_load(open(Path(args.artifact_run_dir)/'manifest.json',encoding='utf-8'))\n"
        "repo=Path(os.environ['A3_TEST_REPO_ROOT'])\n"
        "retrieval._verify_artifact_runtime_bindings(args,manifest,repo,registered_a3_contract=True)\n"
        "paths=make_run_paths(Path(args.artifact_run_dir).parents[3],manifest['study'],manifest['run_id'])\n"
        "retrieval._start_artifact_run(retrieval._ArtifactRunContext(paths,manifest))\n"
        "retrieval._complete_artifact_run(retrieval._ArtifactRunContext(paths,manifest))\n"
        "json.dump({'feature_set':args.contextual_risk_feature_set,'topk':args.contextual_risk_topk,'fit':args.w_contextual_risk_fit,'rank':args.w_contextual_risk_rank,'seed':args.seed,'epochs':args.epochs,'argv':worker},open(os.environ['A3_TEST_RUNTIME_CAPTURE'],'w'))\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    proxy_controller = tmp_path / "proxy-controller.sh"
    proxy_controller.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"[[ \"${{TVR_TORCHRUN}}\" == {fake_torchrun} ]] || exit 86\n"
        "printf 'called\\n' > \"${A3_TEST_CONTROLLER_MARKER}\"\n"
        f"exec bash {REPOSITORY / 'run_train_msrvtt_bg.sh'} \"$@\"\n",
        encoding="utf-8",
    )
    proxy_controller.chmod(0o755)
    return proxy_controller, fake_torchrun, capture


def test_committed_manifest_flows_through_real_wrapper_parser_and_runtime_binding(
    tmp_path,
):
    repo, artifact_root, generated, run = _reserve_committed_registered_a3_run(
        tmp_path
    )
    controller, fake_torchrun, capture = _production_contract_test_processes(
        tmp_path
    )
    marker = tmp_path / "controller-called"
    env = _launcher_env(controller, marker)
    env.pop("LEGACY_A3_LAYOUT")
    env.update(
        {
            "ARTIFACT_RUN_ID": run["run_id"],
            "A3_TEST_REPO_ROOT": str(repo),
            "A3_TEST_ARTIFACT_ROOT": str(artifact_root),
            "A3_TEST_TORCHRUN": str(fake_torchrun),
            "A3_TEST_SKIP_SPLIT_BUILD": "1",
            "A3_TEST_GENERATED_SPLIT_DIR": str(generated),
            "A3_TEST_RUNTIME_CAPTURE": str(capture),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPOSITORY,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    bound = json.loads(capture.read_text(encoding="utf-8"))
    assert {
        name: bound[name]
        for name in ("feature_set", "topk", "fit", "rank", "seed", "epochs")
    } == {
        "feature_set": "score_only", "topk": 4, "fit": 0.25,
        "rank": 0.0, "seed": 1, "epochs": 3,
    }
    assert "--artifact_run_dir" in bound["argv"]
    assert marker.exists()
    run_dir = Path(run["artifact_path"])
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))[
        "state"
    ] == "completed"
    assert not any((run_dir / "checkpoints").iterdir())
    assert not any((run_dir / "metrics").iterdir())
    assert not any((run_dir / "arrays").iterdir())


@pytest.mark.parametrize("invalid", ["duplicate", "out_of_allowlist"])
def test_real_wrapper_rejects_invalid_committed_variant_before_controller(
    tmp_path, invalid
):
    def mutate(config):
        if invalid == "duplicate":
            config["model"]["seed"] = 1
        else:
            config["model"]["contextual_risk_topk"] = 2

    repo, artifact_root, _generated, run = _reserve_committed_registered_a3_run(
        tmp_path, config_mutation=mutate
    )
    controller, _fake_torchrun, _capture = _production_contract_test_processes(
        tmp_path
    )
    marker = tmp_path / "controller-called"
    env = _launcher_env(controller, marker)
    env.pop("LEGACY_A3_LAYOUT")
    env.update(
        {
            "ARTIFACT_RUN_ID": run["run_id"],
            "A3_TEST_REPO_ROOT": str(repo),
            "A3_TEST_ARTIFACT_ROOT": str(artifact_root),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=REPOSITORY, env=env, text=True,
        capture_output=True, check=False, timeout=15,
    )

    assert result.returncode == 2
    assert not marker.exists()
    expected = "duplicates effective field" if invalid == "duplicate" else "allowlist"
    assert expected in result.stderr


def test_real_wrapper_rejects_env_override_before_committed_manifest_handoff(
    tmp_path,
):
    repo, artifact_root, _generated, run = _reserve_committed_registered_a3_run(
        tmp_path
    )
    controller, _fake_torchrun, _capture = _production_contract_test_processes(
        tmp_path
    )
    marker = tmp_path / "controller-called"
    env = _launcher_env(controller, marker)
    env.pop("LEGACY_A3_LAYOUT")
    env.update(
        {
            "ARTIFACT_RUN_ID": run["run_id"],
            "A3_TEST_REPO_ROOT": str(repo),
            "A3_TEST_ARTIFACT_ROOT": str(artifact_root),
            "CONTEXTUAL_RISK_TOPK": "8",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=REPOSITORY, env=env, text=True,
        capture_output=True, check=False, timeout=15,
    )

    assert result.returncode == 2
    assert "environment CONTEXTUAL_RISK_TOPK cannot be overridden" in result.stderr
    assert not marker.exists()


def test_artifact_campaign_launcher_requires_registered_run_identity(tmp_path):
    controller, marker, _ = _fake_controller(tmp_path)
    env = _launcher_env(controller, marker)
    env.pop("LEGACY_A3_LAYOUT")

    result = subprocess.run(
        ["bash", str(SCRIPT)], text=True, capture_output=True, env=env, check=False
    )

    assert result.returncode == 2
    assert "ARTIFACT_RUN_ID" in result.stderr
    assert not marker.exists()


def test_base_controller_waits_for_entire_worker_process_group(tmp_path):
    base_script = REPOSITORY / "run_train_msrvtt_bg.sh"
    no_op_python = tmp_path / "no-op-python.sh"
    no_op_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    no_op_python.chmod(0o755)
    descendant_marker = tmp_path / "descendant-finished"
    fake_torchrun = tmp_path / "forking-torchrun.py"
    fake_torchrun.write_text(
        "#!/usr/bin/env python3\n"
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(2.5)\n"
        "    with open(os.environ['DESCENDANT_LOG'], 'a') as handle:\n"
        "        handle.write('descendant-final-line\\n')\n"
        "    open(os.environ['DESCENDANT_MARKER'], 'w').close()\n"
        "    os._exit(0)\n"
        "os._exit(0)\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    log_file = tmp_path / "train.log"
    env = os.environ.copy()
    env.update({
        "TVR_PYTHON": str(no_op_python),
        "TVR_TORCHRUN": str(fake_torchrun),
        "LOG_FILE": str(log_file),
        "OUTPUT_DIR": str(tmp_path / "checkpoints"),
        "TRAIN_PID_FILE": str(tmp_path / "worker.pid"),
        "DESCENDANT_LOG": str(log_file),
        "DESCENDANT_MARKER": str(descendant_marker),
        "WORKER_GROUP_DRAIN_TIMEOUT_SECONDS": "4",
    })

    result = subprocess.run(
        ["bash", str(base_script)], cwd=REPOSITORY, env=env,
        text=True, capture_output=True, check=False, timeout=8,
    )

    assert result.returncode == 0, result.stderr
    assert descendant_marker.exists()
    assert "descendant-final-line" in log_file.read_text(encoding="utf-8")


def test_base_controller_escalates_stuck_worker_group_and_records_evidence(tmp_path):
    base_script = REPOSITORY / "run_train_msrvtt_bg.sh"
    no_op_python = tmp_path / "no-op-python.sh"
    no_op_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    no_op_python.chmod(0o755)
    fake_torchrun = tmp_path / "stuck-torchrun.py"
    fake_torchrun.write_text(
        "#!/usr/bin/env python3\n"
        "import os, signal, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        "os._exit(0)\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    log_file = tmp_path / "train.log"
    env = os.environ.copy()
    env.update({
        "TVR_PYTHON": str(no_op_python),
        "TVR_TORCHRUN": str(fake_torchrun),
        "LOG_FILE": str(log_file),
        "OUTPUT_DIR": str(tmp_path / "checkpoints"),
        "TRAIN_PID_FILE": str(tmp_path / "worker.pid"),
        "WORKER_GROUP_DRAIN_TIMEOUT_SECONDS": "1",
        "WORKER_GROUP_TERM_TIMEOUT_SECONDS": "1",
        "WORKER_GROUP_KILL_TIMEOUT_SECONDS": "1",
    })

    result = subprocess.run(
        ["bash", str(base_script)], cwd=REPOSITORY, env=env,
        text=True, capture_output=True, check=False, timeout=8,
    )

    assert result.returncode == 124, result.stderr
    evidence = Path(f"{log_file}.process-group-drain.txt").read_text(
        encoding="utf-8"
    )
    assert "timeout:SIGTERM" in evidence
    assert "timeout:SIGKILL" in evidence
    assert "UNRESOLVED" not in evidence


def test_legacy_layout_seam_is_rejected_outside_launcher_test_mode(tmp_path):
    controller, marker, _ = _fake_controller(tmp_path)
    env = os.environ.copy()
    env["LEGACY_A3_LAYOUT"] = "1"
    env["A3_BG_CONTROLLER"] = str(controller)

    result = subprocess.run(
        ["bash", str(SCRIPT)], text=True, capture_output=True, env=env, check=False
    )

    assert result.returncode == 2
    assert "LEGACY_A3_LAYOUT" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    (
        "stage,test_authorized,expects_final_test,controller_exit,terminal_state,"
        "creates_output,expected_return,feature_set,topk,fit_weight,rank_weight,"
        "seed,epochs,explicit_none"
    ),
    [
        ("baseline", False, False, 0, "completed", True, 0, "score_only", 1, 0.0, 0.0, 0, 3, False),
        ("final", True, True, 0, "completed", True, 0, "full", 16, 0.25, 0.25, 1, 5, True),
        ("baseline", False, False, 7, "cancelled", True, 7, "semantic_only", 4, 0.25, 0.0, 1, 5, False),
        ("baseline", False, False, 7, "cancelled", False, 2, "no_context", 8, 0.0, 0.25, 2, 3, True),
        ("baseline", False, False, 0, "completed", True, 0, "no_score_geometry", 32, 0.25, 0.25, 0, 3, True),
    ],
)
def test_campaign_launcher_routes_every_output_to_artifact_run(
    tmp_path, stage, test_authorized, expects_final_test, controller_exit,
    terminal_state, creates_output, expected_return, feature_set, topk,
    fit_weight, rank_weight, seed, epochs, explicit_none,
):
    run_id = f"msrvtt-{stage}-smoke-seed{seed:03d}-20260814T020304Z-0123456789ab"
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "contextual-risk-v2" / "msrvtt" / stage / run_id
    for relative in ("logs", "checkpoints", "metrics", "arrays", "plots", "provenance"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "study": "contextual-risk-v2", "dataset": "msrvtt",
        "stage": stage, "variant": "smoke", "seed": seed,
        "artifact_path": str(run_dir),
        "registry_path": "experiments/contextual-risk-v2/registry.jsonl",
        "git_sha": "0123456789abcdef0123456789abcdef01234567",
        "tracked_clean": True, "source_patch_sha256": None,
        "environment": {
            "python": "3.10", "pytorch": "2.0", "cuda": "12.0",
            "gpu_model": "test", "gpu_count": 4, "precision": "fp32",
            "distributed_topology": "torchrun-4",
        },
        "input_checkpoints": [], "dataset_splits": [],
        "config": {
            "model": {
                "contextual_risk_feature_set": feature_set,
                "contextual_risk_topk": topk,
            },
            "loss": {
                "w_contextual_risk_fit": fit_weight,
                "w_contextual_risk_rank": rank_weight,
            },
            "optimizer": {"seed": seed},
            "scheduler": {"epochs": epochs},
            "evaluation": (
                {"archive_qualification_scenario": "none"}
                if explicit_none else {}
            ),
        },
        "parent_run_ids": [] if stage == "baseline" else ["parent-run"],
        "selection_lineage": [],
        "validation_authorized": True, "test_authorized": test_authorized,
        "expected_scalar_outputs": [], "array_references": [],
    }
    command = (
        f"env ARTIFACT_RUN_ID={run_id} bash "
        "run_train_msrvtt_contextual_risk_bg.sh\n"
    )
    (run_dir / "command.sh").write_text(command, encoding="utf-8")
    manifest["command_sha256"] = hashlib.sha256(command.encode()).hexdigest()
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manage = tmp_path / "fake-manage.py"
    manage.write_text(
        "import hashlib, json, os, sys\n"
        "p=json.load(open(sys.argv[sys.argv.index('--run-dir')+1]+'/manifest.json'))\n"
        "if sys.argv[1]=='transition-run':\n"
        " log=p['artifact_path']+'/logs/train.log'\n"
        " digest=hashlib.sha256(open(log,'rb').read() if os.path.exists(log) else b'').hexdigest()\n"
        " open(os.environ['A3_TEST_TERMINAL_CAPTURE'],'a').write(sys.argv[sys.argv.index('--next')+1]+' '+digest+'\\n')\n"
        "raw=json.dumps(p,sort_keys=True,separators=(',',':')).encode()\n"
        "c=p['config']\n"
        "variant=['--contextual_risk_feature_set',c['model']['contextual_risk_feature_set'],'--contextual_risk_topk',str(c['model']['contextual_risk_topk']),'--w_contextual_risk_fit',str(c['loss']['w_contextual_risk_fit']),'--w_contextual_risk_rank',str(c['loss']['w_contextual_risk_rank']),'--seed',str(c['optimizer']['seed']),'--epochs',str(c['scheduler']['epochs'])]\n"
        "print(json.dumps({'run_dir':p['artifact_path'],'manifest_sha256':hashlib.sha256(raw).hexdigest(),'state':'reserved','test_authorized':p['test_authorized'],'stage':p['stage'],'command_argv':['env','ARTIFACT_RUN_ID='+p['run_id'],'bash','run_train_msrvtt_contextual_risk_bg.sh'],'registered_a3_variant_argv':variant}))\n",
        encoding="utf-8",
    )
    controller, marker, release = _fake_controller(
        tmp_path, wait_for_output_release=creates_output, exit_code=controller_exit
    )
    capture = tmp_path / "capture.txt"
    terminal_capture = tmp_path / "terminal-capture.txt"
    env = _launcher_env(controller, marker)
    env.pop("LEGACY_A3_LAYOUT")
    env.update({
        "ARTIFACT_RUN_ID": run_id,
        "A3_TEST_ARTIFACT_ROOT": str(artifact_root),
        "A3_TEST_REPO_ROOT": str(tmp_path),
        "A3_TEST_MANAGE_CLI": str(manage),
        "A3_TEST_CAPTURE_FILE": str(capture),
        "A3_TEST_CONTROLLER_RELEASE": str(release),
        "A3_TEST_TERMINAL_CAPTURE": str(terminal_capture),
    })
    release.touch()

    result = subprocess.run(
        ["bash", str(SCRIPT)], text=True, capture_output=True, env=env,
        check=False, timeout=5,
    )

    assert result.returncode == expected_return, result.stderr
    captured = capture.read_text().splitlines()
    assert Path(captured[0]) == run_dir / "checkpoints"
    assert Path(captured[1]) == run_dir / "logs" / "train.log"
    assert "--artifact_run_dir" in captured[2:]
    assert str(run_dir) in captured[2:]
    assert ("--run_final_test" in captured[2:]) is expects_final_test
    for flag, value in (
        ("--contextual_risk_feature_set", feature_set),
        ("--contextual_risk_topk", topk),
        ("--w_contextual_risk_fit", fit_weight),
        ("--w_contextual_risk_rank", rank_weight),
        ("--seed", seed),
        ("--epochs", epochs),
    ):
        position = captured.index(flag, 2)
        assert captured[position + 1] == str(value)
    assert not Path(captured[0]).is_relative_to(REPOSITORY / "ckpts")
    assert not Path(captured[1]).is_relative_to(REPOSITORY / "logs")
    captured_terminal_state, sealed_log_digest = terminal_capture.read_text().split()
    assert captured_terminal_state == terminal_state
    log_bytes = (
        (run_dir / "logs" / "train.log").read_bytes()
        if (run_dir / "logs" / "train.log").exists() else b""
    )
    assert sealed_log_digest == hashlib.sha256(log_bytes).hexdigest()


def test_registered_a3_launcher_preserves_the_fresh_training_contract():
    assert SCRIPT.is_file(), "registered A3 launcher is missing"
    text = SCRIPT.read_text()

    assert (
        "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
        in text
    )
    assert "--git-common-dir" in text
    for flag in (
        "--use_uatvr_full",
        "--use_ada_norm",
        "--use_retrieval_risk_sidecar",
        "--use_contextual_risk_sidecar",
        "--contextual_risk_sidecar_only",
        "--registered_contextual_risk_a3",
        "--contextual_risk_projection_dim 128",
        "--contextual_risk_hidden_dim 128",
        "--contextual_risk_temperature 1.0",
        "REGISTERED_VARIANT_ARGS",
    ):
        assert flag in text

    assert 'CLIP_LAYER_NORM_PRECISION=fp32' in text
    assert 'PYTHONHASHSEED=0' in text
    assert 'CLIP_CACHE_DIR="${COMMON_REPO_ROOT}/.cache"' in text
    assert 'ViT-B-16.pt' in text
    assert '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f' in text
    for binding in (
        'CUBLAS_WORKSPACE_CONFIG=":4096:8"',
        'CUDA_MODULE_LOADING=LAZY',
        'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1', 'NUMEXPR_NUM_THREADS=1',
        'NCCL_P2P_LEVEL=NVL', 'NCCL_IB_DISABLE=1', 'NCCL_DEBUG=WARN',
        'NCCL_ASYNC_ERROR_HANDLING=1',
        'MASTER_PORT=29547', 'TORCH_WARN_ONCE=1',
        'TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python',
        'TVR_TORCHRUN="${A3_TEST_TORCHRUN:-/home/xujie/.conda/envs/tvr/bin/torchrun}"',
    ):
        assert binding in text
    assert '--registered_clip_asset_path "${CLIP_ASSET}"' in text
    assert '--registered_clip_asset_sha256 "${CLIP_ASSET_SHA256}"' in text
    assert 'EXPERIMENT_PROFILE=hygiene' in text
    assert 'TRAIN_BATCH_SIZE=256' in text
    assert 'TRAIN_GRADIENT_ACCUMULATION_STEPS=1' in text
    assert 'CUDA_VISIBLE_DEVICES="0,2,3,4"' in text
    assert 'NPROC=4' in text
    assert 'RUN_DATE="$(date +%Y%m%d)"' in text
    assert 'RUN_TIME="$(date +%H%M%S)"' in text
    assert 'git -C "${ROOT_DIR}" diff --quiet --' in text
    assert 'git -C "${ROOT_DIR}" diff --cached --quiet --' in text


@pytest.mark.parametrize(
    "feature_set",
    ["full", "score_only", "semantic_only", "no_context", "no_score_geometry"],
)
def test_retrieval_parser_accepts_only_registered_feature_sets(
    monkeypatch, tmp_path, feature_set
):
    import main_task_retrieval

    monkeypatch.setattr(
        sys, "argv", [
            "prog", "--output_dir", str(tmp_path), "--do_train",
            "--datatype", "fixture",
            "--contextual_risk_feature_set", feature_set,
        ],
    )
    assert main_task_retrieval.get_args().contextual_risk_feature_set == feature_set


def test_registered_launcher_qualification_is_manifest_bound_and_uses_controller():
    text = SCRIPT.read_text()
    base = (REPOSITORY / "run_train_msrvtt_bg.sh").read_text()

    assert "ARCHIVE_QUALIFICATION_SCENARIO" in text
    assert "archive_qualification_scenario" in text
    assert 'bash "${CONTROLLER_SCRIPT}"' in text
    assert "ARCHIVE_QUALIFICATION_SCENARIO" in base
    assert '"${TVR_TORCHRUN}"' in base
    assert '"${ROOT_DIR}/main_task_retrieval.py"' in base
    assert "scripts/build_msrvtt_trusted_split.py" in base
    assert base.index("ARCHIVE_QUALIFICATION_SCENARIO") < base.index(
        "scripts/build_msrvtt_trusted_split.py"
    )


def test_registered_a3_launcher_pins_legacy_async_value_at_controller_boundary(
    tmp_path,
):
    controller, marker, release = _fake_controller(
        tmp_path, wait_for_output_release=True
    )
    output_dir = tmp_path / "pinned-controller-output"
    env_marker = tmp_path / "controller-environment"
    env = _launcher_env(controller, marker, output_dir=output_dir)
    env["A3_TEST_CONTROLLER_RELEASE"] = str(release)
    env["A3_TEST_CONTROLLER_ENV_MARKER"] = str(env_marker)
    release.touch()

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert env_marker.read_text().splitlines() == ["1", "<absent>"]


@pytest.mark.parametrize(
    "override", ["--seed", "--contextual_risk_topk=8", "--n_display=1", "note"]
)
def test_registered_a3_launcher_rejects_all_trailing_arguments_before_controller(
    tmp_path, override
):
    controller, marker, _ = _fake_controller(tmp_path)
    command = ["bash", str(SCRIPT), override]
    if override == "--seed":
        command.append("9")

    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=_launcher_env(controller, marker),
        check=False,
    )

    assert result.returncode == 2
    assert "does not accept trailing arguments" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "name",
    [
        "CONTEXTUAL_RISK_FEATURE_SET",
        "CONTEXTUAL_RISK_TOPK",
        "W_CONTEXTUAL_RISK_FIT",
        "W_CONTEXTUAL_RISK_RANK",
        "SEED",
        "EPOCHS",
    ],
)
def test_registered_a3_launcher_rejects_scientific_environment_override(
    tmp_path, name
):
    controller, marker, _ = _fake_controller(tmp_path)
    env = _launcher_env(controller, marker)
    env[name] = "arbitrary"

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert f"environment {name} cannot be overridden" in result.stderr
    assert not marker.exists()


def test_registered_a3_launcher_rejects_existing_output_before_controller(tmp_path):
    controller, marker, _ = _fake_controller(tmp_path)
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=_launcher_env(controller, marker, output_dir=output_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "A3 output directory already exists" in result.stderr
    assert not marker.exists()


def test_registered_a3_launcher_rejects_preexisting_reservation_before_controller(
    tmp_path,
):
    controller, marker, _ = _fake_controller(tmp_path)
    output_dir = tmp_path / "reserved-output"
    reservation = Path(f"{output_dir}.reserve")
    reservation.mkdir()

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=_launcher_env(controller, marker, output_dir=output_dir),
        check=False,
    )

    assert result.returncode == 2
    assert "A3 output reservation already exists" in result.stderr
    assert not marker.exists()


def test_registered_a3_launcher_reservation_excludes_concurrent_handoff_until_output_exists(
    tmp_path,
):
    controller, marker, release = _fake_controller(
        tmp_path, wait_for_output_release=True
    )
    output_dir = tmp_path / "concurrent-output"
    env = _launcher_env(controller, marker, output_dir=output_dir)
    handoff = Path(env["A3_TEST_HANDOFF_MARKER"])
    env["A3_TEST_CONTROLLER_RELEASE"] = str(release)
    first = subprocess.Popen(
        ["bash", str(SCRIPT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        deadline = time.monotonic() + 5
        while not handoff.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert handoff.exists()
        assert Path(f"{output_dir}.reserve").is_dir()

        second = subprocess.run(
            ["bash", str(SCRIPT)],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert second.returncode == 2
        assert "A3 output reservation already exists" in second.stderr
    finally:
        release.touch()
        _, first_stderr = first.communicate(timeout=5)

    assert first.returncode == 0, first_stderr
    assert marker.read_text().count("\n") == 1
    assert output_dir.is_dir()
    assert not Path(f"{output_dir}.reserve").exists()


def test_registered_a3_launcher_retains_reservation_when_worker_exits_before_output(
    tmp_path,
):
    controller, marker, _ = _fake_controller(tmp_path)
    output_dir = tmp_path / "worker-exited-output"
    env = _launcher_env(controller, marker, output_dir=output_dir)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=5,
    )

    assert result.returncode == 2
    assert "worker handoff is not a live PID" in result.stderr
    assert marker.exists()
    assert Path(f"{output_dir}.reserve").is_dir()


@pytest.mark.parametrize(
    "name,value",
    [
        ("RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER", "1"),
        ("TVR_PYTHON", "/bin/false"),
        ("TVR_TORCHRUN", "/bin/false"),
        ("DATA_PATH", "/tmp/not-msrvtt"),
        ("TRAIN_NUM_WORKERS", "1"),
        ("TRAIN_PREFETCH_FACTOR", "1"),
        ("MASTER_PORT", "29999"),
        ("SMOKE_STEPS", "1"),
        ("PYTHONHASHSEED", "9"),
        ("REGISTERED_CLIP_ASSET_SHA256", "0" * 64),
        ("NCCL_ASYNC_ERROR_HANDLING", "1"),
        ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
        ("NCCL_ALGO", "Ring"),
        ("NCCL_PROTO", "Simple"),
        ("NCCL_P2P_DISABLE", "1"),
        ("NVIDIA_TF32_OVERRIDE", "1"),
        ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1"),
        ("CUBLAS_LOGINFO_DBG", "1"),
        ("CUDA_LAUNCH_BLOCKING", "1"),
    ],
)
def test_registered_a3_launcher_rejects_inherited_controller_execution_overrides(
    tmp_path, name, value
):
    controller, marker, _ = _fake_controller(tmp_path)
    output_dir = tmp_path / "override-output"
    env = _launcher_env(controller, marker, output_dir=output_dir)
    env[name] = value

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert f"environment {name}" in result.stderr
    assert not marker.exists()
    assert not Path(f"{output_dir}.reserve").exists()


@pytest.mark.parametrize(
    ("seam", "value"),
    [
        ("A3_BG_CONTROLLER", "controller"),
        ("A3_TEST_OUTPUT_DIR", "output"),
        ("A3_TEST_TORCHRUN", "torchrun"),
        ("A3_TEST_SKIP_SPLIT_BUILD", "1"),
        ("A3_TEST_GENERATED_SPLIT_DIR", "generated"),
    ],
)
def test_registered_a3_launcher_rejects_test_seams_without_test_mode(
    tmp_path, seam, value
):
    controller, marker, _ = _fake_controller(tmp_path)
    env = os.environ.copy()
    path_values = {
        "controller": controller,
        "output": tmp_path / "test-output",
        "torchrun": tmp_path / "fake-torchrun",
        "generated": tmp_path / "generated",
    }
    env[seam] = str(path_values.get(value, value))

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 2
    assert "A3_LAUNCHER_TEST_MODE=1" in result.stderr
    assert not marker.exists()


def test_registered_a3_launcher_omits_other_sidecar_only_modes():
    assert SCRIPT.is_file(), "registered A3 launcher is missing"
    text = SCRIPT.read_text()

    assert "\n    --fp16" not in text
    for forbidden in (
        "--retrieval_risk_sidecar_only",
        "--csd_logvar_sidecar_only",
        "--intrinsic_concentration_only",
    ):
        assert f"\n    {forbidden}" not in text


def test_registered_a3_launcher_is_executable_and_parses():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_registered_a3_launcher_retains_reservation_on_ownership_timeout(tmp_path):
    controller, marker, release = _fake_controller(
        tmp_path, wait_for_output_release=True
    )
    output_dir = tmp_path / "timeout-output"
    env = _launcher_env(controller, marker, output_dir=output_dir)
    env["A3_TEST_CONTROLLER_RELEASE"] = str(release)
    env["A3_TEST_OWNERSHIP_TIMEOUT_SECONDS"] = "1"

    try:
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            text=True,
            capture_output=True,
            env=env,
            check=False,
            timeout=5,
        )
    finally:
        release.touch()

    assert result.returncode == 2
    assert "ownership timed out" in result.stderr
    assert Path(f"{output_dir}.reserve").is_dir()


def test_registered_a3_launcher_signal_retains_reservation_after_worker_handoff(
    tmp_path,
):
    controller, marker, release = _fake_controller(
        tmp_path, wait_for_output_release=True
    )
    output_dir = tmp_path / "signal-output"
    env = _launcher_env(controller, marker, output_dir=output_dir)
    env["A3_TEST_CONTROLLER_RELEASE"] = str(release)
    handoff = Path(env["A3_TEST_HANDOFF_MARKER"])
    process = subprocess.Popen(
        ["bash", str(SCRIPT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        deadline = time.monotonic() + 5
        while not handoff.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert handoff.exists()
        process.terminate()
        _, stderr = process.communicate(timeout=5)
    finally:
        release.touch()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode != 0
    assert "retaining reservation" in stderr
    assert Path(f"{output_dir}.reserve").is_dir()
