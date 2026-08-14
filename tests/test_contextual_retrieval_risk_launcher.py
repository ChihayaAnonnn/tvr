import os
import json
import hashlib
import subprocess
import time
from pathlib import Path

import pytest


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
    "stage,test_authorized,expects_final_test,controller_exit,terminal_state,creates_output,expected_return",
    [
        ("baseline", False, False, 0, "completed", True, 0),
        ("final", True, True, 0, "completed", True, 0),
        ("baseline", False, False, 7, "cancelled", True, 7),
        ("baseline", False, False, 7, "cancelled", False, 2),
    ],
)
def test_campaign_launcher_routes_every_output_to_artifact_run(
    tmp_path, stage, test_authorized, expects_final_test, controller_exit,
    terminal_state, creates_output, expected_return,
):
    run_id = f"msrvtt-{stage}-smoke-seed000-20260814T020304Z-0123456789ab"
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "contextual-risk-v2" / "msrvtt" / stage / run_id
    for relative in ("logs", "checkpoints", "metrics", "arrays", "plots", "provenance"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "study": "contextual-risk-v2", "dataset": "msrvtt",
        "stage": stage, "variant": "smoke", "seed": 0,
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
        "config": {name: {} for name in ("model", "loss", "optimizer", "scheduler", "evaluation")},
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
        "print(json.dumps({'run_dir':p['artifact_path'],'manifest_sha256':hashlib.sha256(raw).hexdigest(),'state':'reserved','test_authorized':p['test_authorized'],'stage':p['stage'],'command_argv':['env','ARTIFACT_RUN_ID='+p['run_id'],'bash','run_train_msrvtt_contextual_risk_bg.sh']}))\n",
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
        "--contextual_risk_topk 16",
        "--contextual_risk_projection_dim 128",
        "--contextual_risk_hidden_dim 128",
        "--contextual_risk_temperature 1.0",
        "--w_contextual_risk_fit 0.25",
        "--w_contextual_risk_rank 0.25",
        "--seed 0",
        "--epochs 3",
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
        'TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun',
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


@pytest.mark.parametrize("seam", ["A3_BG_CONTROLLER", "A3_TEST_OUTPUT_DIR"])
def test_registered_a3_launcher_rejects_test_seams_without_test_mode(tmp_path, seam):
    controller, marker, _ = _fake_controller(tmp_path)
    env = os.environ.copy()
    env["A3_BG_CONTROLLER"] = str(controller)
    if seam == "A3_TEST_OUTPUT_DIR":
        env["A3_TEST_OUTPUT_DIR"] = str(tmp_path / "test-output")

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
