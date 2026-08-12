import os
import subprocess
import time
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_contextual_risk_bg.sh"


def _fake_controller(tmp_path, *, wait_for_output_release=False):
    marker = tmp_path / "controller-called"
    release = tmp_path / "release-controller"
    controller = tmp_path / "fake-controller.sh"
    wait = (
        """
while [[ ! -e "${A3_TEST_CONTROLLER_RELEASE}" ]]; do sleep 0.01; done
mkdir -p "${OUTPUT_DIR}"
sleep 0.2
"""
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
        f"{wait}"
    )
    controller.chmod(0o755)
    return controller, marker, release


def _launcher_env(controller, marker, *, output_dir=None):
    env = os.environ.copy()
    env.update({
        "A3_LAUNCHER_TEST_MODE": "1",
        "A3_BG_CONTROLLER": str(controller),
        "A3_TEST_CONTROLLER_MARKER": str(marker),
        "A3_TEST_HANDOFF_MARKER": str(marker.with_name("worker-pid-written")),
    })
    if output_dir is not None:
        env["A3_TEST_OUTPUT_DIR"] = str(output_dir)
    return env


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
        "--run_final_test",
    ):
        assert flag in text

    assert 'CLIP_LAYER_NORM_PRECISION=fp32' in text
    assert 'PYTHONHASHSEED=0' in text
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
    ],
)
def test_registered_a3_launcher_rejects_inherited_controller_execution_overrides(
    tmp_path, name, value
):
    controller, marker, _ = _fake_controller(tmp_path)
    env = _launcher_env(controller, marker)
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
