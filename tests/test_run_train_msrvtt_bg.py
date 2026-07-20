import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_bg.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _read_nul_arguments(path: Path) -> list[str]:
    payload = path.read_bytes()
    return [item.decode() for item in payload.split(b"\0") if item]


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 2.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def _copy_script(tmp_path: Path) -> Path:
    copied = tmp_path / SCRIPT.name
    shutil.copy2(SCRIPT, copied)
    return copied


def _environment(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "DATA_PATH": "/dataset",
            "OUTPUT_DIR": str(tmp_path / "checkpoints"),
            "RUN_ID": "unit-run",
            "EXPERIMENT_PROFILE": "default",
            "CUDA_VISIBLE_DEVICES": "3,5",
            "NPROC": "2",
            "CLIP_GRADIENT_CHECKPOINTING": "1",
        }
    )
    return environment


def test_controller_relaunches_same_script_and_tails_log(tmp_path):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    setsid_arguments = tmp_path / "setsid.args"
    tail_arguments = tmp_path / "tail.args"
    pid_file = tmp_path / "train.pid"
    _write_executable(
        fake_bin / "setsid",
        'printf "%s\\0" "$@" > "$SETSID_ARGUMENTS"\n',
    )
    _write_executable(
        fake_bin / "tail",
        'printf "%s\\0" "$@" > "$TAIL_ARGUMENTS"\n',
    )
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {
            "RUN_DATE": "20260720",
            "RUN_TIME": "121314",
            "RUN_TAG": "rspr",
            "RUN_ID": "20260720_121314_rspr",
            "TRAIN_PID_FILE": str(pid_file),
            "SETSID_ARGUMENTS": str(setsid_arguments),
            "TAIL_ARGUMENTS": str(tail_arguments),
        }
    )

    subprocess.run(
        [
            "bash",
            str(script),
            "--rspr_mode",
            "stochastic",
            "--experiment_desc",
            "two words",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    _wait_for(setsid_arguments)
    setsid_args = _read_nul_arguments(setsid_arguments)
    assert setsid_args == [
        "env",
        "RUN_ID=20260720_121314_rspr",
        "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1",
        "bash",
        str(script),
        "--rspr_mode",
        "stochastic",
        "--experiment_desc",
        "two words",
    ]
    assert _read_nul_arguments(tail_arguments) == [
        "-n",
        "50",
        "-F",
        "logs/20260720/121314_rspr_train_msrvtt.log",
    ]
    assert pid_file.read_text().strip().isdigit()


def test_worker_runs_split_builder_and_torchrun_without_recursing(tmp_path):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_arguments = tmp_path / "python.args"
    torchrun_arguments = tmp_path / "torchrun.args"
    worker_environment = tmp_path / "worker.env"
    forbidden_call = tmp_path / "forbidden"
    _write_executable(
        fake_bin / "python3",
        'printf "%s\\0" "$@" > "$PYTHON_ARGUMENTS"\n',
    )
    _write_executable(
        fake_bin / "torchrun",
        (
            'printf "%s\\0" "$@" > "$TORCHRUN_ARGUMENTS"\n'
            'printf "%s" "${RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER-unset}" '
            '> "$WORKER_ENVIRONMENT"\n'
        ),
    )
    for command in ("setsid", "tail"):
        _write_executable(
            fake_bin / command,
            'touch "$FORBIDDEN_CALL"\nexit 97\n',
        )
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {
            "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER": "1",
            "PYTHON_ARGUMENTS": str(python_arguments),
            "TORCHRUN_ARGUMENTS": str(torchrun_arguments),
            "WORKER_ENVIRONMENT": str(worker_environment),
            "FORBIDDEN_CALL": str(forbidden_call),
        }
    )

    subprocess.run(
        [
            "bash",
            str(script),
            "--rspr_mode",
            "stochastic",
            "--experiment_desc",
            "two words",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert not forbidden_call.exists()
    assert worker_environment.read_text() == "unset"
    python_args = _read_nul_arguments(python_arguments)
    assert python_args[0] == str(tmp_path / "scripts/build_msrvtt_trusted_split.py")
    manifest_index = python_args.index("--manifest")
    assert python_args[manifest_index + 1] == str(
        tmp_path / "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
    )
    torchrun_args = _read_nul_arguments(torchrun_arguments)
    assert "--nproc_per_node=2" in torchrun_args
    assert str(tmp_path / "main_task_retrieval.py") in torchrun_args
    assert "--run_final_test" in torchrun_args
    assert torchrun_args[-4:] == [
        "--rspr_mode",
        "stochastic",
        "--experiment_desc",
        "two words",
    ]


@pytest.mark.parametrize(
    ("environment_update", "arguments", "message"),
    (
        (
            {"EXPERIMENT_PROFILE": "default", "CUDA_VISIBLE_DEVICES": "gpu0"},
            (),
            "malformed CUDA_VISIBLE_DEVICES",
        ),
        (
            {
                "EXPERIMENT_PROFILE": "hygiene",
                "CUDA_VISIBLE_DEVICES": "0,1,2,4",
                "NPROC": "4",
            },
            ("--batch_size", "128"),
            "hygiene cannot override protected baseline option",
        ),
    ),
)
def test_worker_preserves_validation_failures(
    tmp_path, environment_update, arguments, message
):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    torchrun_marker = tmp_path / "torchrun.called"
    _write_executable(fake_bin / "python3", "exit 0\n")
    _write_executable(
        fake_bin / "torchrun",
        'touch "$TORCHRUN_MARKER"\n',
    )
    environment = _environment(tmp_path, fake_bin)
    environment.update(environment_update)
    environment.update(
        {
            "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER": "1",
            "TORCHRUN_MARKER": str(torchrun_marker),
        }
    )

    result = subprocess.run(
        ["bash", str(script), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not torchrun_marker.exists()


def test_repository_has_one_supported_msrvtt_training_script():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    assert not (REPOSITORY / "train_msrvtt.sh").exists()
    subprocess.run(
        ["bash", "-n", str(SCRIPT), str(REPOSITORY / "run_train_bg.sh")],
        check=True,
    )
