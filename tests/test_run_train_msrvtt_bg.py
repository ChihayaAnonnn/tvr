import os
import signal
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
    assert payload.endswith(b"\0")
    return [item.decode() for item in payload[:-1].split(b"\0")]


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 2.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def _wait_for_text(path: Path, text: str) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if path.exists() and text in path.read_text():
            return
        time.sleep(0.01)
    assert path.exists()
    assert text in path.read_text()


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _process_group_is_running(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_processes_exit(process_group: int, pids: list[int]) -> None:
    deadline = time.monotonic() + 2.0
    while (
        _process_group_is_running(process_group)
        or any(_process_is_running(pid) for pid in pids)
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _process_group_is_running(process_group)
    assert not any(_process_is_running(pid) for pid in pids)


def _terminate_process_group(process_group: int, pids: list[int]) -> None:
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        _wait_for_processes_exit(process_group, pids)
    except AssertionError:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_for_processes_exit(process_group, pids)


def _terminate_controller_process_group(
    controller: subprocess.Popen[str], pids: list[int]
) -> None:
    process_group = controller.pid
    if controller.returncode is None:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        controller.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        controller.wait(timeout=2.0)
    try:
        _wait_for_processes_exit(process_group, pids)
    except AssertionError:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_for_processes_exit(process_group, pids)


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


def test_controller_cleanup_reaps_exited_session_leader():
    controller = subprocess.Popen(
        ["bash", "-c", "exit 0"],
        start_new_session=True,
    )
    try:
        os.waitid(os.P_PID, controller.pid, os.WEXITED | os.WNOWAIT)
        with pytest.raises(AssertionError):
            _terminate_process_group(controller.pid, [controller.pid])
        assert _process_is_running(controller.pid)

        _terminate_controller_process_group(controller, [controller.pid])
        assert controller.poll() is not None
        assert not _process_is_running(controller.pid)
    finally:
        try:
            controller.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(controller.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            controller.wait(timeout=2.0)


def test_controller_relaunches_same_script_and_tails_log(tmp_path):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    setsid_arguments = tmp_path / "setsid.args"
    setsid_pid = tmp_path / "setsid.pid"
    tail_arguments = tmp_path / "tail.args"
    tail_pid = tmp_path / "tail.pid"
    pid_file = tmp_path / "train.pid"
    torchrun_pid = tmp_path / "torchrun.pid"
    log_marker = "fake worker is running"
    real_setsid = shutil.which("setsid")
    real_tail = shutil.which("tail")
    assert real_setsid is not None
    assert real_tail is not None
    _write_executable(
        fake_bin / "setsid",
        (
            'printf "%s\\0" "$@" > "$SETSID_ARGUMENTS"\n'
            'printf "%s" "$$" > "$SETSID_PID"\n'
            f'exec "{real_setsid}" "$@"\n'
        ),
    )
    _write_executable(
        fake_bin / "tail",
        (
            'printf "%s\\0" "$@" > "$TAIL_ARGUMENTS"\n'
            'printf "%s" "$$" > "$TAIL_PID"\n'
            f'exec "{real_tail}" "$@"\n'
        ),
    )
    _write_executable(fake_bin / "python3", "exit 0\n")
    _write_executable(
        fake_bin / "torchrun",
        (
            'printf "%s" "$$" > "$TORCHRUN_PID"\n'
            f'printf "%s\\n" "{log_marker}"\n'
            "trap 'exit 0' INT TERM\n"
            "while :; do sleep 0.01; done\n"
        ),
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
            "SETSID_PID": str(setsid_pid),
            "TAIL_ARGUMENTS": str(tail_arguments),
            "TAIL_PID": str(tail_pid),
            "TORCHRUN_PID": str(torchrun_pid),
        }
    )

    controller = subprocess.Popen(
        [
            str(script),
            "--rspr_mode",
            "stochastic",
            "--experiment_desc",
            "two words",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    worker_pid = None
    tail_process_pid = None
    torchrun_process_pid = None
    try:
        _wait_for(pid_file)
        worker_pid = int(pid_file.read_text().strip())
        _wait_for(setsid_arguments)
        _wait_for(setsid_pid)
        _wait_for(tail_pid)
        tail_process_pid = int(tail_pid.read_text())
        _wait_for(torchrun_pid)
        torchrun_process_pid = int(torchrun_pid.read_text())
        log_file = tmp_path / "logs/20260720/121314_rspr_train_msrvtt.log"
        _wait_for_text(log_file, log_marker)
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
        assert worker_pid == int(setsid_pid.read_text())
        assert os.getsid(worker_pid) == worker_pid
        assert _process_is_running(worker_pid)
        assert _process_is_running(torchrun_process_pid)

        os.killpg(controller.pid, signal.SIGINT)
        controller.wait(timeout=2.0)
        assert _process_is_running(worker_pid)
        assert _process_is_running(torchrun_process_pid)
    finally:
        try:
            controller_pids = [controller.pid]
            if tail_process_pid is not None:
                controller_pids.append(tail_process_pid)
            _terminate_controller_process_group(controller, controller_pids)
        finally:
            if worker_pid is not None:
                worker_pids = [worker_pid]
                if torchrun_process_pid is not None:
                    worker_pids.append(torchrun_process_pid)
                _terminate_process_group(worker_pid, worker_pids)


@pytest.mark.parametrize(
    ("checkpointing", "extra_clip_arguments"),
    (
        (
            "1",
            [
                "--clip_gradient_checkpointing",
                "--clip_visual_checkpoint_layers",
                "4",
            ],
        ),
        ("0", []),
    ),
)
def test_worker_runs_split_builder_and_torchrun_without_recursing(
    tmp_path, checkpointing, extra_clip_arguments
):
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
    if checkpointing == "0":
        environment.update(
            {
                "RUN_ID": "a800_no_ckpt_unit",
                "OUTPUT_DIR": str(tmp_path / "a800_no_ckpt_unit_checkpoints"),
                "A800_THROUGHPUT_COMPARISON": "1",
            }
        )
    environment.update(
        {
            "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER": "1",
            "CLIP_GRADIENT_CHECKPOINTING": checkpointing,
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
    split_manifest = tmp_path / "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
    generated_split_dir = tmp_path / "data/generated/msrvtt_trusted_v1"
    assert _read_nul_arguments(python_arguments) == [
        str(tmp_path / "scripts/build_msrvtt_trusted_split.py"),
        "--train-csv",
        "/dataset/csv/MSRVTT_train.9k.csv",
        "--annotation-json",
        "/dataset/annotation/MSRVTT_v2.json",
        "--test-csv",
        "/dataset/csv/MSRVTT_JSFUSION_test.csv",
        "--manifest",
        str(split_manifest),
        "--output-dir",
        str(generated_split_dir),
    ]
    torchrun_args = _read_nul_arguments(torchrun_arguments)
    output_dir = Path(environment["OUTPUT_DIR"])
    assert torchrun_args == [
        "--nproc_per_node=2",
        "--master_addr=127.0.0.9",
        "--master_port=29547",
        str(tmp_path / "main_task_retrieval.py"),
        "--do_train",
        "--run_final_test",
        "--num_thread_reader",
        "8",
        "--prefetch_factor",
        "2",
        "--epochs=5",
        "--batch_size",
        "256",
        "--gradient_accumulation_steps",
        "1",
        "--n_display=20",
        "--train_csv",
        str(generated_split_dir / "train.csv"),
        "--val_csv",
        str(generated_split_dir / "val.csv"),
        "--source_train_csv",
        "/dataset/csv/MSRVTT_train.9k.csv",
        "--test_csv",
        "/dataset/csv/MSRVTT_JSFUSION_test.csv",
        "--split_manifest",
        str(split_manifest),
        "--eval_split",
        "val",
        "--data_path",
        "/dataset/annotation/MSRVTT_v2.json",
        "--features_path",
        "/dataset/videos/compressed_videos/msrvtt_224_12fps/",
        "--tqfs_cache_dir",
        "/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224",
        "--output_dir",
        str(output_dir),
        "--lr",
        "1e-4",
        "--max_words",
        "32",
        "--max_frames",
        "8",
        "--batch_size_val",
        "16",
        "--datatype",
        "msrvtt",
        "--expand_msrvtt_sentences",
        "--feature_framerate",
        "1",
        "--coef_lr",
        "1e-3",
        "--freeze_layer_num",
        "0",
        "--slice_framepos",
        "3",
        "--linear_patch",
        "2d",
        "--sim_header",
        "seqTransf",
        "--pretrained_clip_name",
        "ViT-B/16",
        "--clip_layer_norm_precision",
        "fp16",
        *extra_clip_arguments,
        "--extra_video_cls_num",
        "2",
        "--extra_text_cls_num",
        "2",
        "--experiment_profile",
        "default",
        "--experiment_desc",
        "",
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
