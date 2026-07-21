import os
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "eval.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _read_nul_arguments(path: Path) -> list[str]:
    payload = path.read_bytes()
    assert payload.endswith(b"\0")
    return [item.decode() for item in payload[:-1].split(b"\0")]


def _rspr_arguments(arguments: list[str]) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--rspr_detach_samples", "--rspr_freeze_clip", "--rspr_freeze_dsa"}:
            result[argument] = True
            index += 1
        elif argument.startswith("--rspr_"):
            result[argument] = arguments[index + 1]
            index += 2
        else:
            index += 1
    return result


def _environment(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "EVAL_SPLIT": "val",
            "INIT_MODEL": "checkpoint.bin",
            "DATATYPE": "msvd",
            "OUTPUT_DIR": str(tmp_path / "outputs"),
            "LOG_DIR": str(tmp_path / "logs"),
            "EXPERIMENT_PROFILE": "default",
        }
    )
    return environment


def test_eval_forwards_effective_rspr_defaults_and_boolean_flags(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    torchrun_arguments = tmp_path / "torchrun.args"
    _write_executable(fake_bin / "torchrun", 'printf "%s\\0" "$@" > "$TORCHRUN_ARGUMENTS"\n')
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {
            "RSPR_MODE": "stochastic",
            "RSPR_DETACH_SAMPLES": "1",
            "RSPR_FREEZE_CLIP": "1",
            "TORCHRUN_ARGUMENTS": str(torchrun_arguments),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    rspr_arguments = _rspr_arguments(_read_nul_arguments(torchrun_arguments))
    assert rspr_arguments["--rspr_mode"] == "stochastic"
    assert rspr_arguments["--rspr_sample_count"] == "4"
    assert rspr_arguments["--rspr_eval_sample_count"] == "8"
    assert rspr_arguments["--rspr_top_r"] == "100"
    assert rspr_arguments["--rspr_detach_samples"] is True
    assert rspr_arguments["--rspr_freeze_clip"] is True
    assert "--rspr_freeze_dsa" not in rspr_arguments
    assert "RSPR_MODE=stochastic" in result.stdout


def test_eval_trailing_rspr_arguments_override_environment_once(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    torchrun_arguments = tmp_path / "torchrun.args"
    _write_executable(fake_bin / "torchrun", 'printf "%s\\0" "$@" > "$TORCHRUN_ARGUMENTS"\n')
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {"RSPR_MODE": "legacy", "TORCHRUN_ARGUMENTS": str(torchrun_arguments)}
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--rspr_mode",
            "mean",
            "--rspr_sample_count",
            "1",
            "--rspr_top_r",
            "0",
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    launch_arguments = _read_nul_arguments(torchrun_arguments)
    rspr_arguments = _rspr_arguments(launch_arguments)
    assert rspr_arguments["--rspr_mode"] == "mean"
    assert rspr_arguments["--rspr_sample_count"] == "1"
    assert rspr_arguments["--rspr_top_r"] == "0"
    assert launch_arguments.count("--rspr_mode") == 1
    assert "RSPR_MODE=mean RSPR_K=1" in result.stdout


def test_eval_rejects_invalid_temperature_before_torchrun(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    torchrun_marker = tmp_path / "torchrun.called"
    _write_executable(fake_bin / "torchrun", 'touch "$TORCHRUN_MARKER"\nexit 97\n')
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {"RSPR_MATCH_TEMPERATURE": "0e0", "TORCHRUN_MARKER": str(torchrun_marker)}
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "RSPR_MATCH_TEMPERATURE=0e0" in result.stderr
    assert not torchrun_marker.exists()
