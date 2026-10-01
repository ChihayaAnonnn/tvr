import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("code", [0, 7])
def test_controller_records_actual_child_exit_without_identity_return_field(tmp_path, code):
    from scripts.run_dual_gpu_preflight import launch_once

    result = launch_once(
        [sys.executable, "-c", f'import sys; print("finished"); sys.exit({code})'],
        cwd=tmp_path,
        run_dir=tmp_path,
        command_sha256="a" * 64,
        timeout=10,
    )
    assert result["observed_launcher_exit_code"] == code
    assert result["terminal_exit_code"] == code
    assert "finished" in (tmp_path / "logs/worker.log").read_text()
    assert json.loads((tmp_path / "provenance/controller-launch.json").read_text())["command_sha256"] == "a" * 64


def test_metadata_failure_after_spawn_stops_and_reaps_original_child(tmp_path, monkeypatch):
    from scripts import run_dual_gpu_preflight as controller

    original = controller.write_json

    def rejected(path, payload):
        if Path(path).name == "controller-launch.json":
            raise OSError("planned metadata failure")
        return original(path, payload)

    monkeypatch.setattr(controller, "write_json", rejected)
    result = controller.launch_once(
        [sys.executable, "-c", "import time;time.sleep(60)"],
        cwd=tmp_path,
        run_dir=tmp_path,
        command_sha256="a" * 64,
        timeout=10,
    )
    assert result["terminal_exit_code"] == 1
    assert result["observed_launcher_exit_code"] is not None
    assert "planned metadata failure" in result["reason"]
