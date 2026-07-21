import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.rspr_ablation_matrix import ABLATIONS

REPOSITORY = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = REPOSITORY / "scripts" / "rspr_ablation_matrix.py"


def test_ablation_matrix_covers_a0_through_a8_with_required_deltas():
    assert ABLATIONS == {
        "A0": ["--rspr_mode", "off"],
        "A1": ["--rspr_mode", "mean", "--rspr_sample_count", "1", "--rspr_top_r", "0"],
        "A2": ["--rspr_mode", "stochastic", "--rspr_detach_samples"],
        "A3": ["--rspr_mode", "stochastic"],
        "A4": ["--rspr_mode", "legacy"],
        "A5": ["--rspr_mode", "stochastic", "--rspr_match_mode", "hard"],
        "A6": ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft"],
        "A7": [
            "--rspr_mode",
            "stochastic",
            "--rspr_match_mode",
            "soft",
            "--rspr_rank_weight",
            "0",
        ],
        "A8": [
            "--rspr_mode",
            "stochastic",
            "--rspr_match_mode",
            "soft",
            "--rspr_anchor_weight",
            "0",
        ],
    }


@pytest.mark.parametrize("ablation", sorted(ABLATIONS))
def test_print_shell_args_round_trips_without_launching_training(ablation, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "training-launch.marker"
    for command in ("torchrun", "bash", "setsid"):
        path = fake_bin / command
        path.write_text(
            "#!/bin/bash\n"
            "touch \"$TRAINING_LAUNCH_MARKER\"\n"
            "exit 97\n"
        )
        path.chmod(0o755)
    environment = {"PATH": f"{fake_bin}:{os.environ['PATH']}", "TRAINING_LAUNCH_MARKER": str(marker)}

    result = subprocess.run(
        [sys.executable, str(MATRIX_SCRIPT), "--ablation", ablation, "--print-shell-args"],
        cwd=REPOSITORY,
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )

    assert shlex.split(result.stdout) == ABLATIONS[ablation]
    assert not marker.exists()


def test_training_and_evaluation_shell_scripts_are_valid_bash():
    subprocess.run(
        ["bash", "-n", "run_train_msrvtt_bg.sh", "eval.sh"],
        cwd=REPOSITORY,
        check=True,
    )
