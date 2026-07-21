import shlex
import subprocess
import sys
from pathlib import Path

from scripts.rspr_ablation_matrix import ABLATIONS

REPOSITORY = Path(__file__).resolve().parents[1]
MATRIX_SCRIPT = REPOSITORY / "scripts" / "rspr_ablation_matrix.py"


def test_ablation_matrix_covers_a0_through_a8_with_required_deltas():
    assert set(ABLATIONS) == {f"A{index}" for index in range(9)}
    assert ABLATIONS["A0"] == ["--rspr_mode", "off"]
    assert ABLATIONS["A4"] == ["--rspr_mode", "legacy"]
    assert set(ABLATIONS["A2"]) ^ set(ABLATIONS["A3"]) == {"--rspr_detach_samples"}
    assert ABLATIONS["A5"] == ["--rspr_mode", "stochastic", "--rspr_match_mode", "hard"]
    assert ABLATIONS["A6"] == ["--rspr_mode", "stochastic", "--rspr_match_mode", "soft"]
    assert ABLATIONS["A7"] == [
        "--rspr_mode",
        "stochastic",
        "--rspr_match_mode",
        "soft",
        "--rspr_rank_weight",
        "0",
    ]
    assert ABLATIONS["A8"] == [
        "--rspr_mode",
        "stochastic",
        "--rspr_match_mode",
        "soft",
        "--rspr_anchor_weight",
        "0",
    ]


def test_print_shell_args_round_trips_to_the_selected_ablation():
    result = subprocess.run(
        [sys.executable, str(MATRIX_SCRIPT), "--ablation", "A3", "--print-shell-args"],
        cwd=REPOSITORY,
        check=True,
        text=True,
        capture_output=True,
    )

    assert shlex.split(result.stdout) == ABLATIONS["A3"]


def test_training_and_evaluation_shell_scripts_are_valid_bash():
    subprocess.run(
        ["bash", "-n", "run_train_msrvtt_bg.sh", "eval.sh"],
        cwd=REPOSITORY,
        check=True,
    )
