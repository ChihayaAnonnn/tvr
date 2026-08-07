import os
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_bg.sh"


def test_training_script_is_executable_and_parses():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_training_script_is_the_only_supported_entry():
    assert not (REPOSITORY / "train_msrvtt.sh").exists()
