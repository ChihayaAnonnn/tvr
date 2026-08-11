import os
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_risk_sidecar_bg.sh"


def test_launcher_uses_frozen_june_sidecar_contract():
    text = SCRIPT.read_text()

    assert "ckpt_msrvtt_20260617_b1only_v2_repeat1/pytorch_model.bin.4" in text
    for flag in (
        "--use_uatvr_full",
        "--use_ada_norm",
        "--use_retrieval_risk_sidecar",
        "--retrieval_risk_sidecar_only",
        "--w_retrieval_risk",
        "--dsl_temperature",
    ):
        assert flag in text
    assert "--DSL True" in text
    assert "--dsl_temperature 2.0" in text


def test_launcher_is_executable_and_parses():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
