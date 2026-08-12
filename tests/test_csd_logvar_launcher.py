import os
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_csd_logvar_bg.sh"


def test_launcher_registers_a1_bin4_and_primary_contract():
    text = SCRIPT.read_text()
    assert "ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4" in text
    for flag in (
        "--use_uatvr_full",
        "--use_retrieval_risk_sidecar",
        "--use_csd_logvar_sidecar",
        "--csd_logvar_sidecar_only",
        "--w_csd",
        "--w_csd_grounding",
        "--w_csd_vib",
        "--run_final_test",
    ):
        assert flag in text
    assert 'CSD_EPOCHS="${CSD_EPOCHS:-3}"' in text


def test_launcher_exposes_all_variants():
    text = SCRIPT.read_text()
    assert all(pattern in text for pattern in ('"CG")', '"C")', '"G")'))


def test_launcher_is_executable_and_parses():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
