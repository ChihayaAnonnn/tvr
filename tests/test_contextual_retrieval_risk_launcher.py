import os
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_contextual_risk_bg.sh"


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
    assert 'EXPERIMENT_PROFILE=hygiene' in text
    assert 'TRAIN_BATCH_SIZE=256' in text
    assert 'TRAIN_GRADIENT_ACCUMULATION_STEPS=1' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,3,4}"' in text
    assert 'NPROC=4' in text
    assert 'RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"' in text
    assert 'RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"' in text


def test_registered_a3_launcher_rejects_output_collisions_and_other_sidecar_only_modes():
    assert SCRIPT.is_file(), "registered A3 launcher is missing"
    text = SCRIPT.read_text()

    assert 'if [[ -e "${OUTPUT_DIR}" ]]' in text
    assert "A3 output directory already exists" in text
    assert "--fp16" not in text
    for forbidden in (
        "--retrieval_risk_sidecar_only",
        "--csd_logvar_sidecar_only",
        "--intrinsic_concentration_only",
    ):
        assert forbidden not in text


def test_registered_a3_launcher_is_executable_and_parses():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
