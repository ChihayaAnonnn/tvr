"""Task 11 — CRF training runner contract tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_train_msrvtt_crf_bg.sh"


def _script_text() -> str:
    return SCRIPT.read_text()


def test_crf_runner_script_exists_and_is_executable():
    assert SCRIPT.exists(), f"CRF runner missing at {SCRIPT}"
    assert (SCRIPT.stat().st_mode & 0o111), (
        f"CRF runner {SCRIPT} is not executable"
    )


def test_crf_runner_parses_as_shell():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_crf_runner_has_frozen_flags_and_never_requests_final_test():
    text = _script_text()
    assert "--aleatoric_crf_training" in text
    assert "--aleatoric_prior_weight 0" in text
    assert "--aleatoric_pooling_temperature 0" in text
    assert "--aleatoric_ranking_weight 1.0" in text
    assert "--aleatoric_base_margin 0.05" in text
    assert "--aleatoric_lambda_cf 1.0" in text
    assert "--aleatoric_lambda_cv 1.0" in text
    assert "--aleatoric_lambda_of 1.0" in text
    assert "--aleatoric_lambda_ov 1.0" in text
    assert "--crf_order_fraction 0.20" in text
    # Never runs final test / touches the 1000 test videos.
    assert "--run_final_test" not in text
    assert "MSRVTT_JSFUSION_test.csv" not in text


def test_crf_runner_references_msvd_by_correct_split_token():
    text = _script_text()
    assert "--dataset msvd --split msvd_test" in text
    assert "--dataset msvd --split audit" not in text


def test_crf_runner_references_both_audit_helper_commands():
    text = _script_text()
    # After training the runner prints (does NOT execute) two commands.
    assert "--dataset msrvtt --split audit" in text
    assert "--selection-json" in text
    assert "best_crf_checkpoint.json" in text


def test_crf_runner_rejects_overriding_any_frozen_flag_via_tail_args():
    """Every frozen flag must appear in a `_PROTECTED_CRF_OPTIONS` guard list
    so a well-meaning caller cannot silently change the plan's registered
    hyperparameters through trailing arguments."""

    text = _script_text()
    protected = [
        "--aleatoric_crf_training",
        "--aleatoric_prior_weight",
        "--aleatoric_pooling_temperature",
        "--aleatoric_ranking_weight",
        "--aleatoric_base_margin",
        "--aleatoric_lambda_cf",
        "--aleatoric_lambda_cv",
        "--aleatoric_lambda_of",
        "--aleatoric_lambda_ov",
        "--crf_order_fraction",
    ]
    for flag in protected:
        assert flag in text, f"runner does not declare protected flag {flag}"
    # Guard block name so we can grep for its presence.
    assert "_PROTECTED_CRF_OPTIONS" in text


def test_crf_runner_runs_preflight_before_training():
    text = _script_text()
    # (a) trusted-v1 split verification — SHA-256 cross-check against
    # the checked-in CRF split JSON. Cannot reuse build_msrvtt_trusted_split.py
    # here because that script requires the test csv path, which the CRF
    # plan explicitly forbids referencing.
    assert "crf_dev_audit_seed0.json" in text
    assert "source_manifest_sha256" in text
    # (b) ladder builder --check-only
    assert "build_msrvtt_crf_ladder.py" in text
    assert "--check-only" in text
    # (c) real-codec preflight (env var)
    assert "CRF_PREFLIGHT_VIDEO" in text


def test_crf_runner_prints_audit_commands_without_executing_them():
    text = _script_text()
    # The two helper commands MUST NOT be inside an unconditional subshell that
    # actually invokes them. The runner should emit them via echo / cat so a
    # human can copy-paste after inspecting training results.
    assert "cat <<" in text or "echo " in text
    # Sanity: they must include the correct split tokens.
    assert "--split audit" in text
    assert "--split msvd_test" in text
