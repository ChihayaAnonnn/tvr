import sys
from pathlib import Path

import pytest

from main_task_retrieval import get_args


REPOSITORY = Path(__file__).resolve().parents[1]
CANONICAL_A1 = (
    REPOSITORY
    / "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
)


def _parse(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", [
        "main_task_retrieval.py", "--do_train", "--expand_msrvtt_sentences",
        "--output_dir", "/tmp/csd-logvar-config-test", *extra,
    ])
    return get_args()


def test_old_cli_keeps_csd_logvar_disabled(monkeypatch):
    args = _parse(monkeypatch, [])
    assert args.use_csd_logvar_sidecar is False
    assert args.csd_logvar_sidecar_only is False
    assert (args.w_csd, args.w_csd_grounding, args.w_csd_vib) == (1.0, 0.05, 1e-4)
    assert (args.csd_logvar_min, args.csd_logvar_max, args.csd_logvar_init) == (-12.0, 4.0, -8.0)


@pytest.mark.parametrize("extra", [
    ["--csd_logvar_sidecar_only"],
    ["--use_csd_logvar_sidecar", "--csd_logvar_sidecar_only", "--use_uatvr_full"],
    ["--use_csd_logvar_sidecar", "--use_uatvr_full", "--w_csd", "-1"],
    ["--use_csd_logvar_sidecar", "--use_uatvr_full", "--csd_logvar_min", "4", "--csd_logvar_max", "-12"],
])
def test_invalid_csd_cli_contracts_fail(monkeypatch, extra):
    with pytest.raises(ValueError):
        _parse(monkeypatch, extra)


def _a2_only(extra):
    return [
        "--use_uatvr_full", "--use_retrieval_risk_sidecar",
        "--use_csd_logvar_sidecar", "--csd_logvar_sidecar_only", *extra,
    ]


@pytest.mark.parametrize("extra", [
    [],
    ["--init_model", "/tmp/not-the-canonical-a1.bin"],
])
def test_fresh_a2_only_requires_canonical_a1_checkpoint(monkeypatch, extra):
    with pytest.raises(ValueError):
        _parse(monkeypatch, _a2_only(extra))


@pytest.mark.parametrize("init_model", [
    "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4",
    str(CANONICAL_A1),
])
def test_fresh_a2_only_accepts_resolved_canonical_a1_checkpoint(
    monkeypatch, init_model
):
    monkeypatch.chdir(REPOSITORY)
    args = _parse(monkeypatch, _a2_only(["--init_model", init_model]))
    assert Path(args.init_model).resolve() == CANONICAL_A1.resolve()


def test_a2_only_continuation_accepts_noncanonical_init_checkpoint(monkeypatch):
    args = _parse(monkeypatch, _a2_only([
        "--init_model", "/tmp/a2-bin.2",
        "--resume_model", "/tmp/a2-optimizer-bin.2",
    ]))
    assert args.init_model == "/tmp/a2-bin.2"
    assert args.resume_model == "/tmp/a2-optimizer-bin.2"


def test_training_rejects_csd_sidecar_without_sidecar_only(monkeypatch):
    with pytest.raises(ValueError, match="csd_logvar_sidecar_only"):
        _parse(monkeypatch, [
            "--use_uatvr_full",
            "--use_csd_logvar_sidecar",
        ])


def test_a3_and_csd_sidecar_only_modes_are_mutually_exclusive(monkeypatch):
    with pytest.raises(ValueError, match="sidecar_only"):
        _parse(monkeypatch, [
            "--use_uatvr_full",
            "--use_retrieval_risk_sidecar",
            "--use_csd_logvar_sidecar",
            "--csd_logvar_sidecar_only",
            "--use_contextual_risk_sidecar",
            "--contextual_risk_sidecar_only",
        ])


def test_evaluation_may_report_csd_without_sidecar_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main_task_retrieval.py",
        "--do_eval",
        "--expand_msrvtt_sentences",
        "--output_dir",
        "/tmp/csd-logvar-config-eval-test",
        "--init_model",
        "/tmp/csd-logvar-eval-checkpoint.pt",
        "--use_uatvr_full",
        "--use_csd_logvar_sidecar",
    ])

    args = get_args()

    assert args.do_train is False
    assert args.use_csd_logvar_sidecar is True
    assert args.csd_logvar_sidecar_only is False
