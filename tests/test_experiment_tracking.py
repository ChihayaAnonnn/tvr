"""Task 7 — CRF training config + experiment manifest snapshot tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from crf_ladder import TRAIN_CRFS, X264_RECIPE
from experiment_tracking import build_experiment_manifest


def _crf_args_namespace(**overrides):
    """Minimal argparse-like namespace populated with CRF defaults."""

    defaults = {
        "aleatoric_crf_training": True,
        "crf_ladder_root": "data/generated/msrvtt_crf",
        "crf_order_fraction": 0.20,
        "aleatoric_base_margin": 0.05,
        "aleatoric_lambda_cf": 1.0,
        "aleatoric_lambda_cv": 1.0,
        "aleatoric_lambda_of": 1.0,
        "aleatoric_lambda_ov": 1.0,
        "aleatoric_prior_weight": 0.0,
        "aleatoric_ranking_weight": 1.0,
        "aleatoric_pooling_temperature": 0.0,
        "split_manifest": "dataloaders/splits/msrvtt_trusted_v1_seed0.json",
        "datatype": "msrvtt",
        "eval_split": "val",
        "run_final_test": False,
        "seed": 0,
        "experiment_profile": "default",
        "pretrained_clip_name": "ViT-B/16",
        "num_thread_reader": 4,
        "prefetch_factor": 2,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_crf_default_config_matches_plan_registrations():
    args = _crf_args_namespace()
    assert args.aleatoric_prior_weight == 0.0
    assert args.aleatoric_pooling_temperature == 0.0
    assert args.aleatoric_ranking_weight == 1.0
    assert args.aleatoric_base_margin == 0.05
    assert args.aleatoric_lambda_cf == 1.0
    assert args.aleatoric_lambda_cv == 1.0
    assert args.aleatoric_lambda_of == 1.0
    assert args.aleatoric_lambda_ov == 1.0
    assert args.crf_order_fraction == 0.20
    assert args.datatype == "msrvtt"
    assert args.eval_split == "val"
    assert args.run_final_test is False


def test_experiment_manifest_records_crf_fields_when_enabled():
    args = _crf_args_namespace()
    payload = build_experiment_manifest(
        args=args,
        split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 4},
        git_state={"commit": "abc"},
    )
    assert "crf" in payload
    crf = payload["crf"]
    assert crf["training_crfs"] == list(TRAIN_CRFS)
    assert crf["base_margin"] == 0.05
    assert crf["order_fraction"] == 0.20
    assert crf["lambdas"] == {"cf": 1.0, "cv": 1.0, "of": 1.0, "ov": 1.0}
    assert crf["prior_weight"] == 0.0
    assert crf["pooling_temperature"] == 0.0
    assert crf["ladder_root"] == "data/generated/msrvtt_crf"
    assert crf["x264_recipe"] == {
        "preset": X264_RECIPE.preset,
        "pixel_format": X264_RECIPE.pixel_format,
        "color_range": X264_RECIPE.color_range,
        "color_primaries": X264_RECIPE.color_primaries,
        "color_trc": X264_RECIPE.color_trc,
        "colorspace": X264_RECIPE.colorspace,
        "time_base_policy": X264_RECIPE.time_base_policy,
        "keyint": X264_RECIPE.keyint,
        "min_keyint": X264_RECIPE.min_keyint,
        "scenecut": X264_RECIPE.scenecut,
        "b_frames": X264_RECIPE.b_frames,
    }


def test_experiment_manifest_omits_crf_fields_for_hygiene_runs():
    """Standard runs without --aleatoric_crf_training MUST NOT gain enabled CRF fields."""
    args = _crf_args_namespace(aleatoric_crf_training=False)
    payload = build_experiment_manifest(
        args=args,
        split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 1},
        git_state={"commit": "abc"},
    )
    assert "crf" not in payload


def test_experiment_manifest_records_retrieval_risk_contract():
    args = _crf_args_namespace(
        aleatoric_crf_training=False,
        use_retrieval_risk_sidecar=True,
        retrieval_risk_sidecar_only=True,
        w_retrieval_risk=0.05,
        retrieval_risk_temperature=1.0,
        retrieval_risk_hidden_dim=32,
        retrieval_risk_bootstrap_samples=1000,
        init_model="/tmp/june-bin.4",
        DSL=True,
        dsl_temperature=2.0,
    )

    payload = build_experiment_manifest(
        args=args,
        split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 4},
        git_state={"commit": "abc"},
    )

    assert payload["retrieval_risk"] == {
        "enabled": True,
        "sidecar_only": True,
        "weight": 0.05,
        "temperature": 1.0,
        "hidden_dim": 32,
        "bootstrap_samples": 1000,
        "init_model": "/tmp/june-bin.4",
        "dsl": True,
        "dsl_temperature": 2.0,
        "score_path": "raw_wti_no_risk_rescaling",
    }


def test_experiment_manifest_records_csd_logvar_contract():
    args = _crf_args_namespace(
        aleatoric_crf_training=False, use_retrieval_risk_sidecar=True,
        use_csd_logvar_sidecar=True, csd_logvar_sidecar_only=True,
        w_csd=1.0, w_csd_grounding=0.05, w_csd_vib=1e-4,
        csd_logvar_hidden_dim=128, csd_logvar_min=-12.0,
        csd_logvar_max=4.0, csd_logvar_init=-8.0,
        csd_initial_scale=5.0, csd_initial_shift=5.0,
        csd_hardness_temperature=1.0, csd_bootstrap_samples=1000,
        init_model="/tmp/a2-bin.2", resume_model="/tmp/a2-optimizer-bin.2",
    )
    payload = build_experiment_manifest(
        args, split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 4}, git_state={"commit": "abc"},
    )
    assert payload["csd_logvar"] == {
        "enabled": True, "sidecar_only": True,
        "weights": {"csd": 1.0, "grounding": 0.05, "vib": 1e-4},
        "hidden_dim": 128, "bounds": [-12.0, 4.0], "init_logvar": -8.0,
        "initial_scale": 5.0, "initial_shift": 5.0,
        "hardness_temperature": 1.0, "bootstrap_samples": 1000,
        "run_mode": "resume",
        "canonical_a1_checkpoint": str(
            Path(__file__).resolve().parents[1]
            / "ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/pytorch_model.bin.4"
        ),
        "canonical_a1_sha256": (
            "a2c074e3988b0e4a115d054c352785271712ffdfb4e4c2da8b372bde424e2607"
        ),
        "init_model": "/tmp/a2-bin.2",
        "resume_model": "/tmp/a2-optimizer-bin.2",
        "score_path": "auxiliary_only_raw_wti_unchanged",
    }


def test_experiment_manifest_marks_fresh_csd_logvar_run():
    payload = build_experiment_manifest(
        _crf_args_namespace(
            aleatoric_crf_training=False, use_csd_logvar_sidecar=True,
            init_model="/tmp/a1-bin.4", resume_model="",
        ),
        split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 1}, git_state={"commit": "abc"},
    )
    assert payload["csd_logvar"]["run_mode"] == "fresh"
    assert payload["csd_logvar"]["resume_model"] == ""


def test_experiment_manifest_omits_csd_logvar_when_disabled():
    payload = build_experiment_manifest(
        _crf_args_namespace(aleatoric_crf_training=False),
        split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 1}, git_state={"commit": "abc"},
    )
    assert "csd_logvar" not in payload


def test_experiment_manifest_records_ladder_manifest_sha_when_present(tmp_path):
    ladder_root = tmp_path / "ladder"
    (ladder_root / "crf30").mkdir(parents=True)
    (ladder_root / "crf30" / "manifest.jsonl").write_text('{"video_id": "x"}\n')
    args = _crf_args_namespace(crf_ladder_root=str(ladder_root))
    payload = build_experiment_manifest(
        args=args,
        split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 1},
        git_state={"commit": "abc"},
    )
    crf = payload["crf"]
    assert "ladder_manifest_sha256_by_level" in crf
    assert "crf30" in crf["ladder_manifest_sha256_by_level"]
    assert len(crf["ladder_manifest_sha256_by_level"]["crf30"]) == 64


def test_crf_args_parse_from_cli_produces_expected_defaults():
    """Round-trip through the real argparse parser."""

    import main_task_retrieval as m

    argv = [
        "--do_train",
        "--aleatoric_crf_training",
        "--output_dir", "/tmp/does-not-matter",
        "--datatype", "msrvtt",
        "--split_manifest", "dataloaders/splits/msrvtt_trusted_v1_seed0.json",
        "--expand_msrvtt_sentences",
    ]
    import sys
    saved_argv = sys.argv[:]
    try:
        sys.argv = ["main_task_retrieval.py", *argv]
        args = m.get_args()
    finally:
        sys.argv = saved_argv

    assert args.aleatoric_crf_training is True
    assert args.aleatoric_prior_weight == 0.0
    assert args.aleatoric_pooling_temperature == 0.0
    assert args.aleatoric_ranking_weight == 1.0
    assert args.aleatoric_base_margin == 0.05
    assert args.aleatoric_lambda_cf == 1.0
    assert args.aleatoric_lambda_cv == 1.0
    assert args.aleatoric_lambda_of == 1.0
    assert args.aleatoric_lambda_ov == 1.0
    assert args.crf_order_fraction == 0.20
    assert args.crf_ladder_root == "data/generated/msrvtt_crf"
    assert args.run_final_test is False
    assert args.eval_split == "val"


def test_crftrainbatch_flows_through_train_batch_dispatch():
    """train_epoch's batch-move code path must recognize CRFTrainBatch and use
    its `.to(device)` method rather than trying to iterate it as a tuple."""

    import torch

    from dataloaders.crf_pairing import CRFTrainBatch
    from main_task_retrieval import _move_batch_to_device

    batch = CRFTrainBatch(
        input_ids=torch.zeros(2, 4, dtype=torch.long),
        input_mask=torch.zeros(2, 4, dtype=torch.long),
        segment_ids=torch.zeros(2, 4, dtype=torch.long),
        clean_video=torch.zeros(2, 3, 1, 3, 4, 4),
        video_mask=torch.zeros(2, 3, dtype=torch.long),
        group_ids=torch.zeros(2, dtype=torch.long),
        degraded_video=torch.zeros(2, 3, 1, 3, 4, 4),
        primary_rank=torch.tensor([1, 2]),
        order_video=torch.zeros(1, 3, 1, 3, 4, 4),
        order_rank=torch.tensor([[1, 2]]),
        order_batch_indices=torch.tensor([0]),
    )
    moved = _move_batch_to_device(batch, torch.device("cpu"))
    assert isinstance(moved, CRFTrainBatch)
    assert moved.clean_video.device == torch.device("cpu")
