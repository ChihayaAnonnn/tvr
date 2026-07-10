import importlib.machinery
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

cv2_stub = types.ModuleType("cv2")
cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
sys.modules.setdefault("cv2", cv2_stub)
spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules["modules.spatial_enhancer"] = spatial_enhancer_stub

from main_task_retrieval import (  # noqa: E402
    _should_keep_clip_parameter_trainable,
    _trainable_named_parameters,
    _unpack_train_batch,
    get_args,
    train_epoch,
)


def tensor_id(value):
    return torch.tensor([value])


def test_get_args_parses_explicit_hard_negative_and_uacl_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            "--use_explicit_hard_negative_loss",
            "--w_hard_negative",
            "0.07",
            "--use_uacl_intra_alignment",
            "--w_uacl_intra",
            "0.02",
            "--w_uacl_kl",
            "0.0003",
            "--uacl_temperature",
            "0.05",
            "--uacl_sample_strategy",
            "random",
        ],
    )

    args = get_args()

    assert args.use_explicit_hard_negative_loss is True
    assert args.w_hard_negative == 0.07
    assert args.use_uacl_intra_alignment is True
    assert args.w_uacl_intra == 0.02
    assert args.w_uacl_kl == 0.0003
    assert args.uacl_temperature == 0.05
    assert args.uacl_sample_strategy == "random"


def test_get_args_accepts_evidential_uncertainty_mode(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            "--uncertainty_mode",
            "evidential",
        ],
    )

    args = get_args()

    assert args.uncertainty_mode == "evidential"
    assert args.experiment_profile == "default"


def test_get_args_accepts_final_score_mode_and_weights(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            "--final_score_mode",
            "wti_qc_sap",
            "--lambda_prob",
            "0.25",
            "--lambda_anchor",
            "0.5",
            "--lambda_qc_sap",
            "0.1",
            "--qc_sap_temperature",
            "0.2",
        ],
    )

    args = get_args()

    assert args.final_score_mode == "wti_qc_sap"
    assert args.lambda_prob == 0.25
    assert args.lambda_anchor == 0.5
    assert args.lambda_qc_sap == 0.1
    assert args.qc_sap_temperature == 0.2


def test_get_args_accepts_eva_clip_backbone_options(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            "--backbone_type",
            "eva_clip",
            "--backbone_name",
            "EVA02-CLIP-B-16",
            "--backbone_path",
            "ref/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt",
            "--eva_clip_root",
            "ref/EVA/EVA-CLIP/rei",
        ],
    )

    args = get_args()

    assert args.backbone_type == "eva_clip"
    assert args.backbone_name == "EVA02-CLIP-B-16"
    assert args.backbone_path == "ref/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt"
    assert args.eva_clip_root == "ref/EVA/EVA-CLIP/rei"


def test_clip_layer_norm_precision_defaults_to_fp16(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
        ],
    )

    assert get_args().clip_layer_norm_precision == "fp16"


def test_clip_layer_norm_precision_accepts_fp32(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            "--clip_layer_norm_precision",
            "fp32",
        ],
    )

    assert get_args().clip_layer_norm_precision == "fp32"


def _run_with_fake_torchrun(script_name, tmp_path, xattn_value):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_path = tmp_path / f"{script_name}.args"
    torchrun_path = fake_bin / "torchrun"
    torchrun_path.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${CAPTURE_PATH}\"\n",
        encoding="utf-8",
    )
    torchrun_path.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CAPTURE_PATH": str(capture_path),
            "EVA_CLIP_USE_XATTN": xattn_value,
            "RUN_ID": "xattn-test",
            "OUTPUT_DIR": str(tmp_path / "output"),
            "LOG_DIR": str(tmp_path / "logs"),
            "INIT_MODEL": str(tmp_path / "checkpoint.pt"),
            "EVAL_SPLIT": "val",
            "CUDA_VISIBLE_DEVICES": "0",
            "NPROC": "1",
        }
    )
    return subprocess.run(
        ["bash", str(PROJECT_ROOT / script_name)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    ), capture_path


@pytest.mark.parametrize("script_name", ["train_msrvtt.sh", "eval.sh"])
@pytest.mark.parametrize(("xattn_value", "expects_flag"), [("0", False), ("1", True)])
def test_scripts_conditionally_forward_and_log_eva_clip_xattn(script_name, xattn_value, expects_flag, tmp_path):
    result, capture_path = _run_with_fake_torchrun(script_name, tmp_path, xattn_value)

    assert result.returncode == 0, result.stderr
    captured_args = capture_path.read_text(encoding="utf-8").splitlines()
    assert ("--eva_clip_use_xattn" in captured_args) is expects_flag
    assert f"EVA_CLIP_USE_XATTN={xattn_value}" in result.stdout


@pytest.mark.parametrize("script_name", ["train_msrvtt.sh", "eval.sh"])
def test_scripts_reject_invalid_eva_clip_xattn_value(script_name, tmp_path):
    result, capture_path = _run_with_fake_torchrun(script_name, tmp_path, "yes")

    assert result.returncode == 2
    assert "EVA_CLIP_USE_XATTN=yes" in result.stderr
    assert not capture_path.exists()


def test_get_args_accepts_explicit_trusted_hygiene_contract(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--datatype",
            "msrvtt",
            "--experiment_profile",
            "hygiene",
            "--expand_msrvtt_sentences",
            "--final_score_mode",
            "wti",
            "--w_mil",
            "0",
            "--w_evidential",
            "0",
            "--w_neg_reg",
            "0",
            "--w_orth",
            "0",
            "--uncertainty_mode",
            "none",
        ],
    )

    args = get_args()

    assert args.experiment_profile == "hygiene"
    assert args.uncertainty_mode == "none"
    assert args.w_mil == 0
    assert args.w_evidential == 0
    assert args.w_neg_reg == 0
    assert args.w_orth == 0
    assert args.final_score_mode == "wti"
    assert args.use_explicit_hard_negative_loss is False
    assert args.use_uacl_intra_alignment is False


def test_get_args_keeps_non_msrvtt_hygiene_normalization(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--datatype",
            "msvd",
            "--experiment_profile",
            "hygiene",
            "--uncertainty_mode",
            "evidential",
            "--w_mil",
            "0.01",
            "--use_explicit_hard_negative_loss",
        ],
    )

    args = get_args()

    assert args.w_mil == 0.0
    assert args.uncertainty_mode == "none"
    assert args.use_explicit_hard_negative_loss is False


def test_clip_freeze_policy_recognizes_eva_visual_blocks_and_heads():
    args = SimpleNamespace(freeze_layer_num=0, linear_patch="2d")

    assert _should_keep_clip_parameter_trainable("visual.blocks.0.attn.qkv.weight", args)
    assert _should_keep_clip_parameter_trainable("visual.norm.weight", args)
    assert _should_keep_clip_parameter_trainable("visual.head.weight", args)
    assert not _should_keep_clip_parameter_trainable("visual.patch_embed.proj.weight", args)


def test_trainable_named_parameters_excludes_frozen_parameters():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 1),
    )
    model[1].weight.requires_grad = False
    model[1].bias.requires_grad = False

    names = [name for name, _param in _trainable_named_parameters(model)]

    assert names == ["0.weight", "0.bias"]


def test_unpack_train_batch_supports_explicit_hard_negative_without_attributes():
    args = SimpleNamespace(use_attributes=False, use_explicit_hard_negative_loss=True)
    batch = tuple(tensor_id(i) for i in range(10))

    unpacked = _unpack_train_batch(batch, args)

    assert unpacked["input_ids"] is batch[0]
    assert unpacked["input_mask"] is batch[1]
    assert unpacked["segment_ids"] is batch[2]
    assert unpacked["video"] is batch[3]
    assert unpacked["video_mask"] is batch[4]
    assert unpacked["sample_index"] is batch[5]
    assert unpacked["hard_video"] is batch[6]
    assert unpacked["hard_video_mask"] is batch[7]
    assert unpacked["hard_valid"] is batch[8]
    assert unpacked["video_group_id"] is batch[9]


def test_unpack_train_batch_supports_explicit_hard_negative_with_attributes():
    args = SimpleNamespace(use_attributes=True, use_explicit_hard_negative_loss=True)
    batch = tuple(tensor_id(i) for i in range(13))

    unpacked = _unpack_train_batch(batch, args)

    assert unpacked["input_ids"] is batch[0]
    assert unpacked["input_mask"] is batch[1]
    assert unpacked["segment_ids"] is batch[2]
    assert unpacked["video"] is batch[6]
    assert unpacked["video_mask"] is batch[7]
    assert unpacked["sample_index"] is batch[8]
    assert unpacked["hard_video"] is batch[9]
    assert unpacked["hard_video_mask"] is batch[10]
    assert unpacked["hard_valid"] is batch[11]
    assert unpacked["video_group_id"] is batch[12]


def test_unpack_train_batch_supports_trusted_base_group_id():
    args = SimpleNamespace(use_attributes=False, use_explicit_hard_negative_loss=False)
    batch = tuple(tensor_id(i) for i in range(6))

    unpacked = _unpack_train_batch(batch, args)

    assert unpacked["sample_index"] is None
    assert unpacked["video_group_id"] is batch[5]


def test_unpack_train_batch_supports_trusted_attributes_group_id():
    args = SimpleNamespace(use_attributes=True, use_explicit_hard_negative_loss=False)
    batch = tuple(tensor_id(i) for i in range(9))

    unpacked = _unpack_train_batch(batch, args)

    assert unpacked["video"] is batch[6]
    assert unpacked["video_mask"] is batch[7]
    assert unpacked["sample_index"] is None
    assert unpacked["video_group_id"] is batch[8]


def test_train_epoch_passes_video_group_id_to_model():
    class CapturingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.clip = torch.nn.Module()
            self.clip.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
            self.received_group_ids = None

        def forward(
            self,
            input_ids,
            segment_ids,
            input_mask,
            video,
            video_mask,
            video_group_id=None,
            **_kwargs,
        ):
            self.received_group_ids = video_group_id
            loss = self.weight.square()
            return {"total": loss, "sim_loss": loss}

    model = CapturingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    group_ids = torch.tensor([17, 18])
    batch = (
        torch.zeros(2, 1, 2, dtype=torch.long),
        torch.ones(2, 1, 2, dtype=torch.long),
        torch.zeros(2, 1, 2, dtype=torch.long),
        torch.zeros(2, 1, 1, 1, 1, 1, 1),
        torch.ones(2, 1, 1, dtype=torch.long),
        group_ids,
    )
    args = SimpleNamespace(
        n_display=1,
        gate_log_interval=None,
        gradient_accumulation_steps=1,
        epochs=1,
        log_moe_weights=False,
    )

    train_epoch(
        epoch=0,
        args=args,
        model=model,
        train_dataloader=[batch],
        device=torch.device("cpu"),
        n_gpu=1,
        optimizer=optimizer,
        scheduler=None,
        global_step=0,
        local_rank=1,
    )

    assert torch.equal(model.received_group_ids, group_ids)
