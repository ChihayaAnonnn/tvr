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

RETIRED_CLI_CASES = (
    ("--gate_log_interval", "10"),
    ("--gate_log_dir", "/tmp/gates"),
    ("--log_moe_weights", None),
    ("--moe_log_dir", "/tmp/moe"),
    ("--use_mil", None),
    ("--sampled_use_mil", None),
    ("--n_video_embeddings", "7"),
    ("--n_text_embeddings", "7"),
    ("--mamba_lr_ratio", "0.1"),
    ("--uncertainty_text_head", "text"),
    ("--log_sigma_min", "-1.5"),
    ("--log_sigma_max", "4"),
    ("--rope_mode", "2d"),
    ("--disable_spatial_enhancer", None),
    ("--num_expansion_tokens", "4"),
    ("--use_ada_norm", None),
    ("--eval_branch_mode", "base_only"),
    ("--disable_query_gate_in_retrieval", None),
    ("--fusion_mode", "prob_mos"),
    ("--w_mil", "0"),
    ("--w_evidential", "0"),
    ("--w_neg_reg", "0"),
    ("--final_score_mode", "wti"),
    ("--lambda_prob", "0"),
    ("--lambda_anchor", "0"),
    ("--lambda_qc_sap", "0"),
    ("--qc_sap_temperature", "0.1"),
    ("--w_uncertainty_reg", "0"),
    ("--w_orth", "0"),
    ("--w_query_sim", "0"),
    ("--use_uacl_intra_alignment", None),
    ("--w_uacl_intra", "0"),
    ("--w_uacl_kl", "0"),
    ("--uacl_temperature", "0.07"),
    ("--uacl_sample_strategy", "closest"),
    ("--anneal_warmup_epochs", "0"),
    ("--warmup_steps", "500"),
    ("--uncertainty_mode", "none"),
    ("--fusion_temperature", "1.5"),
)


def tensor_id(value):
    return torch.tensor([value])


@pytest.mark.parametrize(("flag", "value"), RETIRED_CLI_CASES)
def test_get_args_rejects_retired_cli(flag, value, monkeypatch, capsys):
    argv = [
        "prog",
        "--do_train",
        "--output_dir",
        "/tmp/uatvr-test-out",
        "--expand_msrvtt_sentences",
        flag,
    ]
    if value is not None:
        argv.append(value)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        get_args()

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr
    assert flag in stderr


def test_get_args_parses_explicit_hard_negative_flags(monkeypatch):
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
        ],
    )

    args = get_args()

    assert args.use_explicit_hard_negative_loss is True
    assert args.w_hard_negative == 0.07


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--num_thread_reader", "-1", "--num_thread_reader must be non-negative"),
        ("--prefetch_factor", "0", "--prefetch_factor must be positive"),
    ],
)
def test_dataloader_cli_rejects_invalid_worker_settings(
    monkeypatch, flag, value, message
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            flag,
            value,
        ],
    )

    with pytest.raises(ValueError, match=message):
        get_args()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--batch_size", "320", "hygiene requires --batch_size=256"),
        (
            "--gradient_accumulation_steps",
            "2",
            "hygiene requires --gradient_accumulation_steps=1",
        ),
    ],
)
def test_hygiene_cli_rejects_changed_batch_protocol(
    monkeypatch, flag, value, message
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            "--experiment_profile",
            "hygiene",
            flag,
            value,
        ],
    )

    with pytest.raises(ValueError, match=message):
        get_args()


def test_cli_rejects_abbreviated_long_options(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
            "--experiment_p",
            "default",
            "--gradient_accumulation_step",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        get_args()

    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


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
            "research_refs/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt",
            "--eva_clip_root",
            "research_refs/EVA/EVA-CLIP/rei",
        ],
    )

    args = get_args()

    assert args.backbone_type == "eva_clip"
    assert args.backbone_name == "EVA02-CLIP-B-16"
    assert args.backbone_path == "research_refs/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt"
    assert args.eva_clip_root == "research_refs/EVA/EVA-CLIP/rei"


def test_get_args_defaults_eva_paths_to_research_refs(monkeypatch):
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

    args = get_args()

    assert args.backbone_path == "research_refs/model_weights/eva_clip/EVA02_CLIP_B_psz16_s8B.pt"
    assert args.eva_clip_root == "research_refs/EVA/EVA-CLIP/rei"


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


def test_clip_gradient_checkpointing_is_explicit(monkeypatch):
    base_argv = [
        "prog",
        "--do_train",
        "--output_dir",
        "/tmp/uatvr-test-out",
        "--expand_msrvtt_sentences",
    ]
    monkeypatch.setattr("sys.argv", base_argv)
    assert get_args().clip_gradient_checkpointing is False
    assert get_args().clip_visual_checkpoint_layers == 4

    monkeypatch.setattr(
        "sys.argv",
        [
            *base_argv,
            "--clip_gradient_checkpointing",
            "--clip_visual_checkpoint_layers",
            "6",
        ],
    )
    args = get_args()
    assert args.clip_gradient_checkpointing is True
    assert args.clip_visual_checkpoint_layers == 6


def _run_with_fake_torchrun(
    script_name,
    tmp_path,
    xattn_value,
    layer_norm_precision="fp16",
    gradient_checkpointing="1",
    visual_checkpoint_layers="4",
    extra_env=None,
    script_args=None,
):
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
    visible_devices = "0,1,2,4" if script_name == "train_msrvtt.sh" else "0"
    nproc = "4" if script_name == "train_msrvtt.sh" else "1"
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "CAPTURE_PATH": str(capture_path),
            "EVA_CLIP_USE_XATTN": xattn_value,
            "CLIP_LAYER_NORM_PRECISION": layer_norm_precision,
            "CLIP_GRADIENT_CHECKPOINTING": gradient_checkpointing,
            "CLIP_VISUAL_CHECKPOINT_LAYERS": visual_checkpoint_layers,
            "RUN_ID": "xattn-test",
            "OUTPUT_DIR": str(tmp_path / "output"),
            "LOG_DIR": str(tmp_path / "logs"),
            "INIT_MODEL": str(tmp_path / "checkpoint.pt"),
            "EVAL_SPLIT": "val",
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "NPROC": nproc,
        }
    )
    for name, value in (extra_env or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return subprocess.run(
        ["bash", str(PROJECT_ROOT / script_name), *(script_args or [])],
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


@pytest.mark.parametrize("script_name", ["train_msrvtt.sh", "eval.sh"])
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_scripts_forward_and_log_clip_layer_norm_precision(
    script_name, precision, tmp_path
):
    result, capture_path = _run_with_fake_torchrun(
        script_name, tmp_path, "0", layer_norm_precision=precision
    )

    assert result.returncode == 0, result.stderr
    captured_args = capture_path.read_text(encoding="utf-8").splitlines()
    index = captured_args.index("--clip_layer_norm_precision")
    assert captured_args[index + 1] == precision
    assert f"CLIP_LAYER_NORM_PRECISION={precision}" in result.stdout


@pytest.mark.parametrize("script_name", ["train_msrvtt.sh", "eval.sh"])
def test_scripts_reject_invalid_clip_layer_norm_precision(script_name, tmp_path):
    result, capture_path = _run_with_fake_torchrun(
        script_name, tmp_path, "0", layer_norm_precision="tf32"
    )

    assert result.returncode == 2
    assert "CLIP_LAYER_NORM_PRECISION=tf32" in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize(("value", "expects_flag"), [("0", False), ("1", True)])
def test_train_script_controls_clip_gradient_checkpointing(
    value, expects_flag, tmp_path
):
    comparison_env = None
    if value == "0":
        comparison_env = {
            "A800_THROUGHPUT_COMPARISON": "1",
            "RUN_ID": "a800_no_ckpt_test",
            "OUTPUT_DIR": str(tmp_path / "a800_no_ckpt_test"),
        }
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        gradient_checkpointing=value,
        extra_env=comparison_env,
    )

    assert result.returncode == 0, result.stderr
    captured_args = capture_path.read_text(encoding="utf-8").splitlines()
    assert ("--clip_gradient_checkpointing" in captured_args) is expects_flag
    assert f"CLIP_GRADIENT_CHECKPOINTING={value}" in result.stdout
    if expects_flag:
        index = captured_args.index("--clip_visual_checkpoint_layers")
        assert captured_args[index + 1] == "4"


def test_train_script_rejects_invalid_clip_gradient_checkpointing(tmp_path):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        gradient_checkpointing="yes",
    )

    assert result.returncode == 2
    assert "CLIP_GRADIENT_CHECKPOINTING=yes" in result.stderr
    assert not capture_path.exists()


def test_train_script_rejects_invalid_clip_visual_checkpoint_layers(tmp_path):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        visual_checkpoint_layers="four",
    )

    assert result.returncode == 2
    assert "CLIP_VISUAL_CHECKPOINT_LAYERS=four" in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize(
    ("extra_env", "message"),
    [
        (
            {"A800_THROUGHPUT_COMPARISON": "0"},
            "requires A800_THROUGHPUT_COMPARISON=1",
        ),
        (
            {
                "A800_THROUGHPUT_COMPARISON": "1",
                "RUN_ID": "ordinary-run",
            },
            "RUN_ID must start with a800_no_ckpt_",
        ),
        (
            {
                "A800_THROUGHPUT_COMPARISON": "1",
                "RUN_ID": "a800_no_ckpt_test",
                "OUTPUT_DIR": "/tmp/shared-output",
            },
            "OUTPUT_DIR must contain RUN_ID",
        ),
    ],
)
def test_checkpoint_off_requires_comparison_identity(
    tmp_path, extra_env, message
):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        gradient_checkpointing="0",
        extra_env=extra_env,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not capture_path.exists()


def test_train_script_forwards_a800_pipeline_settings(tmp_path):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        extra_env={
            "TRAIN_NUM_WORKERS": "12",
            "TRAIN_PREFETCH_FACTOR": "4",
            "TRAIN_BATCH_SIZE": "256",
            "TRAIN_GRADIENT_ACCUMULATION_STEPS": "1",
            "TQFS_CACHE_DIR": "/nvme/tqfs",
        },
    )

    assert result.returncode == 0, result.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--num_thread_reader") + 1] == "12"
    assert args[args.index("--prefetch_factor") + 1] == "4"
    assert args[args.index("--batch_size") + 1] == "256"
    assert args[args.index("--gradient_accumulation_steps") + 1] == "1"
    assert args[args.index("--tqfs_cache_dir") + 1] == "/nvme/tqfs"


def test_train_script_uses_a800_defaults(tmp_path):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        extra_env={
            "CUDA_VISIBLE_DEVICES": None,
            "NPROC": None,
            "TRAIN_NUM_WORKERS": None,
            "TRAIN_PREFETCH_FACTOR": None,
            "TQFS_CACHE_DIR": None,
        },
    )

    assert result.returncode == 0, result.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert "CUDA_VISIBLE_DEVICES=0,1,2,4" in result.stdout
    assert args[args.index("--nproc_per_node=4")] == "--nproc_per_node=4"
    assert args[args.index("--num_thread_reader") + 1] == "8"
    assert args[args.index("--prefetch_factor") + 1] == "2"
    assert args[args.index("--tqfs_cache_dir") + 1] == (
        "/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224"
    )


@pytest.mark.parametrize(
    ("visible", "nproc", "message"),
    [
        ("0,1,2", "3", "requires exactly 4 GPUs"),
        ("0,1,2,2", "4", "duplicate GPU IDs"),
        ("0,1,2,4", "3", "NPROC=3 does not match"),
    ],
)
def test_hygiene_train_script_rejects_invalid_gpu_world(
    tmp_path, visible, nproc, message
):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        extra_env={"CUDA_VISIBLE_DEVICES": visible, "NPROC": nproc},
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize("visible", ["0,1,2, 4", "0,1,,4", "0,1,2,x"])
def test_hygiene_train_script_rejects_malformed_gpu_list(tmp_path, visible):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        extra_env={"CUDA_VISIBLE_DEVICES": visible, "NPROC": "4"},
    )

    assert result.returncode == 2
    assert "malformed CUDA_VISIBLE_DEVICES" in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize(
    ("extra_env", "message"),
    [
        ({"TRAIN_BATCH_SIZE": "320"}, "requires TRAIN_BATCH_SIZE=256"),
        (
            {"TRAIN_GRADIENT_ACCUMULATION_STEPS": "2"},
            "requires TRAIN_GRADIENT_ACCUMULATION_STEPS=1",
        ),
    ],
)
def test_hygiene_train_script_rejects_changed_batch_protocol(
    tmp_path, extra_env, message
):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh", tmp_path, "0", extra_env=extra_env
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize(
    ("extra_env", "message"),
    [
        ({"TRAIN_NUM_WORKERS": "-1"}, "Unsupported TRAIN_NUM_WORKERS=-1"),
        (
            {"TRAIN_PREFETCH_FACTOR": "0"},
            "Unsupported TRAIN_PREFETCH_FACTOR=0",
        ),
    ],
)
def test_train_script_rejects_invalid_pipeline_worker_settings(
    tmp_path, extra_env, message
):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh", tmp_path, "0", extra_env=extra_env
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize(
    "script_args",
    [
        ["--batch_size", "320"],
        ["--batch_size=320"],
        ["--gradient_accumulation_steps", "2"],
        ["--gradient_accumulation_step", "2"],
        ["--experiment_profile", "default"],
        ["--experiment_p", "default"],
        ["--backbone_type", "eva_clip"],
    ],
)
def test_hygiene_train_script_rejects_protected_cli_override(
    tmp_path, script_args
):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh", tmp_path, "0", script_args=script_args
    )

    assert result.returncode == 2
    assert "cannot override protected P0 option" in result.stderr
    assert not capture_path.exists()


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


@pytest.mark.parametrize(
    "name",
    [
        "_log_moe_weights_tsv",
        "_log_causal_summary_tsv",
        "_log_eval_stats_tsv",
    ],
)
def test_retired_auxiliary_loggers_are_absent(name):
    import main_task_retrieval

    assert not hasattr(main_task_retrieval, name)


def test_mus_logger_remains_available_for_wti_risk_diagnostics():
    import inspect

    import main_task_retrieval

    assert hasattr(main_task_retrieval, "_log_mus_scores_tsv")
    source = inspect.getsource(main_task_retrieval.eval_epoch)
    assert source.rindex("_log_mus_scores_tsv") > source.rindex(
        "sim_matrix = np.concatenate"
    )


def test_optimizer_source_has_no_uncertainty_mamba_group():
    import inspect

    import main_task_retrieval as retrieval

    source = inspect.getsource(retrieval.prep_optimizer)

    assert "mamba_keywords" not in source
    assert "mamba_lr_ratio" not in source


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


def test_train_epoch_passes_video_group_id_to_model(tmp_path):
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
            return {
                "total": loss,
                "sim_loss": loss,
                "unique_video_count": loss.new_tensor(2.0),
                "duplicate_sample_count": loss.new_tensor(0.0),
                "mean_positive_count": loss.new_tensor(1.0),
            }

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
        gradient_accumulation_steps=1,
        epochs=1,
        output_dir=str(tmp_path),
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
