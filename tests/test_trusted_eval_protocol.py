import importlib.machinery
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

cv2_stub = types.ModuleType("cv2")
cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
sys.modules.setdefault("cv2", cv2_stub)
spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules.setdefault("modules.spatial_enhancer", spatial_enhancer_stub)

import main_task_retrieval as retrieval  # noqa: E402


def _args(**overrides):
    values = {
        "datatype": "msrvtt",
        "do_train": True,
        "do_eval": False,
        "eval_split": "val",
        "init_model": None,
        "expand_msrvtt_sentences": True,
        "experiment_profile": "hygiene",
        "use_hard_negative_packing": False,
        "use_explicit_hard_negative_loss": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_rejects_test_split():
    with pytest.raises(ValueError, match="training cannot use eval_split=test"):
        retrieval.validate_trusted_cli(_args(eval_split="test"))


def test_training_requires_expanded_captions():
    with pytest.raises(ValueError, match="requires --expand_msrvtt_sentences"):
        retrieval.validate_trusted_cli(_args(expand_msrvtt_sentences=False))


def test_eval_only_requires_initial_checkpoint():
    with pytest.raises(ValueError, match="--do_eval requires --init_model"):
        retrieval.validate_trusted_cli(
            _args(do_train=False, do_eval=True, eval_split="test")
        )


@pytest.mark.parametrize(
    "flag",
    ["use_hard_negative_packing", "use_explicit_hard_negative_loss"],
)
def test_hygiene_rejects_hard_negative_diagnostics(flag):
    with pytest.raises(ValueError, match="hard-negative diagnostic"):
        retrieval.validate_trusted_cli(_args(**{flag: True}))


def test_default_profile_allows_explicit_hard_negative_diagnostic():
    retrieval.validate_trusted_cli(
        _args(
            experiment_profile="default",
            use_explicit_hard_negative_loss=True,
        )
    )


def test_get_args_exposes_trusted_data_paths_and_eval_split(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_eval",
            "--init_model",
            "checkpoint.bin",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--eval_split",
            "test",
            "--source_train_csv",
            "source-train.csv",
            "--test_csv",
            "blind-test.csv",
            "--split_manifest",
            "trusted.json",
        ],
    )

    args = retrieval.get_args()

    assert args.eval_split == "test"
    assert args.source_train_csv == "source-train.csv"
    assert args.test_csv == "blind-test.csv"
    assert args.split_manifest == "trusted.json"


def _run_with_fake_torchrun(script_name, tmp_path, xattn_value):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    python3_path = fake_bin / "python3"
    python3_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    python3_path.chmod(0o755)
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
            "CLIP_LAYER_NORM_PRECISION": "fp16",
            "RUN_ID": "supported-args-test",
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


@pytest.mark.parametrize(
    "script_name", ["train_msrvtt.sh", "eval.sh", "train_msvd.sh"]
)
def test_supported_scripts_emit_only_supported_retrieval_args(script_name, tmp_path):
    retired_flags = {flag for flag, _ in RETIRED_CLI_CASES}
    result, capture_path = _run_with_fake_torchrun(script_name, tmp_path, "0")
    assert result.returncode == 0, result.stderr
    captured = capture_path.read_text(encoding="utf-8").splitlines()
    assert "--experiment_profile" in captured
    profile_index = captured.index("--experiment_profile")
    assert captured[profile_index + 1] == "hygiene"
    if script_name == "eval.sh":
        split_index = captured.index("--eval_split")
        assert captured[split_index + 1] == "val"
    emitted_flags = {
        token.split("=", 1)[0]
        for token in captured
        if token.startswith("--")
    }
    assert retired_flags.isdisjoint(emitted_flags)
    if script_name == "eval.sh":
        assert "--log_mus_scores" in emitted_flags
    else:
        assert "--log_mus_scores" not in emitted_flags


def test_training_constructs_val_but_not_test(monkeypatch):
    calls = []
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {
            "train": lambda args, tokenizer: (
                calls.append("train") or ("train", 1, "sampler")
            ),
            "val": lambda args, tokenizer, subset="val": (
                calls.append("val") or ("val", 2)
            ),
            "test": lambda args, tokenizer, subset="test": (
                calls.append("test") or ("test", 3)
            ),
        },
    )

    result = retrieval.prepare_requested_dataloaders(_args(), object())

    assert result == (("train", 1, "sampler"), "val", 2, "val")
    assert calls == ["train", "val"]


@pytest.mark.parametrize("eval_split", ["val", "test"])
def test_eval_only_constructs_only_requested_split(monkeypatch, eval_split):
    calls = []
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {
            "train": lambda args, tokenizer: calls.append("train"),
            "val": lambda args, tokenizer, subset="val": (
                calls.append("val") or ("val", 2)
            ),
            "test": lambda args, tokenizer, subset="test": (
                calls.append("test") or ("test", 3)
            ),
        },
    )

    result = retrieval.prepare_requested_dataloaders(
        _args(
            do_train=False,
            do_eval=True,
            eval_split=eval_split,
            init_model="checkpoint.bin",
        ),
        object(),
    )

    expected_length = 2 if eval_split == "val" else 3
    assert result == (None, eval_split, expected_length, eval_split)
    assert calls == [eval_split]


def test_msrvtt_training_rejects_missing_val_loader(monkeypatch):
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {"train": lambda args, tokenizer: ("train", 1, "sampler"), "test": None},
    )

    with pytest.raises(ValueError, match="msrvtt has no val dataloader"):
        retrieval.prepare_requested_dataloaders(_args(), object())


def test_eval_only_rejects_missing_requested_loader(monkeypatch):
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "msrvtt",
        {"train": object(), "val": object(), "test": None},
    )

    with pytest.raises(ValueError, match="msrvtt has no test dataloader"):
        retrieval.prepare_requested_dataloaders(
            _args(
                do_train=False,
                do_eval=True,
                eval_split="test",
                init_model="checkpoint.bin",
            ),
            object(),
        )


def test_non_msrvtt_training_retains_test_fallback_when_val_is_missing(monkeypatch):
    calls = []
    monkeypatch.setitem(
        retrieval.DATALOADER_DICT,
        "legacy_dataset",
        {
            "train": lambda args, tokenizer: (
                calls.append("train") or ("train", 1, "sampler")
            ),
            "val": None,
            "test": lambda args, tokenizer, subset="test": (
                calls.append("test") or ("test", 3)
            ),
        },
    )

    result = retrieval.prepare_requested_dataloaders(
        _args(datatype="legacy_dataset", experiment_profile="default"), object()
    )

    assert result == (("train", 1, "sampler"), "test", 3, "test")
    assert calls == ["train", "test"]


def test_checkpoint_selection_prefers_higher_internal_val_score():
    assert retrieval.select_best_checkpoint(
        48.0, "epoch1.bin", 48.5, "epoch2.bin"
    ) == (48.5, "epoch2.bin")
    assert retrieval.select_best_checkpoint(
        48.5, "epoch2.bin", 48.1, "epoch3.bin"
    ) == (48.5, "epoch2.bin")


def test_checkpoint_selection_tie_prefers_later_checkpoint():
    assert retrieval.select_best_checkpoint(
        48.5, "epoch2.bin", 48.5, "epoch3.bin"
    ) == (48.5, "epoch3.bin")


def test_training_evaluation_errors_propagate(monkeypatch):
    def fail_eval(*args, **kwargs):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(retrieval, "eval_epoch", fail_eval)

    with pytest.raises(RuntimeError, match="validation failed"):
        retrieval.evaluate_training_checkpoint(
            args=object(),
            model=object(),
            eval_dataloader=object(),
            device=object(),
            n_gpu=1,
            best_score=48.0,
            best_path="epoch1.bin",
            candidate_path="epoch2.bin",
        )


def test_multi_sentence_eval_selects_one_video_row_per_caption_group():
    cut_off_points = [20, 40]
    assert retrieval.select_multi_sentence_video_rows(0, 25, cut_off_points) == [19]
    assert retrieval.select_multi_sentence_video_rows(25, 15, cut_off_points) == [14]


def test_multi_sentence_eval_handles_group_ends_on_batch_boundaries():
    cut_off_points = [20, 40, 60]
    assert retrieval.select_multi_sentence_video_rows(0, 20, cut_off_points) == [19]
    assert retrieval.select_multi_sentence_video_rows(20, 20, cut_off_points) == [19]
    assert retrieval.select_multi_sentence_video_rows(40, 20, cut_off_points) == [19]
