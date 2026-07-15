import json
import logging
import math
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiment_tracking import (
    append_batch_protocol_stats,
    atomic_write_json,
    build_experiment_manifest,
    collect_git_state,
    compute_batch_semantics,
    extract_batch_protocol_stats,
    is_global_rank_zero,
)


def test_batch_semantics_distinguishes_all_batch_sizes():
    result = compute_batch_semantics(
        requested_effective_batch=256,
        gradient_accumulation_steps=2,
        world_size=2,
        dataloader_steps=664,
        epochs=5,
    )
    assert result == {
        "requested_effective_batch": 256,
        "forward_global_contrastive_batch": 128,
        "per_rank_micro_batch": 64,
        "gradient_accumulation_steps": 2,
        "optimizer_effective_batch": 256,
        "forward_steps_per_epoch": 664,
        "optimizer_steps_per_epoch": 332,
        "total_optimizer_steps": 1660,
        "world_size": 2,
    }


@pytest.mark.parametrize(
    "values",
    [
        (255, 2, 2, 664, 5),
        (256, 3, 2, 664, 5),
        (256, 2, 3, 664, 5),
        (256, 2, 2, 663, 5),
        (0, 1, 1, 1, 1),
    ],
)
def test_batch_semantics_rejects_invalid_values(values):
    with pytest.raises(ValueError, match="(?:positive|divisible)"):
        compute_batch_semantics(*values)


def _args(tmp_path):
    return SimpleNamespace(
        seed=42,
        experiment_profile="hygiene",
        backbone_type="openai_clip",
        pretrained_clip_name="ViT-B/16",
        clip_layer_norm_precision="fp16",
        clip_gradient_checkpointing=True,
        clip_visual_checkpoint_layers=4,
        backbone_name="",
        backbone_path="",
        output_dir=str(tmp_path),
        train_csv="data/generated/msrvtt_trusted_v1/train.csv",
        val_csv="data/generated/msrvtt_trusted_v1/val.csv",
        test_csv="/data/MSRVTT_JSFUSION_test.csv",
        source_train_csv="/data/MSRVTT_train.9k.csv",
        data_path="/data/MSRVTT_v2.json",
        split_manifest="dataloaders/splits/msrvtt_trusted_v1_seed42.json",
        tqfs_cache_dir="cache_dir/tqfs/test",
        num_thread_reader=8,
        prefetch_factor=4,
        pin_memory=torch.cuda.is_available(),
        use_hard_negative_packing=False,
        use_explicit_hard_negative_loss=False,
        hard_negative_path="cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json",
        hard_negative_pack_seed=42,
        w_hard_negative=0.0,
    )


def test_manifest_contains_protocol_code_data_and_backbone(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,4")
    payload = build_experiment_manifest(
        _args(tmp_path),
        split_summary={"protocol_version": "trusted-v1", "manifest_sha256": "abc"},
        batch_semantics={"world_size": 2},
        git_state={"commit": "deadbeef", "dirty": True, "modified_paths": ["x.py"]},
    )
    assert payload["protocol_version"] == "trusted-v1"
    assert payload["git"]["dirty"] is True
    assert payload["backbone"]["type"] == "openai_clip"
    assert payload["backbone"]["clip_layer_norm_precision"] == "fp16"
    assert payload["backbone"]["clip_gradient_checkpointing"] is True
    assert payload["backbone"]["clip_visual_checkpoint_layers"] == 4
    assert payload["data"]["test_csv"] == "/data/MSRVTT_JSFUSION_test.csv"
    assert payload["data"]["tqfs_cache_dir"] == "cache_dir/tqfs/test"
    assert set(payload) == {
        "protocol_version",
        "git",
        "split",
        "seed",
        "profile",
        "backbone",
        "data",
        "batch",
        "runtime",
        "hard_negative",
        "pair_evidence_refiner",
    }
    assert payload["runtime"] == {
        "cuda_visible_devices": "0,1,2,4",
        "num_dataloader_workers_per_rank": 8,
        "prefetch_factor": 4,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
    }
    assert "final_score_mode" not in payload
    assert "losses" not in payload
    assert payload["hard_negative"] == {
        "packing_enabled": False,
        "explicit_loss_enabled": False,
        "mapping_path": "cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json",
        "pack_seed": 42,
        "loss_weight": 0.0,
    }
    assert payload["pair_evidence_refiner"] == {
        "enabled": False,
        "num_views": 4,
        "lambda_max": 0.1,
        "query_block_size": 16,
        "candidate_block_size": 32,
        "alignment_temperature": 0.07,
    }
    encoded = json.dumps(payload)
    assert "argv" not in encoded
    for retired in (
        "w_mil",
        "w_evidential",
        "w_neg_reg",
        "w_orth",
        "w_uacl_intra",
        "w_uacl_kl",
        "w_query_sim",
        "w_uncertainty_reg",
        "final_score_mode",
    ):
        assert retired not in encoded
    path = tmp_path / "experiment_manifest.json"
    atomic_write_json(path, payload)
    assert json.loads(path.read_text())["split"]["manifest_sha256"] == "abc"


def test_manifest_records_frozen_pair_refiner_configuration(tmp_path):
    args = _args(tmp_path)
    args.experiment_profile = "pair_evidence_refiner"
    payload = build_experiment_manifest(
        args,
        split_summary={"protocol_version": "trusted-v1"},
        batch_semantics={"world_size": 4},
        git_state={"commit": "deadbeef", "dirty": False, "modified_paths": []},
    )

    assert payload["pair_evidence_refiner"] == {
        "enabled": True,
        "num_views": 4,
        "lambda_max": 0.1,
        "query_block_size": 16,
        "candidate_block_size": 32,
        "alignment_temperature": 0.07,
    }


def test_manifest_allows_non_msrvtt_without_trusted_split(tmp_path):
    payload = build_experiment_manifest(
        _args(tmp_path), split_summary=None, batch_semantics={}, git_state={}
    )
    assert payload["protocol_version"] is None
    assert payload["split"] is None


def test_collect_git_state_reports_dirty_paths_without_argv(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    tracked.write_text("changed\n")
    state = collect_git_state(tmp_path)
    assert state["dirty"] is True
    assert "tracked.txt" in state["modified_paths"]
    assert "argv" not in state


def test_atomic_write_overwrites_without_leaving_temp_file(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    atomic_write_json(path, {"version": 1})
    atomic_write_json(path, {"version": 2})
    assert json.loads(path.read_text()) == {"version": 2}
    assert list(path.parent.glob("*.tmp")) == []


def test_batch_stats_header_once_and_append(tmp_path):
    path = tmp_path / "batch_protocol_stats.tsv"
    stats = {"unique_video_count": 2, "duplicate_sample_count": 1, "mean_positive_count": 1.5}
    append_batch_protocol_stats(path, 0, 3, 2, stats)
    append_batch_protocol_stats(path, 0, 4, 2, stats)
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == [
        "epoch", "forward_step", "global_step", "unique_video_count",
        "duplicate_sample_count", "mean_positive_count",
    ]
    assert len(lines) == 3


def test_batch_stats_rejects_extra_telemetry_keys(tmp_path):
    stats = {
        "unique_video_count": 2,
        "duplicate_sample_count": 1,
        "mean_positive_count": 1.5,
        "unexpected": 99,
    }
    with pytest.raises(ValueError, match="(?:extra|schema)"):
        append_batch_protocol_stats(tmp_path / "stats.tsv", 0, 0, 0, stats)


@pytest.mark.parametrize(
    "header",
    [
        "epoch\tglobal_step\tforward_step\tunique_video_count\tduplicate_sample_count\tmean_positive_count",
        "epoch\tforward_step\tglobal_step\tunique_video_count\tduplicate_sample_count",
        "epoch\tforward_step\tglobal_step\tunique_video_count\tduplicate_sample_count\tmean_positive_count\textra",
    ],
)
def test_batch_stats_rejects_corrupt_existing_schema(tmp_path, header):
    path = tmp_path / "stats.tsv"
    path.write_text(header + "\n", encoding="utf-8")
    stats = {"unique_video_count": 2, "duplicate_sample_count": 1, "mean_positive_count": 1.5}
    with pytest.raises(ValueError, match="header"):
        append_batch_protocol_stats(path, 0, 0, 0, stats)
    assert path.read_text(encoding="utf-8") == header + "\n"


def test_sidecar_writer_uses_global_rank_not_local_rank():
    assert is_global_rank_zero(SimpleNamespace(rank=0, local_rank=1)) is True
    assert is_global_rank_zero(SimpleNamespace(rank=1, local_rank=0)) is False
    assert is_global_rank_zero(SimpleNamespace(local_rank=1)) is True


@pytest.mark.parametrize(
    "stats",
    [
        {"unique_video_count": 1, "duplicate_sample_count": 0},
        {"unique_video_count": 1, "duplicate_sample_count": 0, "mean_positive_count": math.nan},
    ],
)
def test_batch_stats_reject_missing_or_nonfinite_values(tmp_path, stats):
    with pytest.raises(ValueError, match="(?:missing|finite)"):
        append_batch_protocol_stats(tmp_path / "stats.tsv", 0, 0, 0, stats)


def test_extract_batch_protocol_stats_rejects_missing_and_nonfinite():
    with pytest.raises(ValueError, match="missing"):
        extract_batch_protocol_stats({"total": 1.0})
    with pytest.raises(ValueError, match="finite"):
        extract_batch_protocol_stats(
            {"unique_video_count": float("inf"), "duplicate_sample_count": 0, "mean_positive_count": 1}
        )


def _run_train_epoch_for_sidecar(tmp_path, telemetry, rank=0, local_rank=1):
    from main_task_retrieval import train_epoch

    class TelemetryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.clip = torch.nn.Module()
            self.clip.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, *args, **kwargs):
            loss = self.weight.square()
            return {"total": loss, "sim_loss": loss, **telemetry}

    model = TelemetryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (
        torch.zeros(2, 1, 2, dtype=torch.long),
        torch.ones(2, 1, 2, dtype=torch.long),
        torch.zeros(2, 1, 2, dtype=torch.long),
        torch.zeros(2, 1, 1, 1, 1, 1, 1),
        torch.ones(2, 1, 1, dtype=torch.long),
        torch.tensor([17, 18]),
    )
    args = SimpleNamespace(
        n_display=1,
        gradient_accumulation_steps=1,
        epochs=1,
        output_dir=str(tmp_path),
        rank=rank,
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
        local_rank=local_rank,
    )


@pytest.mark.parametrize(
    "telemetry",
    [
        {},
        {"unique_video_count": torch.tensor(2.0)},
    ],
)
def test_train_epoch_rejects_missing_batch_protocol_telemetry(tmp_path, telemetry):
    with pytest.raises(ValueError, match="missing required keys"):
        _run_train_epoch_for_sidecar(tmp_path, telemetry)


def test_train_epoch_writes_all_telemetry_fields_to_sidecar(tmp_path):
    _run_train_epoch_for_sidecar(
        tmp_path,
        {
            "unique_video_count": torch.tensor(2.0),
            "duplicate_sample_count": torch.tensor(1.0),
            "mean_positive_count": torch.tensor(1.5),
        },
    )
    lines = (tmp_path / "batch_protocol_stats.tsv").read_text(encoding="utf-8").splitlines()
    assert lines == [
        "epoch\tforward_step\tglobal_step\tunique_video_count\tduplicate_sample_count\tmean_positive_count",
        "0\t0\t0\t2.0\t1.0\t1.5",
    ]


def test_train_epoch_sidecar_uses_global_rank_gate(tmp_path, monkeypatch):
    # ``train_epoch`` logs progress when ``local_rank == 0``.  This test
    # intentionally exercises the multi-node case (global rank 1 on local
    # rank 0), so provide the module logger explicitly instead of relying on
    # training-entrypoint initialisation.
    import main_task_retrieval

    monkeypatch.setattr(
        main_task_retrieval,
        "logger",
        logging.getLogger("test_train_epoch_sidecar_uses_global_rank_gate"),
        raising=False,
    )
    _run_train_epoch_for_sidecar(
        tmp_path,
        {
            "unique_video_count": torch.tensor(2.0),
            "duplicate_sample_count": torch.tensor(1.0),
            "mean_positive_count": torch.tensor(1.5),
        },
        rank=1,
        local_rank=0,
    )
    assert not (tmp_path / "batch_protocol_stats.tsv").exists()
