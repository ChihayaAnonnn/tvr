from types import SimpleNamespace
import importlib.machinery
import sys
import types

import torch

cv2_stub = types.ModuleType("cv2")
cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)
sys.modules.setdefault("cv2", cv2_stub)
spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules["modules.spatial_enhancer"] = spatial_enhancer_stub

from main_task_retrieval import _trainable_named_parameters, _unpack_train_batch, get_args


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


def test_get_args_normalizes_hygiene_profile_to_clean_wti_only(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--experiment_profile",
            "hygiene",
            "--uncertainty_mode",
            "evidential",
            "--w_mil",
            "0.01",
            "--w_evidential",
            "0.01",
            "--w_neg_reg",
            "0.05",
            "--w_orth",
            "0.1",
            "--final_score_mode",
            "wti_anchor_wti",
            "--lambda_anchor",
            "0.2",
            "--use_explicit_hard_negative_loss",
            "--use_uacl_intra_alignment",
        ],
    )

    args = get_args()

    assert args.experiment_profile == "hygiene"
    assert args.uncertainty_mode == "none"
    assert args.w_mil == 0
    assert args.w_evidential == 0
    assert args.w_neg_reg == 0
    assert args.w_orth == 0
    assert args.final_score_mode == "wti_anchor_wti"
    assert args.lambda_anchor == 0.2
    assert args.use_explicit_hard_negative_loss is False
    assert args.use_uacl_intra_alignment is False


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
    batch = tuple(tensor_id(i) for i in range(9))

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


def test_unpack_train_batch_supports_explicit_hard_negative_with_attributes():
    args = SimpleNamespace(use_attributes=True, use_explicit_hard_negative_loss=True)
    batch = tuple(tensor_id(i) for i in range(12))

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
