from types import SimpleNamespace
import sys
import types

import torch

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
spatial_enhancer_stub = types.ModuleType("modules.spatial_enhancer")
spatial_enhancer_stub.SpatialEnhancer = object
sys.modules["modules.spatial_enhancer"] = spatial_enhancer_stub

from main_task_retrieval import _unpack_train_batch, get_args


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
        ],
    )

    args = get_args()

    assert args.use_explicit_hard_negative_loss is True
    assert args.w_hard_negative == 0.07
    assert args.use_uacl_intra_alignment is True
    assert args.w_uacl_intra == 0.02
    assert args.w_uacl_kl == 0.0003
    assert args.uacl_temperature == 0.05


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
