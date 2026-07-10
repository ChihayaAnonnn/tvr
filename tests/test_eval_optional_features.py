import pytest
import torch

from main_task_retrieval import _cat_optional_tensors, _to_cpu_optional


def test_optional_feature_helpers_preserve_none():
    assert _cat_optional_tensors([None, None]) is None
    assert _to_cpu_optional(None) is None


def test_optional_feature_helpers_concatenate_tensors():
    result = _cat_optional_tensors([torch.ones(1, 2), torch.zeros(2, 2)])
    assert result.shape == (3, 2)


def test_optional_feature_helpers_reject_mixed_values():
    with pytest.raises(ValueError, match="mixed optional tensor"):
        _cat_optional_tensors([torch.ones(1), None])
