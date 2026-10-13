import numpy as np
import pytest


def test_comparison_reports_first_real_difference_and_rejects_nonfinite():
    from scripts.audit_full_data_implementation import compare_arrays
    assert compare_arrays(np.ones(3), np.ones(3))['passed']
    assert not compare_arrays(np.ones(3), np.zeros(3))['passed']
    with pytest.raises(ValueError, match='shape|finite'):
        compare_arrays(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match='finite'):
        compare_arrays(np.array([np.nan]), np.array([np.nan]))


def test_loss_summary_preserves_components_and_rejects_missing_updates():
    from scripts.audit_full_data_implementation import loss_summary
    losses = np.array([[3., 2., .7, .3], [2., 1., .8, .2]], dtype=np.float32)
    row = loss_summary(losses, 2)
    assert row['mean']['loss'] == 2.5
    assert row['mean']['contrastive'] == 1.5
    assert row['component_sum_max_error'] < 1e-6
    with pytest.raises(ValueError, match='updates'):
        loss_summary(losses, 3)
