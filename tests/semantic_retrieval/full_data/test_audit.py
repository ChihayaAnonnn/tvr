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


def test_input_binding_uses_file_bytes_not_array_content_digest(tmp_path):
    from scripts.audit_full_data_implementation import Inputs, digest
    from experiment_artifacts import ArtifactArrayStore
    ref = ArtifactArrayStore(tmp_path/'arrays').write('test', np.arange(5)).as_dict()
    path = tmp_path/ref['path']
    # Store paths may be rooted at arrays/; resolve the published path explicitly.
    if not path.exists(): path=tmp_path/'arrays'/ref['path']
    assert digest(path) != ref['sha256']
    inputs=Inputs.__new__(Inputs);inputs.verified=set();inputs.references={str(path.resolve()):ref['sha256']}
    with pytest.raises(ValueError,match='changed'):inputs.check(path)
    inputs.references[str(path.resolve())]=digest(path)
    assert inputs.check(path)==path.resolve()
