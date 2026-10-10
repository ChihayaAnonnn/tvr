import pytest


def test_validation_selection_keeps_earliest_tie_and_rejects_nonfinite():
    from semantic_retrieval.full_data.baseline import choose_epoch
    assert choose_epoch([dict(epoch=1, t2v_r1=25.), dict(epoch=2, t2v_r1=25.), dict(epoch=3, t2v_r1=20.)]) == 1
    assert choose_epoch([dict(epoch=1, t2v_r1=25.), dict(epoch=2, t2v_r1=30.)]) == 2
    with pytest.raises(ValueError): choose_epoch([dict(epoch=1, t2v_r1=float('nan'))])
    with pytest.raises(ValueError): choose_epoch([dict(epoch=1, t2v_r1=10.), dict(epoch=1, t2v_r1=20.)])
