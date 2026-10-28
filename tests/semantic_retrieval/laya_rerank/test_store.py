import numpy as np
import pytest
from semantic_retrieval.laya_rerank.store import ArrayCache


def test_restore_and_identity(tmp_path):
    c = ArrayCache(tmp_path, {'weights': 'a', 'precision': 'fp32'})
    key = {'query': 'one', 'video': 'v'}
    c.write(key, np.array([1., 2.], dtype=np.float32))
    np.testing.assert_array_equal(c.read(key), [1, 2])
    assert c.read({'query': 'two', 'video': 'v'}) is None
    np.testing.assert_array_equal(ArrayCache(tmp_path, c.identity).read(key), [1, 2])
    with pytest.raises(ValueError): ArrayCache(tmp_path, {'weights': 'b'})
    with pytest.raises(ValueError): c.write(key, np.array([2., 3.], dtype=np.float32))
    next(tmp_path.glob('*.npy')).write_bytes(b'corrupt')
    with pytest.raises(ValueError): c.read(key)
