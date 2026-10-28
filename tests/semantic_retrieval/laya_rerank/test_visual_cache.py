import numpy as np
import torch
from semantic_retrieval.laya_rerank.visual_cache import VisualCache


def test_visual_cache_roundtrip(tmp_path):
    class Judge:
        def __init__(self): self.calls=0
        def encode(self,images):
            self.calls+=1
            return {'ids':[1,2,3], 'features':torch.ones((len(images),2,3),dtype=torch.float32)}
    j=Judge(); record={'sha256':'video-sha','indices':[1,3]}
    cache=VisualCache(tmp_path,{'weights':'sha','processor':'sha','precision':'fp32'})
    a=cache.get(record,lambda:[1,2],j)
    b=cache.get(record,lambda:(_ for _ in ()).throw(AssertionError()),j)
    assert j.calls==1 and a['ids']==b['ids']
    np.testing.assert_array_equal(a['features'],b['features'])
