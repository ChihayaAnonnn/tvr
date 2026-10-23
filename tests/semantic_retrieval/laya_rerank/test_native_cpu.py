"""Explicit real-weight CPU check; enable with LAYA_CPU_WEIGHTS in the isolated env."""
import os
import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.skipif(not os.environ.get('LAYA_CPU_WEIGHTS'),reason='explicit offline real-model CPU check')
def test_real_native_batch_cache_and_query_mapping(tmp_path):
    from laya.vlm import build_vlm_inputs,collate_vlm,QTYPES
    from semantic_retrieval.laya_rerank.judge import LayaJudge,QUESTION,CRITERIA,STATE
    from semantic_retrieval.laya_rerank.visual_cache import VisualCache
    from semantic_retrieval.laya_rerank.core import rerank
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    judge=LayaJudge(os.environ['LAYA_CPU_WEIGHTS'],device='cpu')
    images=[Image.fromarray(np.full((32,48,3),i*70,np.uint8)) for i in range(4)]
    cache=VisualCache(tmp_path,{'model':'real-cpu','precision':'fp32'})
    direct=judge.encode(images)
    saved=cache.get({'video':'synthetic-four-frames'},lambda:images,judge)
    np.testing.assert_array_equal(direct['features'],saved['features'])
    queries=['A red car is driving.', 'A person is swimming.']*4
    z8,p8=judge.score(queries,[saved]*8)
    singles=[judge.score([q],[direct]) for q in queries]
    z1=np.concatenate([z for z,p in singles]);p1=np.concatenate([p for z,p in singles])
    assert not np.array_equal(z1[0],z1[1]), 'different queries must retain independent language conditions'
    np.testing.assert_allclose(z8,z1,rtol=0,atol=1e-5)
    np.testing.assert_array_equal(z8.argmax(axis=1),z1.argmax(axis=1))
    np.testing.assert_array_equal(rerank(np.arange(12),np.r_[p1,p1[:2]]),rerank(np.arange(12),np.r_[p8,p8[:2]]))
    permutation=[7,4,1,0,5,2,6,3]
    zr,pr=judge.score([queries[i] for i in permutation],[saved]*8)
    np.testing.assert_allclose(zr,z8[permutation],rtol=0,atol=1e-5)
    # Compare a true uncached pixel forward with the cached-image-feature path.
    a=judge.agent
    q=a._to_internal({'type':'choice','instructions':QUESTION.format(queries[0]),'criteria':CRITERIA})
    row=build_vlm_inputs(a.processor,{**STATE,'images':images},q,a.cfg['max_len'],a.cfg['head_max_len'],option_order=[0,1])
    row['qtype']=QTYPES['choice']
    b=collate_vlm([row],a.processor.tokenizer.pad_token_id,with_pixels=True)
    args=[b[k] for k in ('input_ids','attention_mask','marker_pos','marker_mask','qtype')]
    with torch.inference_mode():
        raw,_=a.model(*args,pixel_values=b['pixel_values'],pixel_attention_mask=b['pixel_attention_mask'],option_span=b['option_span'])
    np.testing.assert_allclose(raw.numpy(),z1[:1],rtol=0,atol=1e-5)
    with pytest.raises(ValueError,match='truncated'):
        judge.score(['description '*2000],[saved])
    assert all(not p.requires_grad for p in a.model.parameters()) and not a.model.training
