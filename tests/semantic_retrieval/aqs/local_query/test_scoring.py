import pytest
import torch
import torch.nn.functional as F
from semantic_retrieval.aqs.local_scoring import local_scores, add_local_logits


def oracle(t,wm,o,fm,tau):
    rows=[]
    for i in range(len(t)):
        row=[]
        for j in range(len(o)):
            slots=F.normalize(o[j,fm[j]].flatten(0,1).double(),dim=-1,eps=1e-6)
            words=F.normalize(t[i,wm[i]].double(),dim=-1,eps=1e-6)
            per_word=[]
            for word in words:
                c=slots@word
                per_word.append(((c/tau).softmax(0)*c).sum())
            row.append(torch.stack(per_word).mean())
        rows.append(torch.stack(row))
    return torch.stack(rows)


@pytest.mark.parametrize('chunks',[(1,1),(2,2),(5,5)])
def test_oracle_chunk_values_and_gradients(chunks):
    torch.manual_seed(8)
    t=torch.randn(2,4,16,requires_grad=True)
    o=torch.randn(3,2,9,16,requires_grad=True)
    wm=torch.tensor([[0,1,1,0],[0,1,0,0]],dtype=torch.bool)
    fm=torch.tensor([[1,0],[1,1],[0,1]],dtype=torch.bool)
    got=local_scores(t,wm,o,fm,tau=.1,query_chunk=chunks[0],video_chunk=chunks[1])
    expected=oracle(t,wm,o,fm,.1)
    torch.testing.assert_close(got.double(),expected,atol=1e-6,rtol=1e-5)
    grads=torch.autograd.grad(got.sum(),(t,o))
    refs=torch.autograd.grad(expected.sum(),(t,o))
    for a,b in zip(grads,refs): torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    changed=t.detach().clone(); changed[~wm]=10000
    changed_o=o.detach().clone(); changed_o[~fm]=10000
    torch.testing.assert_close(got,local_scores(changed,wm,changed_o,fm,tau=.1,query_chunk=1,video_chunk=1))
    assert (grads[0][~wm]==0).all() and (grads[1][~fm]==0).all()


def test_scale_once_and_saved_tensors():
    t=torch.randn(2,4,16,requires_grad=True); o=torch.randn(3,2,9,16,requires_grad=True)
    wm=torch.ones(2,4,dtype=torch.bool); fm=torch.ones(3,2,dtype=torch.bool)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(tuple(x.shape)) or x), lambda x:x):
        raw=local_scores(t,wm,o,fm,tau=.2,query_chunk=1,video_chunk=1)
    assert (2,3,4,18) not in saved and (1,1,4,18) not in saved
    scale=torch.tensor(.7,requires_grad=True); global_logits=torch.randn(2,3,requires_grad=True)
    result=add_local_logits(global_logits,raw,scale,beta=.2)
    expected=global_logits+scale.exp()*.2*raw
    torch.testing.assert_close(result,expected)
    result.sum().backward()
    torch.testing.assert_close(scale.grad,(scale.exp()*.2*raw).sum())
    with pytest.raises(ValueError): local_scores(t,wm&False,o,fm,tau=.2,query_chunk=1,video_chunk=1)
    with pytest.raises(ValueError): local_scores(t,wm,o,fm,tau=0,query_chunk=1,video_chunk=1)


def test_padding_extension_negative_scores_and_temperature():
    t=torch.tensor([[[1.,0.]]]); o=torch.tensor([[[[-1.,.1]]*9]])
    wm=torch.ones(1,1,dtype=torch.bool); fm=wm.clone()
    a=local_scores(t,wm,o,fm,tau=.1,query_chunk=1,video_chunk=1)
    assert a.item()<0
    tp=torch.cat([t,torch.randn_like(t)],1); op=torch.cat([o,torch.randn_like(o)],1)
    padded=torch.tensor([[True,False]])
    torch.testing.assert_close(a,local_scores(tp,padded,op,padded,tau=.1,query_chunk=1,video_chunk=1))
    varied=torch.randn(1,2,9,2)
    low=local_scores(t,wm,varied,torch.ones(1,2,dtype=torch.bool),tau=.01,query_chunk=1,video_chunk=1)
    high=local_scores(t,wm,varied,torch.ones(1,2,dtype=torch.bool),tau=1.,query_chunk=1,video_chunk=1)
    assert low.item()>high.item()
