import pytest
import torch
from semantic_retrieval.aqs.local_config import LocalQueryConfig
from semantic_retrieval.aqs.local_tokens import window_indices, lexical_mask, LocalQueryTokens


def test_windows_overlap_and_coverage():
    w = window_indices()
    assert w.shape == (9, 9)
    assert w[0].tolist() == [0,1,2,7,8,9,14,15,16]
    assert set(w.flatten().tolist()) == set(range(49))
    for a,b,n in [(0,1,3),(0,3,3),(0,4,1),(0,2,0)]:
        assert len(set(w[a].tolist()) & set(w[b].tolist())) == n


def test_strict_config():
    cfg = LocalQueryConfig(True, .2, .1)
    assert cfg.active and LocalQueryConfig.from_recipe(cfg.recipe()) == cfg
    for field,value in [('beta',True),('tau',0),('tau',float('nan')),('enabled',1),('beta',-1)]:
        args = dict(enabled=True,beta=.2,tau=.1); args[field]=value
        with pytest.raises(ValueError): LocalQueryConfig(**args)
    for key,value in [('heads',4),('epsilon',True),('unexpected',0)]:
        with pytest.raises(ValueError): LocalQueryConfig.from_recipe(dict(cfg.recipe(), **{key:value}))


def test_lexical_mask():
    mask=torch.tensor([[1,1,1,1,0],[1,1,1,0,0]],dtype=torch.bool)
    assert lexical_mask(mask).tolist()==[[False,True,True,False,False],[False,True,False,False,False]]
    with pytest.raises(ValueError): lexical_mask(torch.tensor([[True,True]]))
    with pytest.raises(ValueError): lexical_mask(torch.tensor([[True,False,True]]))


def test_window_restriction_sharing_padding_and_gradients():
    torch.manual_seed(3)
    net=LocalQueryTokens(64)
    p=torch.randn(1,3,49,64,requires_grad=True)
    mask=torch.tensor([[True,True,False]])
    out=net(p,mask)
    altered=p.detach().clone(); altered[:,:,48]+=10
    torch.testing.assert_close(net(altered,mask)[:,:,0],out[:,:,0])
    copied=p.detach()[:,0:1].expand(-1,3,-1,-1).clone()
    result=net(copied,mask)
    torch.testing.assert_close(result[:,0],result[:,1])
    assert torch.count_nonzero(out[:,2])==0
    out[0,0,0,0].backward()
    for param in net.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0
    assert p.grad[0,0,48].abs().sum()==0
    assert p.grad[0,0,16].abs().sum()>0
    assert 'indices' not in net.state_dict()
