import pytest
import torch
from torch.distributions import Independent, Normal, kl_divergence


def test_config_strict_recipe():
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    config = FusionConfig(True, .1)
    assert config.active
    assert FusionConfig.from_recipe(config.recipe()) == config
    assert not FusionConfig(True, 0).active
    for args in [(1,.1), (True,True), (True,-1), (True,float('nan')), (True,float('inf'))]:
        with pytest.raises(ValueError): FusionConfig(*args)
    for key,value in [('extra',1),('logvar_min',-9),('schema','uatvr-aqs-v1')]:
        with pytest.raises(ValueError): FusionConfig.from_recipe(dict(config.recipe(), **{key:value}))


def test_oracle_and_gradients():
    from semantic_retrieval.aqs.fusion_math import score_gaussian, symmetric_kl
    generator = torch.Generator().manual_seed(713)
    mt = torch.randn(3,5,generator=generator,requires_grad=True)
    mv = torch.randn(4,5,generator=generator,requires_grad=True)
    at = torch.randn(3,5,generator=generator,requires_grad=True)
    av = torch.randn(4,5,generator=generator,requires_grad=True)
    qt = Independent(Normal(mt.double()[:,None],at.double()[:,None].exp()),1)
    qv = Independent(Normal(mv.double()[None],av.double()[None].exp()),1)
    expected = (kl_divergence(qt,qv)+kl_divergence(qv,qt))/10
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.shape) or t, lambda t:t):
        result=symmetric_kl(score_gaussian(mt,at),score_gaussian(mv,av))
    torch.testing.assert_close(result.double(),expected,atol=1e-5,rtol=2e-4)
    assert all(len(shape)<=2 for shape in saved)
    grads=torch.autograd.grad(result.sum(),(mt,mv,at,av))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)


def test_identity_symmetry_bounds_and_variance_changes_ranking():
    from semantic_retrieval.aqs.fusion_math import score_gaussian,symmetric_kl,fuse_scores
    from semantic_retrieval.aqs.fusion_config import FusionConfig
    mu=torch.zeros(2,3)
    q=score_gaussian(mu,torch.tensor([[-100.,100.,0.],[-100.,100.,0.]]))
    assert q.logvar.tolist()==[[-8.,4.,0.],[-8.,4.,0.]]
    torch.testing.assert_close(symmetric_kl(q,q),torch.zeros(2,2),atol=1e-5,rtol=0)
    text=score_gaussian(torch.zeros(1,3),torch.zeros(1,3))
    video=score_gaussian(mu,torch.tensor([[0.,0.,0.],[1.,1.,1.]]))
    d=symmetric_kl(text,video)
    torch.testing.assert_close(d,symmetric_kl(video,text).T)
    raw,logits=fuse_scores(torch.ones(1,2),d,torch.tensor(0.),FusionConfig(True,.1))
    assert raw.argmax(1).item()==0
    assert torch.equal(raw,logits)
    swapped=score_gaussian(mu,torch.tensor([[1.,1.,1.],[0.,0.,0.]]))
    assert fuse_scores(torch.ones(1,2),symmetric_kl(text,swapped),torch.tensor(0.),FusionConfig(True,.1))[0].argmax(1).item()==1


@pytest.mark.parametrize('defect',['nan','int','shape','empty'])
def test_invalid_distribution(defect):
    from semantic_retrieval.aqs.fusion_math import score_gaussian
    mean=torch.ones(2,3); a=mean.clone()
    if defect=='nan': a[0,0]=float('nan')
    if defect=='int': a=a.long()
    if defect=='shape': a=a[:1]
    if defect=='empty': mean,a=mean[:0],a[:0]
    with pytest.raises(ValueError): score_gaussian(mean,a)


def test_scoring_remains_fp32_inside_autocast():
    from semantic_retrieval.aqs.fusion_math import score_gaussian,symmetric_kl
    generator=torch.Generator().manual_seed(713)
    a=score_gaussian(torch.randn(3,16,generator=generator),torch.randn(3,16,generator=generator))
    b=score_gaussian(torch.randn(4,16,generator=generator),torch.randn(4,16,generator=generator))
    expected=symmetric_kl(a,b)
    with torch.autocast('cpu',dtype=torch.bfloat16): actual=symmetric_kl(a,b)
    assert actual.dtype==torch.float32
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.parametrize('dimension',[2,3,4,8,64,512])
def test_equal_small_variance_is_stable_and_has_zero_gradient(dimension):
    from semantic_retrieval.aqs.fusion_math import ScoreGaussian,symmetric_kl
    torch.manual_seed(0)
    mean=torch.nn.functional.normalize(torch.randn(64,dimension),dim=1).requires_grad_()
    logvar=torch.full_like(mean,-8.,requires_grad=True)
    q=ScoreGaussian(mean,logvar)
    scores=symmetric_kl(q,q)
    torch.testing.assert_close(scores.diag(),torch.zeros(64),atol=1e-5,rtol=0)
    grads=torch.autograd.grad(scores.diag().sum(),(mean,logvar))
    for g in grads: torch.testing.assert_close(g,torch.zeros_like(g),atol=1e-5,rtol=0)


def test_near_zero_gradient_matches_independent_oracle_without_pair_storage():
    from semantic_retrieval.aqs.fusion_math import ScoreGaussian,symmetric_kl
    mt=torch.tensor([[.1,.2]],requires_grad=True)
    mv=torch.tensor([[.1001,.2001],[.1002,.1999]],requires_grad=True)
    lt=torch.tensor([[.2,-.3]],requires_grad=True)
    lv=torch.tensor([[.2001,-.3001],[.1999,-.3002]],requires_grad=True)
    qt=Independent(Normal(mt.double()[:,None],(lt.double()[:,None]/2).exp()),1)
    qv=Independent(Normal(mv.double()[None],(lv.double()[None]/2).exp()),1)
    oracle=(kl_divergence(qt,qv)+kl_divergence(qv,qt))/4
    shapes=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:shapes.append(t.shape) or t,lambda t:t):
        actual=symmetric_kl(ScoreGaussian(mt,lt),ScoreGaussian(mv,lv))
    assert all(len(shape)<=2 for shape in shapes)
    torch.testing.assert_close(actual.double(),oracle,atol=1e-10,rtol=2e-4)
    expected=torch.autograd.grad(oracle.sum(),(mt,mv,lt,lv))
    observed=torch.autograd.grad(actual.sum(),(mt,mv,lt,lv))
    for a,b in zip(observed,expected):torch.testing.assert_close(a,b,atol=1e-7,rtol=2e-4)
