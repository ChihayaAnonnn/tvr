"""Differentiable FP32 Gaussian matching with quadratic, not cubic, storage."""
from dataclasses import dataclass
from torch.autograd.function import once_differentiable
import torch
from .fusion_config import FusionConfig


@dataclass(frozen=True)
class ScoreGaussian:
    mean: torch.Tensor
    logvar: torch.Tensor


def _pair(mean,scale):
    if (not isinstance(mean,torch.Tensor) or not isinstance(scale,torch.Tensor)
            or mean.ndim!=2 or mean.shape!=scale.shape or min(mean.shape)<1
            or not mean.is_floating_point() or not scale.is_floating_point()
            or mean.device!=scale.device or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()):
        raise ValueError('invalid Gaussian parameters')


def score_gaussian(mean,logsigma):
    _pair(mean,logsigma)
    result=ScoreGaussian(mean.float(),2*logsigma.float().clamp(-4,2))
    _pair(result.mean,result.logvar)
    return result



class _StableSelectedKL(torch.autograd.Function):
    """Recompute near-equal pairs directly; save parameters/indices, not pair features.

    The ordinary matrix path is fast but loses precision near zero. The selected
    path uses bounded 2D scratch and an analytic first-order backward. Higher-order
    derivatives are intentionally unsupported; training uses first-order gradients.
    """
    @staticmethod
    def forward(ctx,mt,mv,lt,lv,rows,cols):
        ctx.save_for_backward(mt,mv,lt,lv,rows,cols)
        chunk=min(1024,len(mt)+len(mv))
        result=mt.new_empty(len(rows))
        with torch.autocast(device_type=mt.device.type,enabled=False):
            for start in range(0,len(rows),chunk):
                r,c=rows[start:start+chunk],cols[start:start+chunk]
                delta=lt[r]-lv[c];diff=mt[r]-mv[c]
                ratio=torch.expm1(delta)+torch.expm1(-delta)
                result[start:start+chunk]=(ratio+diff.square()*((-lt[r]).exp()+(-lv[c]).exp())).sum(1)/(4*mt.shape[1])
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        mt,mv,lt,lv,rows,cols=ctx.saved_tensors
        gm_t,gm_v,gl_t,gl_v=[torch.zeros_like(x) for x in (mt,mv,lt,lv)]
        chunk=min(1024,len(mt)+len(mv));d=mt.shape[1]
        # Grouped sums avoid CUDA atomic scatter accumulation and preserve replay.
        def accumulate(target,indices,values):
            for i in indices.unique(sorted=True):
                target[i]+=values[indices==i].sum(0)
        with torch.autocast(device_type=mt.device.type,enabled=False):
            for start in range(0,len(rows),chunk):
                r,c=rows[start:start+chunk],cols[start:start+chunk]
                delta=lt[r]-lv[c];diff=mt[r]-mv[c]
                wt,wv=(-lt[r]).exp(),(-lv[c]).exp()
                g=grad[start:start+chunk,None]/(4*d)
                mean_grad=g*2*diff*(wt+wv)
                ratio_grad=torch.expm1(delta)-torch.expm1(-delta)
                accumulate(gm_t,r,mean_grad);accumulate(gm_v,c,-mean_grad)
                accumulate(gl_t,r,g*(ratio_grad-diff.square()*wt))
                accumulate(gl_v,c,g*(-ratio_grad-diff.square()*wv))
        return gm_t,gm_v,gl_t,gl_v,None,None


def symmetric_kl(text,video):
    if not isinstance(text,ScoreGaussian) or not isinstance(video,ScoreGaussian):
        raise ValueError('expected scoring Gaussians')
    for q in (text,video):
        _pair(q.mean,q.logvar)
        if q.mean.dtype!=torch.float32 or q.logvar.dtype!=torch.float32 or (q.logvar < -8).any() or (q.logvar > 4).any():
            raise ValueError('invalid scoring Gaussian precision or bounds')
    if text.mean.shape[1]!=video.mean.shape[1] or text.mean.device!=video.mean.device:
        raise ValueError('Gaussian dimension/device mismatch')
    with torch.autocast(device_type=text.mean.device.type,enabled=False):
        mt,mv,lt,lv=text.mean,video.mean,text.logvar,video.logvar
        wt,wv=(-lt).exp(),(-lv).exp(); d=mt.shape[1]
        ratios=lt.exp()@wv.T+wt@lv.exp().T
        t0=(mt.square()*wt).sum(1)[:,None];t1=wt@mv.square().T
        v0=(mv.square()*wv).sum(1)[None,:];v1=mt.square()@wv.T
        a=t0+t1-2*(mt*wt)@mv.T
        b=v0+v1-2*mt@(mv*wv).T
        distance=(ratios-2*d+a+b)/(4*d)
        # A conservative FP32 dot/reduction error bound; 2|xy| <= x²+y².
        # It must catch positive cancellation too, not merely negative values.
        eps=torch.finfo(torch.float32).eps
        gamma=(d+4)*eps/(1-(d+4)*eps)
        scale=ratios.detach()+2*d+2*(t0.detach()+t1.detach()+v0.detach()+v1.detach())
        threshold=(4*gamma*scale/(4*d)).clamp_min(1e-4)
        near=(distance.detach()<threshold).nonzero(as_tuple=False)
        if len(near):
            rows,cols=near[:,0],near[:,1]
            stable=_StableSelectedKL.apply(mt,mv,lt,lv,rows,cols)
            distance=distance.index_put((rows,cols),stable)
        if not torch.isfinite(distance).all() or (distance < -1e-5).any():
            raise ValueError('invalid symmetric Gaussian KL')
        return distance.clamp_min(0)



def fuse_scores(wti,distance,logit_scale,config):
    if (not isinstance(config,FusionConfig) or not config.active
            or not isinstance(wti,torch.Tensor) or not isinstance(distance,torch.Tensor)
            or wti.ndim!=2 or wti.shape!=distance.shape or wti.device!=distance.device
            or not wti.is_floating_point() or not distance.is_floating_point()
            or not torch.isfinite(wti).all() or not torch.isfinite(distance).all() or (distance<0).any()
            or not isinstance(logit_scale,torch.Tensor) or logit_scale.numel()!=1
            or logit_scale.device!=wti.device or not torch.isfinite(logit_scale).all()):
        raise ValueError('invalid fusion scores')
    raw=wti.float()-config.eta*distance.float()
    logits=logit_scale.float().exp()*raw
    if not torch.isfinite(raw).all() or not torch.isfinite(logits).all():
        raise ValueError('nonfinite fused logits')
    return raw,logits
