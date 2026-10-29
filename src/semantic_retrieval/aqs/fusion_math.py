"""Differentiable FP32 Gaussian matching with quadratic, not cubic, storage."""
from dataclasses import dataclass
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


def symmetric_kl(text,video):
    if not isinstance(text,ScoreGaussian) or not isinstance(video,ScoreGaussian):
        raise ValueError('expected scoring Gaussians')
    for q in (text,video):
        _pair(q.mean,q.logvar)
        if q.mean.dtype!=torch.float32 or q.logvar.dtype!=torch.float32 or (q.logvar < -8).any() or (q.logvar > 4).any():
            raise ValueError('invalid scoring Gaussian precision or bounds')
    if text.mean.shape[1]!=video.mean.shape[1] or text.mean.device!=video.mean.device:
        raise ValueError('Gaussian dimension/device mismatch')
    mt,mv,lt,lv=text.mean,video.mean,text.logvar,video.logvar
    wt,wv=(-lt).exp(),(-lv).exp(); d=mt.shape[1]
    ratio=lt.exp()@wv.T+wt@lv.exp().T-2*d
    a=(mt.square()*wt).sum(1)[:,None]+wt@mv.square().T-2*(mt*wt)@mv.T
    b=(mv.square()*wv).sum(1)[None,:]+mt.square()@wv.T-2*mt@(mv*wv).T
    distance=(ratio+a+b)/(4*d)
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
