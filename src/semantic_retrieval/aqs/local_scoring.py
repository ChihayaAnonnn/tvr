"""Text-conditioned reading of independently encoded local video slots."""
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _block(t, words, o, valid, tau):
    c = torch.einsum('qld,vud->qvlu',t,o)
    # Shift before division to keep the maximum exactly zero even for small tau.
    scaled = c.masked_fill(~valid[None,:,None,:], -torch.inf)
    scaled = (scaled-scaled.amax(-1,keepdim=True))/tau
    weights = scaled.softmax(-1)
    per_word = (weights*c).sum(-1)
    return (per_word*words[:,None,:]).sum(-1)/words.sum(-1)[:,None]


def local_scores(text_tokens, word_mask, local_tokens, frame_mask, *, tau, query_chunk, video_chunk):
    t,o,wm,fm = text_tokens,local_tokens,word_mask,frame_mask
    if (any(not isinstance(x,torch.Tensor) for x in (t,o,wm,fm))
            or t.ndim != 3 or o.ndim != 4 or o.shape[2] != 9 or t.shape[-1] != o.shape[-1]
            or min(t.shape) < 1 or min(o.shape) < 1
            or wm.dtype != torch.bool or fm.dtype != torch.bool
            or wm.shape != t.shape[:2] or fm.shape != o.shape[:2]
            or any(x.device != t.device for x in (o,wm,fm))
            or not t.is_floating_point() or not o.is_floating_point()
            or not torch.isfinite(t).all() or not torch.isfinite(o).all()
            or not wm.any(-1).all() or not fm.any(-1).all()
            or type(tau) not in (int,float) or not math.isfinite(tau) or tau < 1e-38
            or any(type(n) is not int or n < 1 for n in (query_chunk,video_chunk))):
        raise ValueError('invalid local matching tensors or configuration')
    t = F.normalize(t.float().masked_fill(~wm[...,None],0),dim=-1,eps=1e-6)
    o = F.normalize(o.float().masked_fill(~fm[...,None,None],0),dim=-1,eps=1e-6).flatten(1,2)
    valid = fm.repeat_interleave(9,dim=1)
    rows=[]
    for i in range(0,len(t),query_chunk):
        cols=[]
        for j in range(0,len(o),video_chunk):
            args=(t[i:i+query_chunk],wm[i:i+query_chunk],o[j:j+video_chunk],valid[j:j+video_chunk],tau)
            if torch.is_grad_enabled() and (t.requires_grad or o.requires_grad):
                value=checkpoint(_block,*args,use_reentrant=False)
            else:
                value=_block(*args)
            cols.append(value)
        rows.append(torch.cat(cols,dim=1))
    result=torch.cat(rows,dim=0)
    if not torch.isfinite(result).all(): raise ValueError('nonfinite local scores')
    return result


def add_local_logits(global_logits, local_raw, logit_scale, *, beta):
    if (global_logits.shape != local_raw.shape or global_logits.device != local_raw.device
            or type(beta) not in (int,float) or not math.isfinite(beta) or beta < 0
            or logit_scale.numel() != 1 or not torch.isfinite(logit_scale).all()):
        raise ValueError('invalid local score fusion')
    result=global_logits+logit_scale.exp()*beta*local_raw
    if not torch.isfinite(result).all(): raise ValueError('nonfinite fused logits')
    return result
