"""Stateless audited 2D CLIP vision forward exposing projected patch tokens."""
import torch


def visual_tokens(visual, images, *, video_frame):
    if (visual.linear_patch != '2d' or not isinstance(images, torch.Tensor)
            or images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point()
            or not torch.isfinite(images).all() or type(video_frame) is not int
            or not 1 <= video_frame <= 12 or images.shape[0] < 1 or images.shape[0] % video_frame
            or images.shape[-2:] != (visual.input_resolution, visual.input_resolution)
            or visual.positional_embedding.shape[0] != 50):
        raise ValueError('local visual requires valid 2D 7x7 frames')
    x = visual.conv1(images.to(dtype=visual.conv1.weight.dtype))
    if x.shape[-2:] != (7,7): raise ValueError('local visual requires a 7x7 grid')
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],1,x.shape[-1],dtype=x.dtype,device=x.device)
    x = torch.cat([cls,x],dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1,0,2),video_frame=video_frame)
    x = visual.ln_post(x.permute(1,0,2)) @ visual.proj
    return x[:,0,:].float(), x[:,1:,:].float()
