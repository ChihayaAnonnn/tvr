"""Hash-pinned public weight preparation, independent of scientific data gates."""
import os
import time
import urllib.request
from pathlib import Path

from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.types import digest


def preparation_spec():
    sha = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
    return dict(schema_version='evidence-public-clip-input-v1',
                url='https://openaipublic.azureedge.net/clip/models/'+sha+'/ViT-B-32.pt',
                sha256=sha,max_bytes=400_000_000,max_seconds=600,
                clip=dict(embed_dim=512,image_size=224,vision_width=768,vision_layers=12,patch_size=32,
                          context_length=77,vocab_size=49408,text_width=512,text_heads=8,text_layers=12,fixture=False))


def preparation_scenario():
    return 'evidence_weights:'+digest(preparation_spec())


def download_verified(url,target,expected_sha256,*,max_bytes,max_seconds):
    """Never overwrite existing evidence; retain partial bytes when validation fails."""
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    partial = target.with_suffix(target.suffix+'.partial')
    started, count = time.monotonic(), 0
    with partial.open('xb') as output, urllib.request.urlopen(url,timeout=30) as source:
        while True:
            chunk = source.read(1024*1024)
            if time.monotonic()-started > max_seconds:
                raise ValueError('download wall limit exceeded')
            if not chunk:
                break
            count += len(chunk)
            if count > max_bytes:
                raise ValueError('download byte limit exceeded')
            output.write(chunk)
        output.flush();os.fsync(output.fileno())
    if file_sha256(partial) != expected_sha256:
        raise ValueError('download SHA-256 mismatch')
    # Exclusive hard-link publication, then remove only the temporary name we created.
    os.link(partial,target)
    partial.unlink()


def prepare_weights(context):
    import torch
    from semantic_retrieval.data.tokenizer import ClipTokenizer
    from semantic_retrieval.evidence.frame_cache import publish_json
    from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
    from semantic_retrieval.models.weights import load_clip_state
    spec = preparation_spec()
    path = context.paths.arrays/'ViT-B-32.pt'
    started = time.monotonic()
    download_verified(spec['url'],path,spec['sha256'],max_bytes=spec['max_bytes'],max_seconds=spec['max_seconds'])
    torch.manual_seed(713)
    model = ClipBackbone(ClipConfig(**spec['clip'])).eval()
    load_clip_state(path,spec['sha256'],model)
    ids,mask = ClipTokenizer().encode_batch(['a video of a person walking'],77)
    with torch.no_grad():
        _,text = model.encode_text(ids,mask)
        frames = model.encode_frames(torch.randn(1,1,3,224,224),torch.ones(1,1,dtype=torch.bool))
    if text.shape != (1,512) or frames.shape != (1,1,512) or not torch.isfinite(text).all() or not torch.isfinite(frames).all():
        raise ValueError('pretrained CPU text/frame witness failed')
    refs = [context.array_store.write('pretrained_text_witness',text.numpy()).as_dict(),
            context.array_store.write('pretrained_frame_witness',frames.numpy()).as_dict()]
    report = dict(schema_version=spec['schema_version'],source_sha=context.config.source_sha,
                  manifest_sha256=context.config.manifest_sha256,weights=dict(path=str(path),sha256=file_sha256(path)),
                  spec=spec,precision='fp32',device='cpu',scientific_data_read=False,test_authorized=False,
                  exact_state_keys_and_shapes=True,finite_parameters=True,real_text_and_frame_forward=True,
                  parameter_count=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.monotonic()-started,
                  array_references=refs)
    publish_json(context.paths.provenance/'pretrained-input.json',report)
    return report
