"""Actual baseline capacity, communication, metric and fresh-model resume checks."""
import hashlib
import math
import gc
import time

import numpy as np
import torch
import torch.distributed as dist

from semantic_retrieval.models.clip import ClipBackbone
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.runtime.distributed import gather_metadata, gather_with_grad
from semantic_retrieval.full_data.models import BaselineModel
from semantic_retrieval.full_data.metrics import retrieval_metrics
from .runtime import manifest_config
from .loader import batch_loader
from torch.utils.data import TensorDataset
from .training import make_optimizer, save_state, restore_state
from experiment_artifacts import artifact_json_bytes


def optimizer_fingerprint(value):
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().contiguous()
            digest.update(str((str(x.dtype), tuple(x.shape))).encode()); digest.update(x.numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=repr):
                digest.update(repr(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: visit(item)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def initial_clip(settings):
    clip = ClipBackbone(settings.clip)
    load_clip_state(settings.clip_weights, settings.clip_sha256, clip)
    return clip.state_dict()


def make_model(settings, method, clip_state, device):
    model = BaselineModel(method, settings.clip, clip_state, temporal_layers=1 if settings.clip.fixture else 4).to(device)
    network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                                       find_unused_parameters=True) if dist.is_initialized() else model
    return model, network


def step(model, network, optimizer, batch):
    model.train(); optimizer.zero_grad(set_to_none=True)
    output = network(*batch)
    output['loss'].backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError('nonfinite baseline gradient: '+name)
    intended = [model.clip.text_projection, model.clip.visual.proj]
    if model.method == 'uatvr':
        intended += [model.core.uncertain_net_text.fc.weight, model.core.pie_net_video.fc.weight,
                     model.core.frame_position_embeddings.weight]
    if any(p.grad is None or p.grad.abs().sum() == 0 for p in intended):
        raise ValueError('intended baseline parameters lack a nonzero gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        model.clip.logit_scale.clamp_(max=math.log(100))
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite baseline parameter after optimizer step')
    return {k: float(output[k].detach()) for k in ('loss', 'contrastive', 'mil', 'kl')}


def checkpoint_identity(context, settings, method):
    payload = manifest_config(settings)
    payload['evaluation'].pop('resume')  # Operational lineage, not the scientific recipe.
    return dict(source_sha=context.config.source_sha,
                recipe_sha256=hashlib.sha256(artifact_json_bytes(dict(config=payload, method=method))).hexdigest(),
                input_sha256=hashlib.sha256(artifact_json_bytes(dict(clip=settings.clip_sha256, source=settings.source_report_sha256))).hexdigest(),
                method=method, world_size=dist.get_world_size() if dist.is_initialized() else 1)


def _synthetic_batch(settings, count, rank, device):
    gen = torch.Generator().manual_seed(2026+rank)
    ids = torch.randint(1, settings.clip.vocab_size-2, (count, settings.text_length), generator=gen)
    ids[:, 0] = settings.clip.vocab_size-2; ids[:, -1] = settings.clip.vocab_size-1
    text_mask = torch.ones_like(ids, dtype=torch.bool)
    end = settings.text_length//2
    ids[0, end-1] = settings.clip.vocab_size-1; ids[0, end:] = 0; text_mask[0, end:] = False
    video = torch.randn(count, settings.frames, 3, settings.clip.image_size, settings.clip.image_size, generator=gen)
    mask = torch.ones(count, settings.frames, dtype=torch.bool); mask[0, 1:] = False; video[0, 1:] = 0
    return tuple(x.to(device) for x in (ids, text_mask, video, mask, torch.arange(rank*count, (rank+1)*count)))


def _normalization_probe(device, rank, world):
    probe = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad(): probe.weight.fill_(2)
    network = torch.nn.parallel.DistributedDataParallel(probe, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else probe
    local = network(torch.tensor([[float(rank+1)]], device=device))
    gather_with_grad(local).square().mean().backward()
    expected = 4*sum(i*i for i in range(1, world+1))/world
    return bool(torch.allclose(probe.weight.grad, torch.tensor([[expected]], device=device), rtol=0, atol=1e-6))


@torch.no_grad()
def _metric_probe(model, batch, store, method, rank):
    model.eval()
    text, video = model.encode_text(*batch[:2]), model.encode_video(*batch[2:4])
    text = {k: gather_metadata(v) for k, v in text.items()}
    video = {k: gather_metadata(v) for k, v in video.items()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(batch[2].device).clone() if batch[2].is_cuda else None
    score = model.score(text, video)
    n = len(score); perm = torch.arange(n-1, -1, -1, device=score.device)
    other = model.score({k: v[perm] for k, v in text.items()}, {k: v[perm] for k, v in video.items()})
    torch.testing.assert_close(other, score[perm][:, perm], atol=1e-6, rtol=1e-6)
    names = [f'v{i:06d}' for i in range(n)]; queries = [f'q{i:06d}' for i in range(n)]
    scores = score.cpu().numpy()
    result = retrieval_metrics(scores, names, names, query_ids=queries)
    reverse = retrieval_metrics(other.cpu().numpy(), names[::-1], names[::-1], query_ids=queries[::-1])
    if (not np.array_equal(result['t2v_ranks'][::-1], reverse['t2v_ranks'])
            or not np.array_equal(result['v2t_ranks'][::-1], reverse['v2t_ranks'])):
        raise ValueError('actual baseline metric changes under candidate permutation')
    rng_ok = torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None: rng_ok &= torch.equal(cuda_rng, torch.cuda.get_rng_state(batch[2].device))
    refs = []
    if rank == 0:
        refs = [store.write('synthetic_score_'+method, scores).as_dict(),
                store.write('synthetic_t2v_ranks_'+method, result['t2v_ranks']).as_dict(),
                store.write('synthetic_v2t_ranks_'+method, result['v2t_ranks']).as_dict()]
    return bool(rng_ok), refs


def qualify(context, settings, device):
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    if not _normalization_probe(device, rank, world):
        raise ValueError('global DDP gradient normalization probe failed')
    clip_state = initial_clip(settings)
    refs, observations = [], []
    checks = dict(actual_forward_backward=True, intended_gradients=True, global_gradient_normalization=True,
                  exact_resume=True, rank_weights_equal=True, finite_fp32=True, metric_permutation=True,
                  eval_rng_preserved=True, strict_checkpoint_rejection=True)
    for method in settings.methods:
        started = time.monotonic()
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model, network = make_model(settings, method, clip_state, device)
        opt = make_optimizer(model, learning_rate=settings.recipe(method)['learning_rate'], total_steps=20)
        local_count = settings.recipe(method)['global_batch']//world
        cpu_batch = _synthetic_batch(settings, local_count, rank, torch.device('cpu'))
        with batch_loader(TensorDataset(*cpu_batch), [list(range(local_count))], workers=settings.loader_workers,
                          pin_memory=device.type == 'cuda') as iterator:
            batch = tuple(x.to(device, non_blocking=True) for x in next(iterator))
        del cpu_batch
        before = model.clip.text_projection.detach().clone()
        first, second = step(model, network, opt, batch), step(model, network, opt, batch)
        if torch.equal(before, model.clip.text_projection):
            raise ValueError('official warmup produced no CLIP update after two steps')
        identity = checkpoint_identity(context, settings, method)
        checkpoint = context.paths.checkpoints/(method+'-step-2.pt')
        save_state(checkpoint, model, opt, identity, dict(epoch=0, batch_cursor=2, update=2), context.array_store)
        expected = step(model, network, opt, batch)
        expected_model, expected_opt = model_sha256(model), optimizer_fingerprint(opt.state_dict())
        rng_ok, arrays = _metric_probe(model, batch, context.array_store, method, rank)
        if not rng_ok: raise ValueError('evaluation altered training random state')
        refs.extend(arrays)
        del model, network, opt; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        model, network = make_model(settings, method, clip_state, device)
        opt = make_optimizer(model, learning_rate=settings.recipe(method)['learning_rate'], total_steps=20)
        progress = restore_state(checkpoint, model, opt, identity)
        actual = step(model, network, opt, batch)
        if (progress != dict(epoch=0, batch_cursor=2, update=2) or actual != expected
                or model_sha256(model) != expected_model or optimizer_fingerprint(opt.state_dict()) != expected_opt):
            raise ValueError('fresh-model exact checkpoint resume failed')
        pre_reject = model_sha256(model)
        try:
            restore_state(checkpoint, model, opt, dict(identity, input_sha256='0'*64))
        except ValueError:
            if model_sha256(model) != pre_reject: raise ValueError('rejected checkpoint mutated model')
        else:
            raise ValueError('incompatible checkpoint was accepted')
        hashes = [expected_model]
        if world > 1:
            hashes = [None]*world; dist.all_gather_object(hashes, expected_model)
        if len(set(hashes)) != 1: raise ValueError('baseline weights disagree across ranks')
        observations.append(dict(method=method, first_loss=first['loss'], second_loss=second['loss'], resume_loss=actual['loss'],
                                 model_sha256=expected_model, optimizer_sha256=expected_opt,
                                 elapsed_seconds=time.monotonic()-started,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0))
        del model, network, opt, batch; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    return dict(schema_version='full-data-qualification-v1', world_size=world, device_type=device.type,
                production_capacity=not settings.clip.fixture, qualified_methods=list(settings.methods), checks=checks,
                source_sha=context.config.source_sha, manifest_sha256=context.config.manifest_sha256,
                test_authorized=False, observations=observations, array_references=refs)
