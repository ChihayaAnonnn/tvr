"""Ordinary config-driven Gaussian KL training, smoke test and resume."""
import argparse
from collections import Counter
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

from semantic_retrieval.full_data.baseline import evaluate_complete, choose_epoch
from semantic_retrieval.full_data.campaign import FORMAL
from semantic_retrieval.full_data.loader import CaptionDataset, batch_loader
from semantic_retrieval.full_data.qualification import optimizer_fingerprint
from semantic_retrieval.full_data.training import make_optimizer, unique_video_batches, _optimizer_state, _progress
from semantic_retrieval.full_data.worker import _collective_call
from semantic_retrieval.models.clip import ClipBackbone
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.runtime.checkpoint import rng_state, restore_rng, model_sha256
from semantic_retrieval.runtime.distributed import configure_fp32
from .config import AqsConfig
from .fusion_config import FusionConfig
from .fusion_model import FusionAqsModel
from .fusion_training import fusion_recipe, fusion_step
from .training import video_id_mapping


def read_cohort(path, *, train_videos, val_videos, captions_per_video):
    with Path(path).open() as stream:
        cohort = json.load(stream)
    videos, captions = cohort['videos'], cohort['captions']
    if set(captions) != {'train', 'val'} or len(videos) != train_videos + val_videos:
        raise ValueError('unexpected train/validation cohort')
    queries, ownership = set(), {}
    for split, expected in (('train', train_videos), ('val', val_videos)):
        ids = {i for i, row in videos.items() if row['split'] == split}
        if len(ids) != expected:
            raise ValueError('video split cardinality changed')
        counts = Counter()
        for row in captions[split]:
            i, q = row['official_video_id'], row['query_id']
            if i not in ids or q in queries or not row['query_text'].strip():
                raise ValueError('caption identity or train/validation overlap')
            queries.add(q); counts[i] += 1
        if set(counts) != ids or set(counts.values()) != {captions_per_video}:
            raise ValueError('incomplete caption coverage')
        for i in ids:
            component = videos[i]['source_component']
            if component in ownership and ownership[component] != split:
                raise ValueError('source component crosses train/validation')
            ownership[component] = split
            if not Path(videos[i]['path']).is_file():
                raise FileNotFoundError(videos[i]['path'])
    return videos, captions


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def _cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu()
    if isinstance(value, dict): return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(_cpu(v) for v in value)
    return value


def _resume_config(config):
    operational = {'output', 'loader_workers', 'checkpoint_interval', 'log_interval', 'memory_fraction'}
    return {k: v for k, v in config.items() if k not in operational}


def save_checkpoint(path, model, optimizer, config, progress, history):
    _progress(progress)
    world = dist.get_world_size() if dist.is_initialized() else 1
    states = [rng_state()]
    if world > 1:
        states = [None]*world; dist.all_gather_object(states, rng_state())
    def publish():
        if dist.is_initialized() and dist.get_rank() != 0: return
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): raise FileExistsError(target)
        payload = dict(schema='uatvr-gaussian-kl-training-v1', config=_resume_config(config),
            recipe=fusion_recipe(model), id_mapping=model.id_mapping_sha256, world_size=world,
            model=_cpu(model.state_dict()), optimizer=_cpu(optimizer.state_dict()),
            rng=states, progress=progress, history=history)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    _collective_call(publish)


def load_checkpoint(path, model, optimizer, config):
    def prepare():
        payload = torch.load(path, map_location='cpu', weights_only=True)
        world = dist.get_world_size() if dist.is_initialized() else 1
        if (payload['schema'] != 'uatvr-gaussian-kl-training-v1'
                or payload['config'] != _resume_config(config)
                or payload['recipe'] != fusion_recipe(model)
                or payload['id_mapping'] != model.id_mapping_sha256
                or payload['world_size'] != world or len(payload['rng']) != world):
            raise ValueError('checkpoint model/config/world mismatch')
        current = model.state_dict(); saved = payload['model']
        if (saved.keys() != current.keys() or any(saved[k].shape != v.shape or saved[k].dtype != v.dtype
                or not torch.isfinite(saved[k]).all() for k, v in current.items())):
            raise ValueError('checkpoint model state mismatch')
        _optimizer_state(payload['optimizer'], optimizer); _progress(payload['progress'])
        return payload
    payload = _collective_call(prepare)
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'][dist.get_rank() if dist.is_initialized() else 0])
    return payload


def global_losses(row, device):
    keys = ('loss', 'contrastive', 'aqs', 'aqs_weighted', 'mil', 'kl')
    values = torch.tensor([row[k] for k in keys], device=device, dtype=torch.float64)
    if dist.is_initialized(): dist.all_reduce(values); values /= dist.get_world_size()
    return dict(zip(keys, values.cpu().tolist()))


def checkpoint_link(output, name, target):
    temporary = output/(name+f'.{time.time_ns()}.tmp')
    temporary.symlink_to(Path(target).relative_to(output))
    os.replace(temporary, output/name)


def available_checkpoint(path):
    """Keep a completed checkpoint if resuming an earlier update after a crash."""
    candidate = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        path = Path(path); candidate = path; suffix = 0
        while candidate.exists():
            suffix += 1; candidate = path.with_name(f'{path.stem}-retry-{suffix}{path.suffix}')
    selected = [str(candidate) if candidate is not None else None]
    if dist.is_initialized(): dist.broadcast_object_list(selected, src=0)
    return Path(selected[0])


def validate_config(config):
    if (config['architecture'] != 'uatvr-aqs-kl-v1' or config['precision'] != 'fp32'
            or config['world_size'] != 4 or config['global_batch'] % 4
            or min(config['epochs'], config['global_batch'], config['checkpoint_interval'], config['log_interval']) < 1
            or config['learning_rate'] <= 0 or not math.isfinite(config['learning_rate'])
            or not 0 < config['memory_fraction'] <= 1 or config['seed'] < 0):
        raise ValueError('invalid four-rank Gaussian KL training configuration')
    if not FusionConfig(True, config['eta']).active or not AqsConfig(True, config['negatives'], config['aqs_weight']).active:
        raise ValueError('active Gaussian KL and AQS required')


def run(args):
    config = json.loads(args.config.read_text()); validate_config(config)
    config['output'] = str((args.output or Path(config['output'])).resolve())
    output = Path(config['output'])
    if os.environ.get('WORLD_SIZE') != '4' or torch.cuda.device_count() != 4:
        raise ValueError('use four visible GPUs and four torchrun processes')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local); torch.cuda.set_device(device)
    configure_fp32(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.cuda.set_per_process_memory_fraction(config['memory_fraction'], device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
    output_owned = False
    try:
        def prepare_output():
            if rank != 0: return
            if not args.resume: output.mkdir(parents=True, exist_ok=False)
            elif not output.is_dir(): raise FileNotFoundError(output)
            (output/'logs').mkdir(exist_ok=True); (output/'checkpoints').mkdir(exist_ok=True)
            if not args.resume: atomic_json(output/'config.json', config)
            atomic_json(output/'status.json', dict(state='running', smoke=args.smoke, pid=os.getpid(), start_time=time.time()))
            print('CONFIG '+json.dumps(config), flush=True)
            print('COMMAND '+repr(sys.argv), flush=True)
            print('ENV '+json.dumps(dict(python=sys.version, torch=str(torch.__version__),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), visible=os.environ.get('CUDA_VISIBLE_DEVICES'))), flush=True)
        _collective_call(prepare_output)
        output_owned = True
        random.seed(config['seed']+rank); np.random.seed(config['seed']+rank); torch.manual_seed(config['seed']+rank)
        videos, captions = _collective_call(lambda: read_cohort(config['cohort'], train_videos=config['train_videos'],
            val_videos=config['val_videos'], captions_per_video=config['captions_per_video']))
        _, mapping = video_id_mapping(list(videos))
        training = _collective_call(lambda: CaptionDataset(videos, captions['train']))
        identities = [r['official_video_id'] for r in captions['train']]
        batches = [unique_video_batches(identities, global_batch=config['global_batch'], world_size=4,
                    seed=config['seed'], epoch=e) for e in range(config['epochs'])]
        total_steps = sum(map(len, batches))
        def build():
            clip = ClipBackbone(FORMAL)
            load_clip_state(config['clip_weights'], config['clip_sha256'], clip)
            return FusionAqsModel(FORMAL, clip.state_dict(), fusion=FusionConfig(True, config['eta']),
                aqs=AqsConfig(True, config['negatives'], config['aqs_weight']), id_mapping_sha256=mapping).to(device)
        model = build()
        network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
        optimizer = make_optimizer(model, learning_rate=config['learning_rate'], total_steps=total_steps)
        if rank == 0: print(f'DATA train_videos={config["train_videos"]} train_captions={len(training)} val_videos={config["val_videos"]} val_captions={len(captions["val"])} updates={total_steps}', flush=True)
        progress = dict(epoch=0, batch_cursor=0, update=0); history = []; restored = None
        if args.resume:
            restored = load_checkpoint(args.resume, model, optimizer, config)
            progress, history = restored['progress'], restored['history']
            if rank == 0: print('RESUMED '+json.dumps(progress), flush=True)
        def local_batches(rows):
            return [list(b[rank*(len(b)//4):(rank+1)*(len(b)//4)]) for b in rows]
        if args.smoke:
            with batch_loader(training, local_batches(batches[0][:1]), workers=config['loader_workers'], pin_memory=True) as iterator:
                batch = _collective_call(lambda: tuple(x.to(device) for x in next(iterator)))
            first = fusion_step(model, network, optimizer, batch)
            second = fusion_step(model, network, optimizer, batch)
            if any(second[name+'_update_norm'] <= 0 for name in ('text_variance','video_variance','text_mean','video_mean')):
                raise ValueError('probability heads did not update')
            path = output/'checkpoints/smoke.pt'
            save_checkpoint(path, model, optimizer, config, dict(epoch=0, batch_cursor=2, update=2), [])
            expected = fusion_step(model, network, optimizer, batch)
            expected_model, expected_opt = model_sha256(model), optimizer_fingerprint(optimizer.state_dict())
            del model, network, optimizer
            gc.collect(); torch.cuda.empty_cache()
            model = build(); network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
            optimizer = make_optimizer(model, learning_rate=config['learning_rate'], total_steps=total_steps)
            load_checkpoint(path, model, optimizer, config)
            actual = fusion_step(model, network, optimizer, batch)
            if actual != expected or model_sha256(model) != expected_model or optimizer_fingerprint(optimizer.state_dict()) != expected_opt:
                raise ValueError('CUDA checkpoint exact continuation mismatch')
            model.eval()
            with torch.no_grad():
                t = model.encode_text(batch[0][:2], batch[1][:2]); v = model.encode_video(batch[2][:3], batch[3][:3])
                score = model.score(t, v)
                if score.shape != (2, 3) or not torch.isfinite(score).all(): raise ValueError('smoke inference failed')
                torch.testing.assert_close(score, model.score(t, v), rtol=0, atol=0)
            hashes = [None]*4; dist.all_gather_object(hashes, model_sha256(model))
            if len(set(hashes)) != 1: raise ValueError('rank models disagree')
            atomic_json(output/f'logs/rank-{rank}.json', dict(first=first, second=second, resumed=actual,
                peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3))
            _collective_call(lambda: atomic_json(output/'status.json', dict(state='smoke_passed', world_size=4,
                real_data=True, global_batch=config['global_batch'], exact_resume=True, test_accessed=False)) if rank == 0 else None)
            if rank == 0: print('SMOKE PASSED: real data, full capacity, FP32, four ranks, parameter updates, exact resume, inference', flush=True)
            return
        validation = _collective_call(lambda: CaptionDataset(videos, captions['val']))
        update = progress['update']; started = time.monotonic()
        for epoch in range(progress['epoch'], config['epochs']):
            skip = progress['batch_cursor'] if epoch == progress['epoch'] else 0
            rows = local_batches(batches[epoch])[skip:]
            with batch_loader(training, rows, workers=config['loader_workers'], pin_memory=True) as iterator:
                # A mid-epoch restart constructs an extra iterator, which consumes CPU RNG.
                if restored is not None and skip:
                    restore_rng(restored['rng'][rank]); restored = None
                for cursor, indices in enumerate(rows, skip+1):
                    batch = _collective_call(lambda: tuple(x.to(device, non_blocking=True) for x in next(iterator)))
                    if len(batch[0]) != len(indices): raise ValueError('batch coverage changed')
                    row = fusion_step(model, network, optimizer, batch); update += 1
                    losses = global_losses(row, device)
                    if cursor <= 3 or cursor % config['log_interval'] == 0:
                        message = dict(epoch=epoch+1, batch=cursor, batches=len(batches[epoch]), update=update,
                            seconds=time.monotonic()-started, peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
                            **row)
                        with (output/f'logs/rank-{rank}.jsonl').open('a') as stream: stream.write(json.dumps(message, allow_nan=False)+'\n')
                        if rank == 0: print('TRAIN '+json.dumps(dict(epoch=epoch+1, batch=cursor, batches=len(batches[epoch]),
                            update=update, seconds=message['seconds'], **losses)), flush=True)
                    if update % config['checkpoint_interval'] == 0 and cursor < len(batches[epoch]):
                        path = available_checkpoint(output/f'checkpoints/update-{update:06d}.pt')
                        save_checkpoint(path, model, optimizer, config, dict(epoch=epoch,batch_cursor=cursor,update=update), history)
                        _collective_call(lambda: checkpoint_link(output, 'latest.pt', path) if rank == 0 else None)
            scores, metrics = evaluate_complete(model, validation, device, expected_split='val')
            path = available_checkpoint(output/f'checkpoints/epoch-{epoch+1:02d}.pt')
            record = dict(epoch=epoch+1, t2v_r1=metrics['t2v']['R1'], t2v=metrics['t2v'], v2t=metrics['v2t'],
                          update=update, checkpoint=str(path))
            history.append(record)
            save_checkpoint(path, model, optimizer, config, dict(epoch=epoch+1,batch_cursor=0,update=update), history)
            def publish():
                if rank != 0: return
                np.save(output/f'epoch-{epoch+1:02d}-scores.npy', scores, allow_pickle=False)
                checkpoint_link(output, 'latest.pt', path)
                best = choose_epoch(history)
                checkpoint_link(output, 'best.pt', Path(next(r['checkpoint'] for r in history if r['epoch'] == best)))
                atomic_json(output/'metrics.json', dict(selected_epoch=best, epochs=history, test_accessed=False))
                atomic_json(output/'status.json', dict(state='running', epoch=epoch+1, update=update, selected_epoch=best))
                print('VALIDATION '+json.dumps(record), flush=True)
            _collective_call(publish)
        _collective_call(lambda: atomic_json(output/'status.json', dict(state='completed', update=update,
            selected_epoch=choose_epoch(history), test_accessed=False)) if rank == 0 else None)
    except Exception as exc:
        if rank == 0 and output_owned:
            atomic_json(output/'status.json', dict(state='failed', error=repr(exc), timestamp=time.time()))
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.smoke and args.resume: parser.error('smoke and resume cannot be combined')
    run(args)


if __name__ == '__main__': main()
