"""Independent two-rank AQS launch; never enters a baseline/Laya campaign."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from experiment_artifacts import ArtifactArrayStore, artifact_json_bytes, strict_json_load
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.full_data.baseline import evaluate_complete, choose_epoch
from semantic_retrieval.full_data.loader import CaptionDataset, read_complete, batch_loader
from semantic_retrieval.full_data.qualification import _synthetic_batch, optimizer_fingerprint
from semantic_retrieval.full_data.training import make_optimizer, unique_video_batches
from semantic_retrieval.full_data.worker import _collective_call
from semantic_retrieval.models.clip import ClipConfig, ClipBackbone
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.input_bindings import manifest_input_references
from .config import AqsConfig
from .model import AqsBaselineModel
from .training import (CheckpointBinding, aqs_step, model_recipe, video_id_mapping,
                       save_aqs_state, restore_aqs_state)


def digest(value):
    return hashlib.sha256(artifact_json_bytes(value)).hexdigest()


def validate_recipe(recipe):
    expected = {'epochs', 'global_batch', 'learning_rate', 'seed', 'world_size',
                'negatives', 'weight', 'validation_only'}
    if (set(recipe) != expected or recipe['world_size'] != 2
            or recipe['validation_only'] is not True
            or any(type(recipe[k]) is not int or recipe[k] < 1 for k in ('epochs', 'global_batch'))
            or recipe['global_batch'] % 2 or type(recipe['seed']) is not int or recipe['seed'] < 0
            or type(recipe['learning_rate']) not in (int, float)
            or not math.isfinite(recipe['learning_rate']) or recipe['learning_rate'] <= 0):
        raise ValueError('invalid two-rank validation-only AQS recipe')
    if not AqsConfig(True, recipe['negatives'], recipe['weight']).active:
        raise ValueError('AQS training requires active auxiliary loss')


def rank_batches(batches, rank, world):
    if world != 2 or rank not in range(world) or any(not b or len(b) % world for b in batches):
        raise ValueError('batches must shard completely across two ranks')
    return [list(b[rank*(len(b)//world):(rank+1)*(len(b)//world)]) for b in batches]


def check_sources(repo, manifest):
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if sha != manifest['source_sha']:
        raise ValueError('source commit changed after launch preparation')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'scripts', 'experiment_artifacts.py'],
                   cwd=repo, check=True, capture_output=True)
    for name, sha in manifest['source_files'].items():
        if file_sha256(repo/name) != sha:
            raise ValueError('source bytes changed: '+name)


def run(args):
    root = args.run_dir.resolve()
    with (root/'manifest.json').open() as stream:
        manifest = strict_json_load(stream)
    recipe = manifest['recipe']; validate_recipe(recipe)
    repo = Path(__file__).resolve().parents[3]
    check_sources(repo, manifest)
    if os.environ.get('WORLD_SIZE') != '2' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('launch with two torchrun ranks and deterministic cuBLAS')
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    configure_fp32(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.cuda.set_per_process_memory_fraction(60*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=15))
    output = root/'qualification' if args.mode == 'qualify' else root
    output.mkdir(exist_ok=True)
    store = ArtifactArrayStore(output/'arrays'); (output/'checkpoints').mkdir(exist_ok=True)
    try:
        _collective_call(lambda: publish_json(output/'started.json', dict(
            manifest_sha256=digest(manifest), mode=args.mode, world_size=2,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), timestamp=time.time())) if rank == 0 else None)
        if args.mode == 'train':
            with (root/'qualification/result.json').open() as stream: gate = strict_json_load(stream)
            if gate['manifest_sha256'] != digest(manifest) or gate['passed'] is not True:
                raise ValueError('matching dual-GPU qualification is required')
        refs = _collective_call(lambda: manifest_input_references(repo, manifest))
        videos, captions = _collective_call(lambda: read_complete(
            Path(manifest['source_report']['path']), manifest['source_report']['sha256'], refs))
        _, mapping = video_id_mapping(list(videos))
        training = _collective_call(lambda: CaptionDataset(videos, captions['train']))
        cfg = ClipConfig(**manifest['clip'])
        random.seed(recipe['seed']+rank); np.random.seed(recipe['seed']+rank); torch.manual_seed(recipe['seed']+rank)
        def initialize():
            clip = ClipBackbone(cfg)
            load_clip_state(Path(manifest['input_checkpoints'][0]['path']), manifest['input_checkpoints'][0]['sha256'], clip)
            return AqsBaselineModel(cfg, clip.state_dict(), aqs=AqsConfig(True, recipe['negatives'], recipe['weight']),
                                    id_mapping_sha256=mapping).to(device)
        model = initialize()
        network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
        identities = [r['official_video_id'] for r in captions['train']]
        batches = [unique_video_batches(identities, global_batch=recipe['global_batch'], world_size=2,
                    seed=recipe['seed'], epoch=e) for e in range(recipe['epochs'])]
        optimizer = make_optimizer(model, learning_rate=recipe['learning_rate'], total_steps=sum(map(len, batches)))
        base_recipe = dict(model=model_recipe(model), training=recipe)
        binding = CheckpointBinding(dict(source_sha=manifest['source_sha'], recipe_sha256=digest(base_recipe),
            input_sha256=digest(dict(checkpoints=manifest['input_checkpoints'], datasets=manifest['dataset_splits'])),
            method='uatvr', world_size=2), base_recipe, mapping)
        if args.mode == 'qualify':
            synthetic = SimpleNamespace(clip=cfg, frames=12, text_length=32)
            batch = _synthetic_batch(synthetic, recipe['global_batch']//2, rank, device)
            before = model.core.pie_net_video.fc.weight.detach().clone()
            first = aqs_step(model, network, optimizer, batch)
            checkpoint = output/'checkpoints/step-1.pt'
            save_aqs_state(checkpoint, model, optimizer, binding, dict(epoch=0, batch_cursor=1, update=1), store)
            expected = aqs_step(model, network, optimizer, batch)
            expected_model = model_sha256(model); expected_optimizer = optimizer_fingerprint(optimizer.state_dict())
            if torch.equal(before, model.core.pie_net_video.fc.weight): raise ValueError('parameters did not update')
            progress = restore_aqs_state(checkpoint, model, optimizer, binding)
            actual = aqs_step(model, network, optimizer, batch)
            if (progress != dict(epoch=0, batch_cursor=1, update=1) or actual != expected
                    or model_sha256(model) != expected_model
                    or optimizer_fingerprint(optimizer.state_dict()) != expected_optimizer):
                raise ValueError('dual-CUDA checkpoint continuation differs')
            report = dict(passed=True, manifest_sha256=digest(manifest), world_size=2,
                global_batch=recipe['global_batch'], first=first, resumed=actual,
                max_memory_gib=torch.cuda.max_memory_allocated()/1024**3, synthetic=True, test_accessed=False)
            _collective_call(lambda: publish_json(output/'result.json', report) if rank == 0 else None)
            if rank == 0: print('QUALIFICATION PASSED '+json.dumps(report), flush=True)
            return
        validation = _collective_call(lambda: CaptionDataset(videos, captions['val']))
        records = []; update = 0
        for epoch, epoch_batches in enumerate(batches, 1):
            local_batches = rank_batches(epoch_batches, rank, 2)
            rows = []; started = time.monotonic()
            with batch_loader(training, local_batches, workers=4, pin_memory=True) as iterator:
                for cursor, indices in enumerate(local_batches, 1):
                    batch = _collective_call(lambda: next(iterator))
                    if len(batch[0]) != len(indices): raise ValueError('batch coverage changed')
                    row = aqs_step(model, network, optimizer, tuple(x.to(device, non_blocking=True) for x in batch))
                    keys = sorted(row)
                    values = torch.tensor([row[k] for k in keys], device=device, dtype=torch.float64)
                    dist.all_reduce(values); values /= 2
                    row = dict(zip(keys, values.cpu().tolist())); update += 1
                    rows.append(row)
                    if rank == 0 and (cursor <= 3 or cursor % 20 == 0):
                        message = dict(epoch=epoch, batch=cursor, batches=len(local_batches), update=update,
                            seconds=time.monotonic()-started, **row)
                        with (root/'training.jsonl').open('a') as stream: stream.write(json.dumps(message)+'\n')
                        print(json.dumps(message), flush=True)
                    if cursor % 500 == 0:
                        save_aqs_state(root/f'checkpoints/update-{update:06d}.pt', model, optimizer, binding,
                                       dict(epoch=epoch-1, batch_cursor=cursor, update=update), store)
            scores, metrics = evaluate_complete(model, validation, device, expected_split='val')
            path = root/f'checkpoints/epoch-{epoch:02d}.pt'
            save_aqs_state(path, model, optimizer, binding, dict(epoch=epoch, batch_cursor=0, update=update), store)
            def publish():
                if rank != 0: return
                score_ref = store.write(f'epoch_{epoch}_scores', scores).as_dict()
                loss_ref = store.write(f'epoch_{epoch}_losses', np.asarray([[r[k] for k in keys] for r in rows])).as_dict()
                record = dict(epoch=epoch, t2v_r1=metrics['t2v']['R1'], checkpoint=str(path),
                    t2v=metrics['t2v'], v2t=metrics['v2t'], scores=score_ref, losses=loss_ref, loss_columns=keys,
                    train_captions=len(training), validation_captions=len(validation), validation_videos=500,
                    optimizer_updates=update, test_accessed=False)
                records.append(record); publish_json(root/f'epoch-{epoch:02d}.json', record)
                print('VALIDATION '+json.dumps(record), flush=True)
            _collective_call(publish)
        _collective_call(lambda: publish_json(root/'result.json', dict(completed=True,
            selected_epoch=choose_epoch(records), epochs=records, manifest_sha256=digest(manifest),
            test_accessed=False)) if rank == 0 else None)
    except Exception as exc:
        if rank == 0:
            publish_json(output/'failure.json', dict(error=repr(exc), timestamp=time.time()))
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['qualify', 'train'], required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
