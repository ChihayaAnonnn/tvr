"""Sealed four-rank Gaussian KL CUDA entry; independent of historical campaigns."""
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

from experiment_artifacts import ArtifactArrayStore, artifact_json_bytes, strict_json_load, manifest_sha256, verify_committed_reservation, load_registry
from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json
from semantic_retrieval.full_data.baseline import evaluate_complete, choose_epoch
from semantic_retrieval.full_data.loader import CaptionDataset, read_complete, batch_loader
from semantic_retrieval.full_data.qualification import _synthetic_batch, optimizer_fingerprint
from semantic_retrieval.full_data.training import make_optimizer, unique_video_batches
from semantic_retrieval.full_data.worker import _collective_call
from semantic_retrieval.full_data.campaign import FORMAL
from semantic_retrieval.models.clip import ClipConfig, ClipBackbone
from semantic_retrieval.models.weights import load_clip_state
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.runtime.distributed import configure_fp32
from semantic_retrieval.runtime.input_bindings import manifest_input_references
from .config import AqsConfig
from .fusion_model import FusionAqsModel
from .fusion_config import FusionConfig
from .fusion_training import FusionBinding, fusion_step, fusion_recipe, save_fusion_state, restore_fusion_state
from .training import video_id_mapping


def digest(value):
    return hashlib.sha256(artifact_json_bytes(value)).hexdigest()


def validate_recipe(recipe):
    expected={'epochs','global_batch','learning_rate','seed','world_size','negatives','weight','validation_only'}
    if (set(recipe)!=expected or type(recipe['world_size']) is not int or recipe['world_size']!=4
            or recipe['validation_only'] is not True
            or any(type(recipe[k]) is not int or recipe[k]<1 for k in ('epochs','global_batch'))
            or recipe['global_batch']%4 or type(recipe['seed']) is not int or recipe['seed']<0
            or type(recipe['learning_rate']) not in (int,float)
            or not math.isfinite(recipe['learning_rate']) or recipe['learning_rate']<=0):
        raise ValueError('expected four-rank validation-only Gaussian KL recipe')
    if not AqsConfig(True,recipe['negatives'],recipe['weight']).active:
        raise ValueError('Gaussian KL+AQS launch requires active AQS')


def validate_launch(manifest, mode):
    if (manifest.get('study')!='uatvr-aqs-kl-training-v1'
            or manifest.get('test_authorized') is not False
            or type(manifest['environment']['gpu_count']) is not int
            or manifest['environment']['gpu_count']!=4 or manifest['environment']['precision']!='fp32'):
        raise ValueError('wrong four-GPU Gaussian KL scope')
    config=manifest['config'];evaluation=config['evaluation'];model=config['model']
    if (mode not in ('qualify','train') or evaluation['mode']!=mode or evaluation['dsl'] is not False
            or manifest['dataset']!=('synthetic' if mode=='qualify' else 'msrvtt')
            or model['architecture']!='uatvr-aqs-kl-v1'
            or config['optimizer']['kind']!='pinned_bert_adam'
            or config['scheduler']['kind']!='warmup_cosine'):
        raise ValueError('Gaussian KL launch mode/config mismatch')
    fusion=FusionConfig.from_recipe(model['fusion']);aqs=AqsConfig.from_recipe(config['loss']['aqs'])
    recipe=evaluation['training'];validate_recipe(recipe)
    if (not fusion.active or not aqs.active or aqs.negatives!=recipe['negatives'] or aqs.weight!=recipe['weight']
            or ClipConfig(**model['clip']) != FORMAL):
        raise ValueError('active Gaussian KL and production CLIP required')
    return dict(recipe=recipe,fusion=fusion,aqs=aqs)


def require_qualification(report, source_files, recipe, fusion, state):
    if (state!='completed' or report.get('passed') is not True or report.get('world_size')!=4
            or report.get('synthetic') is not True or report.get('test_accessed') is not False
            or report.get('source_files')!=source_files or report.get('recipe')!=recipe or report.get('fusion')!=fusion):
        raise ValueError('completed matching-source four-CUDA Gaussian KL qualification required')


def qualification_evidence(report, report_sha256, archived_sha256, expected_manifest_sha256):
    if (not isinstance(report_sha256,str) or len(report_sha256)!=64
            or report_sha256!=archived_sha256 or report.get('manifest_sha256')!=expected_manifest_sha256):
        raise ValueError('qualification evidence differs from archived terminal inventory')


def rank_batches(batches, rank, world):
    if world != 4 or rank not in range(world) or any(not b or len(b) % world for b in batches):
        raise ValueError('batches must shard completely across four ranks')
    return [list(b[rank*(len(b)//world):(rank+1)*(len(b)//world)]) for b in batches]


def validate_devices(visible, registered, available):
    selected = [value.strip() for value in visible.split(',')]
    actual = [available.get(value, value) for value in selected]
    if (len(registered) != 4 or len(set(registered)) != 4 or len(actual) != 4
            or actual != registered):
        raise ValueError('CUDA devices must match the four frozen GPU UUIDs in order')


def validate_output_root(root, manifest):
    if Path(root).resolve() != Path(manifest['artifact_path']).resolve():
        raise ValueError('output directory differs from the reserved artifact path')


def source_inventory(repo):
    names = subprocess.check_output(['git', 'ls-files', 'src', 'scripts', 'experiment_artifacts.py'],
                                    cwd=repo, text=True).splitlines()
    required = {name for name in names if name.endswith('.py')}
    required.update(('src/semantic_retrieval/aqs/fusion_launch.py',
                     'scripts/run_uatvr_gaussian_kl_4gpu.sh'))
    return {name: file_sha256(repo/name) for name in sorted(required)}


def validate_source_inventory(repo, recorded):
    if recorded != source_inventory(repo):
        raise ValueError('complete matching launch/dependency source inventory required')


def check_sources(repo, native):
    if native['registry_path'] != 'experiments/uatvr-aqs-kl-training-v1/registry.jsonl':
        raise ValueError('wrong Gaussian KL registry')
    verify_committed_reservation(repo, Path(native['registry_path']), native)
    import yaml
    study=yaml.safe_load((repo/'experiments/uatvr-aqs-kl-training-v1/study.yaml').read_text())
    cfg=native['config']
    if (cfg['evaluation']['training']!=study['recipe']
            or cfg['model']['fusion']!=FusionConfig(study['fusion']['enabled'],study['fusion']['eta']).recipe()
            or native['environment']['gpu_uuids']!=study['resource_policy']['gpu_uuids']):
        raise ValueError('manifest differs from frozen Gaussian KL study')
    validate_source_inventory(repo, native['config']['evaluation']['source_files'])


def run(args):
    root = args.run_dir.resolve()
    with (root/'manifest.json').open() as stream:
        native = strict_json_load(stream)
    validate_output_root(root, native)
    selected=validate_launch(native,args.mode)
    evaluation=native['config']['evaluation'];model_config=native['config']['model']
    manifest=dict(native,recipe=selected['recipe'],clip=model_config['clip'],source_sha=native['git_sha'],
        source_report=evaluation.get('source_report'),source_files=evaluation['source_files'])
    recipe = selected['recipe']
    repo = Path(__file__).resolve().parents[3]
    check_sources(repo, native)
    if (os.environ.get('WORLD_SIZE') != '4' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
            or torch.cuda.device_count() != 4):
        raise ValueError('launch with four torchrun ranks and deterministic cuBLAS')
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                       '--format=csv,noheader,nounits'], text=True).splitlines()
    available = dict(tuple(value.strip() for value in row.split(',')) for row in gpu_rows)
    validate_devices(os.environ.get('CUDA_VISIBLE_DEVICES', ''), native['environment']['gpu_uuids'], available)
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    configure_fp32(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.8, device)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=15))
    output = root
    output.mkdir(exist_ok=True)
    store = ArtifactArrayStore(output/'arrays'); (output/'checkpoints').mkdir(exist_ok=True)
    try:
        _collective_call(lambda: publish_json(output/'provenance/worker-started.json', dict(
            manifest_sha256=manifest_sha256(native), mode=args.mode, world_size=4,
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), timestamp=time.time())) if rank == 0 else None)
        env_ref=evaluation['environment_spec']
        def witness():
            import importlib.metadata
            if file_sha256(env_ref['path'])!=env_ref['sha256']:raise ValueError('environment spec changed')
            with Path(env_ref['path']).open() as stream:env=strict_json_load(stream)
            for package,version in env['packages'].items():
                if importlib.metadata.version(package)!=version:raise ValueError('environment version mismatch: '+package)
            x=torch.randn(64,64,generator=torch.Generator().manual_seed(713))
            torch.testing.assert_close((x.to(device)@x.to(device)).cpu(),x@x,atol=1e-4,rtol=1e-4)
        _collective_call(witness)
        if args.mode == 'train':
            def gate_check():
                ref=evaluation['qualification']
                if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('qualification report hash mismatch')
                with Path(ref['path']).open() as stream:gate=strict_json_load(stream)
                with (Path(ref['path']).parent.parent/'status.json').open() as stream:status=strict_json_load(stream)
                qroot=Path(ref['path']).parent.parent
                with (qroot/'manifest.json').open() as stream:parent_manifest=strict_json_load(stream)
                parent_selected = validate_launch(parent_manifest, 'qualify')
                if (parent_manifest['run_id'] != ref['run_id']
                        or parent_manifest['registry_path'] != native['registry_path']
                        or parent_selected != selected
                        or parent_manifest['config']['evaluation']['source_files'] != manifest['source_files']
                        or parent_manifest['config']['evaluation']['environment_spec'] != evaluation['environment_spec']):
                    raise ValueError('qualification model/source/environment identity differs')
                parent_digest=manifest_sha256(parent_manifest)
                qualification_evidence(gate,ref['sha256'],status['output_hashes'].get('provenance/result.json'),parent_digest)
                records=load_registry(repo/native['registry_path'])
                if not any(r['run_id']==ref['run_id'] and r['event']=='completed' and r['manifest_sha256']==parent_digest for r in records):
                    raise ValueError('qualification terminal registry must be committed')
                require_qualification(gate,manifest['source_files'],recipe,selected['fusion'].recipe(),status['state'])
            _collective_call(gate_check)
            refs=_collective_call(lambda:manifest_input_references(repo,native))
            videos,captions=_collective_call(lambda:read_complete(Path(manifest['source_report']['path']),manifest['source_report']['sha256'],refs))
            _,mapping=video_id_mapping(list(videos))
            training=_collective_call(lambda:CaptionDataset(videos,captions['train']))
        else:
            _,mapping=video_id_mapping([f'synthetic-{i:03d}' for i in range(recipe['global_batch'])])
        cfg = ClipConfig(**manifest['clip'])
        random.seed(recipe['seed']+rank); np.random.seed(recipe['seed']+rank); torch.manual_seed(recipe['seed']+rank)
        def initialize():
            clip = ClipBackbone(cfg)
            if args.mode=='train':
                load_clip_state(Path(manifest['input_checkpoints'][0]['path']), manifest['input_checkpoints'][0]['sha256'], clip)
            return FusionAqsModel(cfg,clip.state_dict(),fusion=selected['fusion'],aqs=selected['aqs'],id_mapping_sha256=mapping).to(device)
        model = initialize()
        network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
        if args.mode=='train':
            identities=[r['official_video_id'] for r in captions['train']]
            batches=[unique_video_batches(identities,global_batch=recipe['global_batch'],world_size=4,seed=recipe['seed'],epoch=e) for e in range(recipe['epochs'])]
            total_steps=sum(map(len,batches))
        else:
            # Certified cohort has 8246 videos x20 captions; same pinned schedule.
            total_steps=math.ceil(8246*20/recipe['global_batch'])*recipe['epochs']
        optimizer = make_optimizer(model, learning_rate=recipe['learning_rate'], total_steps=total_steps)
        base_recipe = dict(fusion_recipe(model), training=recipe)
        binding = FusionBinding(dict(source_sha=manifest['source_sha'], recipe_sha256=digest(base_recipe),
            input_sha256=digest(dict(checkpoints=manifest['input_checkpoints'], datasets=manifest['dataset_splits'])),
            method='uatvr', world_size=4), base_recipe, mapping)
        if args.mode == 'qualify':
            synthetic = SimpleNamespace(clip=cfg, frames=12, text_length=32)
            batch = _synthetic_batch(synthetic, recipe['global_batch']//4, rank, device)
            before = model.core.uncertain_net_video.fc.weight.detach().clone()
            first = fusion_step(model, network, optimizer, batch)
            checkpoint = output/'checkpoints/step-1.pt'
            save_fusion_state(checkpoint, model, optimizer, binding, dict(epoch=0, batch_cursor=1, update=1), store)
            expected = fusion_step(model, network, optimizer, batch)
            expected_model = model_sha256(model); expected_optimizer = optimizer_fingerprint(optimizer.state_dict())
            if torch.equal(before,model.core.uncertain_net_video.fc.weight):raise ValueError('Gaussian variance parameters did not update')
            import gc
            del model,network,optimizer
            gc.collect();torch.cuda.empty_cache()
            model=initialize()
            network=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local],find_unused_parameters=True)
            optimizer=make_optimizer(model,learning_rate=recipe['learning_rate'],total_steps=total_steps)
            progress = restore_fusion_state(checkpoint, model, optimizer, binding)
            actual = fusion_step(model, network, optimizer, batch)
            if (progress != dict(epoch=0, batch_cursor=1, update=1) or actual != expected
                    or model_sha256(model) != expected_model
                    or optimizer_fingerprint(optimizer.state_dict()) != expected_optimizer):
                raise ValueError('four-CUDA checkpoint continuation differs')
            with torch.no_grad():
                model.eval()
                text=model.encode_text(batch[0][:2],batch[1][:2]);visual=model.encode_video(batch[2][:3],batch[3][:3])
                score=model.score(text,visual)
                if score.shape!=(2,3) or not torch.isfinite(score).all():raise ValueError('Gaussian KL inference qualification failed')
                torch.testing.assert_close(score,model.score(text,visual),rtol=0,atol=0)
            weights=[None]*4;dist.all_gather_object(weights,model_sha256(model))
            if len(set(weights))!=1:raise ValueError('rank model states disagree after exact resume')
            report = dict(passed=True, manifest_sha256=manifest_sha256(native), world_size=4,
                source_files=manifest['source_files'],recipe=recipe,fusion=selected['fusion'].recipe(),
                global_batch=recipe['global_batch'], first=first, resumed=actual,
                max_memory_gib=torch.cuda.max_memory_allocated()/1024**3, synthetic=True, test_accessed=False)
            _collective_call(lambda: publish_json(output/'provenance/result.json', report) if rank == 0 else None)
            if rank == 0: print('QUALIFICATION PASSED Gaussian KL four-CUDA exact fresh-model resume',flush=True)
            return
        validation = _collective_call(lambda: CaptionDataset(videos, captions['val']))
        records = []; update = 0
        for epoch, epoch_batches in enumerate(batches, 1):
            check_sources(repo,native)
            local_batches = rank_batches(epoch_batches, rank, 4)
            rows = []; started = time.monotonic()
            with batch_loader(training, local_batches, workers=4, pin_memory=True) as iterator:
                for cursor, indices in enumerate(local_batches, 1):
                    batch = _collective_call(lambda: next(iterator))
                    if len(batch[0]) != len(indices): raise ValueError('batch coverage changed')
                    row = fusion_step(model, network, optimizer, tuple(x.to(device, non_blocking=True) for x in batch))
                    keys = sorted(row)
                    values = torch.tensor([row[k] for k in keys], device=device, dtype=torch.float64)
                    dist.all_reduce(values); values /= 4
                    row = dict(zip(keys, values.cpu().tolist())); update += 1
                    rows.append(row)
                    if rank == 0 and (cursor <= 3 or cursor % 20 == 0):
                        message = dict(epoch=epoch, batch=cursor, batches=len(local_batches), update=update,
                            seconds=time.monotonic()-started, **row)
                        with (root/'logs/training.jsonl').open('a') as stream: stream.write(json.dumps(message)+'\n')
                        print(json.dumps(message), flush=True)
                    if cursor % 500 == 0:
                        save_fusion_state(root/f'checkpoints/update-{update:06d}.pt', model, optimizer, binding,
                                       dict(epoch=epoch-1, batch_cursor=cursor, update=update), store)
            scores, metrics = evaluate_complete(model, validation, device, expected_split='val')
            path = root/f'checkpoints/epoch-{epoch:02d}.pt'
            save_fusion_state(path, model, optimizer, binding, dict(epoch=epoch, batch_cursor=0, update=update), store)
            def publish():
                if rank != 0: return
                score_ref = store.write(f'epoch_{epoch}_scores', scores).as_dict()
                loss_ref = store.write(f'epoch_{epoch}_losses', np.asarray([[r[k] for k in keys] for r in rows])).as_dict()
                record = dict(epoch=epoch, t2v_r1=metrics['t2v']['R1'], checkpoint=str(path),
                    t2v=metrics['t2v'], v2t=metrics['v2t'], scores=score_ref, losses=loss_ref, loss_columns=keys,
                    train_captions=len(training), validation_captions=len(validation), validation_videos=500,
                    optimizer_updates=update, test_accessed=False)
                records.append(record); publish_json(root/f'provenance/epoch-{epoch:02d}.json', record)
                print('VALIDATION '+json.dumps(record), flush=True)
            _collective_call(publish)
        _collective_call(lambda: publish_json(root/'provenance/result.json', dict(completed=True,
            selected_epoch=choose_epoch(records), epochs=records, manifest_sha256=manifest_sha256(native),
            test_accessed=False)) if rank == 0 else None)
    except Exception as exc:
        if rank == 0:
            publish_json(output/'provenance/failure.json', dict(error=repr(exc), timestamp=time.time()))
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
