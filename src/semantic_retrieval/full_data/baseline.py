"""Complete-caption training and full-gallery, bidirectional validation."""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from experiment_artifacts import strict_json_load, verify_array_reference

from semantic_retrieval.data.episodes import file_sha256
from semantic_retrieval.evidence.frame_cache import publish_json, safe_path
from semantic_retrieval.runtime.artifacts import publish_metrics
from semantic_retrieval.runtime.checkpoint import model_sha256
from semantic_retrieval.runtime.distributed import gather_metadata
from .loader import read_complete, CaptionDataset, batch_loader
from semantic_retrieval.full_data.metrics import retrieval_metrics
from .qualification import initial_clip, make_model, checkpoint_identity, step
from .training import make_optimizer, unique_video_batches, save_state, restore_state, carry_checkpoint
from .worker import _collective_call


def choose_epoch(rows):
    if (not rows or len({r['epoch'] for r in rows}) != len(rows)
            or any(type(r['epoch']) is not int or r['epoch'] < 1 or not math.isfinite(r['t2v_r1'])
                   or not 0 <= r['t2v_r1'] <= 100 for r in rows)):
        raise ValueError('invalid validation epoch selection records')
    return min(rows, key=lambda r: (-r['t2v_r1'], r['epoch']))['epoch']


def train_epoch(model, network, optimizer, dataset, batches, device, *, loader_workers=0):
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    local = [list(batch[rank*(len(batch)//world):(rank+1)*(len(batch)//world)]) for batch in batches]
    if any(not b for b in local): raise ValueError('empty training rank cannot be silently skipped')
    values, count = [], 0
    data_seconds, step_seconds, window_count = 0.0, 0.0, 0
    with batch_loader(dataset, local, workers=loader_workers, pin_memory=device.type == 'cuda') as iterator:
        for indices in local:
            started = time.perf_counter()
            batch = _collective_call(lambda: next(iterator))
            data_seconds += time.perf_counter()-started
            if len(batch[0]) != len(indices): raise ValueError('training batch cardinality changed')
            started = time.perf_counter()
            result = step(model, network, optimizer, tuple(x.to(device, non_blocking=True) for x in batch))
            loss = torch.tensor([result[k] for k in ('loss', 'contrastive', 'mil', 'kl')], device=device, dtype=torch.float64)
            if world > 1: dist.all_reduce(loss); loss /= world
            count += len(indices); window_count += len(indices)*world
            values.append(loss.cpu().tolist())  # Synchronizes the measured update.
            step_seconds += time.perf_counter()-started
            if len(values) % 100 == 0:
                if rank == 0:
                    print(f'updates={len(values)}/{len(local)} loss={result["loss"]:.6f} '
                          f'data_wait_s={data_seconds/100:.4f} update_s={step_seconds/100:.4f} '
                          f'global_samples_s={window_count/(data_seconds+step_seconds):.2f}', flush=True)
                data_seconds, step_seconds, window_count = 0.0, 0.0, 0
    if count != sum(len(b) for b in local): raise ValueError('incomplete caption traversal')
    return np.asarray(values, dtype=np.float32)


@torch.no_grad()
def evaluate_complete(model, dataset, device, *, expected_split='val'):
    model.eval()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    if len(dataset) % world or len(dataset) < world:
        raise ValueError('validation queries cannot be equally sharded without omission')
    gallery = sorted({r['official_video_id'] for r in dataset.captions})
    if expected_split not in {'val', 'test'} or any(dataset.videos[i]['split'] != expected_split for i in gallery):
        raise ValueError('validation gallery contains training videos')
    n = len(dataset)//world; begin, end = rank*n, (rank+1)*n
    def concatenate(parts):
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    text_parts = [model.encode_text(dataset.tokens[i:min(i+128, end)].to(device),
                                    dataset.text_masks[i:min(i+128, end)].to(device)) for i in range(begin, end, 128)]
    text = concatenate(text_parts)
    video_parts = []
    for i in range(0, len(gallery), 16):
        decoded = _collective_call(lambda: [dataset.decode(v) for v in gallery[i:i+16]])
        frames = torch.stack([v[0] for v in decoded]).to(device)
        masks = torch.stack([v[1] for v in decoded]).to(device)
        video_parts.append(model.encode_video(frames, masks))
    video = concatenate(video_parts)
    scores = torch.empty((n, len(gallery)), dtype=torch.float32, device=device)
    for q in range(0, n, 32):
        for v in range(0, len(gallery), 64):
            scores[q:q+32, v:v+64] = model.score({k: x[q:q+32] for k, x in text.items()},
                                                {k: x[v:v+64] for k, x in video.items()})
    all_scores = gather_metadata(scores).cpu().numpy()
    result = retrieval_metrics(all_scores, [r['official_video_id'] for r in dataset.captions], gallery,
                                query_ids=[r['query_id'] for r in dataset.captions])
    return all_scores, result


def train_baseline(context, settings, device):
    rank, world = dist.get_rank(), dist.get_world_size()
    videos, captions = _collective_call(lambda: read_complete(settings.source_report, settings.source_report_sha256,
                                                             context.input_references))
    training = _collective_call(lambda: CaptionDataset(videos, captions['train'], temporal_offset=settings.factor_arm == 'temporal_offset', seed=settings.seed))
    validation = _collective_call(lambda: CaptionDataset(videos, captions['val']))
    recipe = settings.recipe(settings.method)
    identities = [r['official_video_id'] for r in captions['train']]
    from .factors import caption_batches
    sampler = caption_batches if settings.factor_arm == 'caption_shuffle' else unique_video_batches
    batches = [sampler(identities, global_batch=recipe['global_batch'], world_size=world,
                                   seed=settings.seed, epoch=epoch) for epoch in range(recipe['epochs'])]
    initial = _collective_call(lambda: initial_clip(settings))
    model, network = make_model(settings, settings.method, initial, device)
    optimizer = make_optimizer(model, learning_rate=recipe['learning_rate'], total_steps=sum(map(len, batches)))
    identity = checkpoint_identity(context, settings, settings.method)
    records, selected, update, start_epoch = [], None, 0, 0
    if settings.resume is not None:
        ref = settings.resume
        if file_sha256(ref.checkpoint.with_suffix('.json')) != ref.sidecar_sha256:
            raise ValueError('resume sidecar changed after authorization')
        with ref.checkpoint.with_suffix('.json').open() as stream: metadata = strict_json_load(stream)
        identity['source_sha'] = metadata['identity']['source_sha']  # Compatibility proven by reserve/start guards.
        progress = restore_state(ref.checkpoint, model, optimizer, identity)
        start_epoch, update = progress['epoch'], progress['update']
        if (progress['batch_cursor'] != 0 or not 1 <= start_epoch <= recipe['epochs']
                or update != sum(len(b) for b in batches[:start_epoch])):
            raise ValueError('resume cursor does not prove complete prior epochs')
        def recover_history():
            nonlocal selected
            if rank != 0: return
            parent_root = ref.checkpoint.parent.parent
            for epoch in range(1, start_epoch+1):
                path = parent_root/'provenance'/f'epoch-{epoch:02d}.json'
                with (parent_root/'status.json').open() as stream: status = strict_json_load(stream)
                if status['output_hashes'].get(path.relative_to(parent_root).as_posix()) != file_sha256(path):
                    raise ValueError('resume epoch history is not terminal-bound')
                with path.open() as stream: row = strict_json_load(stream)
                summary = row['metrics']['summary_metrics']
                if (row['epoch'] != epoch or summary['epoch'] != epoch or summary['train_caption_count'] != 164920
                        or summary['validation_caption_count'] != 10000 or summary['validation_video_count'] != 500
                        or summary['optimizer_updates'] != sum(len(b) for b in batches[:epoch])):
                    raise ValueError('resume epoch history coverage/step mismatch')
                historical = Path(row['checkpoint']['path'])
                if historical.parent != parent_root/'checkpoints':
                    raise ValueError('historical checkpoint is outside authenticated parent')
                for p, digest in ((historical, row['checkpoint']['sha256']),
                                  (historical.with_suffix('.json'), row['checkpoint']['sidecar_sha256'])):
                    safe_path(p)
                    if (status['output_hashes'].get(p.relative_to(parent_root).as_posix()) != digest
                            or file_sha256(p) != digest):
                        raise ValueError('historical selected checkpoint changed')
                carried = carry_checkpoint(historical, context.paths.checkpoints/historical.name, context.array_store)
                row['checkpoint']['path'] = str(carried)
                old_arrays = row['metrics']['array_references']
                row['metrics']['array_references'] = [context.array_store.write(a['key'], verify_array_reference(parent_root, a)).as_dict() for a in old_arrays]
                publish_metrics(context, row['metrics'])
                publish_json(context.paths.provenance/f'epoch-{epoch:02d}.json', row)
                records.append(row)
            selected = next(r for r in records if r['epoch'] == choose_epoch(records))
        _collective_call(recover_history)
    resumed_updates = update
    for epoch in range(start_epoch+1, recipe['epochs']+1):
        training.epoch = epoch-1
        epoch_batches = batches[epoch-1]
        losses = train_epoch(model, network, optimizer, training, epoch_batches, device, loader_workers=settings.loader_workers)
        update += len(epoch_batches)
        scores, result = evaluate_complete(model, validation, device)
        checkpoint = context.paths.checkpoints/f'epoch-{epoch:02d}.pt'
        save_state(checkpoint, model, optimizer, identity, dict(epoch=epoch, batch_cursor=0, update=update), context.array_store)
        summary = {f'{direction}_{name.lower()}': value for direction in ('t2v', 'v2t') for name, value in result[direction].items()}
        summary.update(epoch=epoch, optimizer_updates=update, train_caption_count=len(training),
                       validation_caption_count=len(validation), validation_video_count=500, training_mean_loss=float(losses[:, 0].mean()))
        def publish():
            nonlocal selected
            if rank == 0:
                arrays = [context.array_store.write(f'epoch_{epoch}_scores', scores).as_dict(),
                          context.array_store.write(f'epoch_{epoch}_t2v_ranks', result['t2v_ranks']).as_dict(),
                          context.array_store.write(f'epoch_{epoch}_v2t_ranks', result['v2t_ranks']).as_dict(),
                          context.array_store.write(f'epoch_{epoch}_global_training_losses', losses).as_dict()]
                metrics = dict(split='validation', summary_metrics=summary, array_references=arrays)
                publish_metrics(context, metrics)
                checkpoint_ref = dict(path=str(checkpoint), sha256=file_sha256(checkpoint),
                                      sidecar_sha256=file_sha256(checkpoint.with_suffix('.json')))
                row = dict(epoch=epoch, t2v_r1=summary['t2v_r1'], checkpoint=checkpoint_ref, metrics=metrics)
                publish_json(context.paths.provenance/f'epoch-{epoch:02d}.json', row)
                records.append(row)
                if choose_epoch(records) == epoch: selected = row
                print(f'epoch={epoch} raw_t2v_R1={summary["t2v_r1"]:.4f} raw_v2t_R1={summary["v2t_r1"]:.4f}', flush=True)
        _collective_call(publish)
    def finish():
        if rank == 0:
            publish_json(context.paths.metrics/'selected_validation.json', selected['metrics'])
            publish_json(context.paths.provenance/'baseline-result.json', dict(
                schema_version='full-data-baseline-result-v1', source_sha=context.config.source_sha,
                manifest_sha256=context.config.manifest_sha256, method=settings.method, seed=settings.seed,
                complete_epochs=recipe['epochs'], optimizer_updates=update, selected_epoch=selected['epoch'],
                resumed_epochs=start_epoch, updates_this_run=update-resumed_updates,
                resume_parent=None if settings.resume is None else settings.resume.parent_run_id,
                selected_checkpoint=selected['checkpoint'], validation_only=True, test_authorized=False,
                epoch_records=[dict(epoch=r['epoch'], t2v_r1=r['t2v_r1'], checkpoint=r['checkpoint']) for r in records],
                final_model_sha256=model_sha256(model)))
    _collective_call(finish)
