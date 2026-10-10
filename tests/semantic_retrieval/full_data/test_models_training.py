import copy

import numpy as np
import pytest
import torch

from semantic_retrieval.models.clip import ClipBackbone, ClipConfig


@pytest.fixture
def clip():
    torch.manual_seed(713)
    return ClipBackbone(ClipConfig(64, 16, 64, 1, 4, 32, 128, 64, 1, 1, fixture=True))


@pytest.fixture
def batch():
    ids = torch.tensor([[126, 4, 127, 0, 0, 0, 0, 0], [126, 5, 6, 127, 0, 0, 0, 0]])
    return ids, ids != 0, torch.randn(2, 3, 3, 16, 16), torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool), torch.arange(2)


@pytest.mark.parametrize('method', ['clip4clip', 'uatvr'])
def test_actual_adapter_forward_backward_strict_clip_and_intended_updates(clip, batch, method):
    from semantic_retrieval.full_data.models import BaselineModel
    model = BaselineModel(method, clip.config, clip.state_dict(), temporal_layers=1)
    for name, value in clip.state_dict().items():
        torch.testing.assert_close(model.clip.state_dict()[name], value, rtol=0, atol=0)
    output = model(*batch)
    assert output['logits'].shape == (2, 2) and torch.isfinite(output['loss'])
    output['loss'].backward()
    assert model.clip.visual.conv1.weight.grad is None  # Official freeze_layer_num=0.
    assert not model.clip.token_embedding.weight.requires_grad
    assert model.clip.text_projection.grad.abs().sum() > 0
    assert model.clip.visual.proj.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if method == 'uatvr':
        assert model.core.uncertain_net_text.fc.weight.grad.abs().sum() > 0
        assert model.core.pie_net_video.fc.weight.grad.abs().sum() > 0
        assert model.core.frame_position_embeddings.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        text = model.encode_text(*batch[:2])
        video = model.encode_video(*batch[2:4])
        rng = torch.get_rng_state().clone()
        score = model.score(text, video)
        assert torch.equal(torch.get_rng_state(), rng)
        assert score.shape == (2, 2) and torch.isfinite(score).all()
        torch.testing.assert_close(model.score(text, video), score, rtol=0, atol=0)


def test_meanp_matches_official_per_frame_normalization_formula(clip, batch):
    from semantic_retrieval.full_data.models import BaselineModel
    model = BaselineModel('clip4clip', clip.config, clip.state_dict()).eval()
    text, mask, video, frames, _ = batch
    with torch.no_grad():
        _, eos = clip.encode_text(text, mask)
        raw = clip.encode_frames(video, frames)
        normalized = torch.nn.functional.normalize(raw, dim=-1)
        pooled = (normalized*frames[..., None]).sum(1)/frames.sum(1)[:, None]
        expected = clip.logit_scale.exp()*torch.nn.functional.normalize(eos, dim=-1) @ torch.nn.functional.normalize(pooled, dim=-1).T
        actual = model.score(model.encode_text(text, mask), model.encode_video(video, frames))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('method', ['clip4clip', 'uatvr'])
def test_adapter_rejects_false_negative_duplicates_and_corrupt_input_weights(clip, batch, method):
    from semantic_retrieval.full_data.models import BaselineModel
    model = BaselineModel(method, clip.config, clip.state_dict(), temporal_layers=1)
    with pytest.raises(ValueError, match='unique'):
        model(*batch[:-1], torch.tensor([1, 1]))
    broken = dict(clip.state_dict())
    broken.pop('text_projection')
    with pytest.raises(ValueError, match='CLIP'):
        BaselineModel(method, clip.config, broken, temporal_layers=1)


def test_unique_video_batches_cover_every_caption_once_and_restore_by_cursor():
    from semantic_retrieval.full_data.training import unique_video_batches
    identities = [f'v{v}' for v in range(13) for _ in range(4)]
    batches = unique_video_batches(identities, global_batch=8, world_size=4, seed=713, epoch=0)
    assert sorted(i for b in batches for i in b) == list(range(len(identities)))
    assert all(len(b) % 4 == 0 and len({identities[i] for i in b}) == len(b) for b in batches)
    assert batches == unique_video_batches(identities, global_batch=8, world_size=4, seed=713, epoch=0)
    assert batches != unique_video_batches(identities, global_batch=8, world_size=4, seed=713, epoch=1)
    with pytest.raises(ValueError):
        unique_video_batches(['v0']*5, global_batch=8, world_size=4, seed=713, epoch=0)


def test_upstream_optimizer_freeze_groups_warmup_and_nonzero_update(clip, batch):
    from semantic_retrieval.full_data.models import BaselineModel
    from semantic_retrieval.full_data.training import make_optimizer
    model = BaselineModel('uatvr', clip.config, clip.state_dict(), temporal_layers=1)
    optimizer = make_optimizer(model, learning_rate=5e-5, total_steps=10)
    clip_ids = {id(p) for p in model.clip.parameters()}
    for group in optimizer.param_groups:
        assert all(p.requires_grad for p in group['params'])
        assert group['lr'] == pytest.approx(5e-8 if id(group['params'][0]) in clip_ids else 5e-5, rel=1e-15)
        assert group['schedule'] == 'warmup_cosine' and group['warmup'] == 0.1
        assert group['b1'] == 0.9 and group['b2'] == 0.98 and group['e'] == 1e-6
    before = model.clip.text_projection.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(); model(*batch)['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    assert not torch.equal(before, model.clip.text_projection)


def _ddp_worker(rank, init_file, output, method, cfg, initial, batch):
    import datetime
    from semantic_retrieval.full_data.models import BaselineModel
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://'+init_file, rank=rank, world_size=2,
                                         timeout=datetime.timedelta(seconds=45))
    try:
        model = BaselineModel(method, cfg, {k.removeprefix('core.clip.' if method == 'uatvr' else 'core.'): v
                                          for k, v in initial.items() if k.startswith('core.clip.' if method == 'uatvr' else 'core.')}, temporal_layers=1)
        model.load_state_dict(initial, strict=True)
        ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        torch.manual_seed(101)
        output_loss = ddp(*(x[rank*2:rank*2+2] for x in batch))['loss']
        output_loss.backward()
        torch.save({k: p.grad for k, p in model.named_parameters() if p.grad is not None}, output+f'-{rank}.pt')
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize('method', ['clip4clip', 'uatvr'])
def test_two_rank_cpu_gradients_match_serial_global_objective(tmp_path, clip, batch, method):
    from semantic_retrieval.full_data.models import BaselineModel
    model = BaselineModel(method, clip.config, clip.state_dict(), temporal_layers=1)
    initial = copy.deepcopy(model.state_dict())
    global_batch = tuple(torch.cat([x, x], dim=0) for x in batch[:-1]) + (torch.arange(4),)
    torch.manual_seed(101)
    model(*global_batch)['loss'].backward()
    expected = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path/'gloo'), str(tmp_path/'gradient'), method,
                                                 clip.config, initial, global_batch), nprocs=2, join=True)
    for rank in range(2):
        gradients = torch.load(tmp_path/f'gradient-{rank}.pt', weights_only=True)
        assert gradients.keys() == expected.keys()
        for name, value in gradients.items():
            torch.testing.assert_close(value, expected[name], atol=2e-6, rtol=2e-4, msg=name)


def test_checkpoint_exact_rng_optimizer_resume_and_identity_tamper(tmp_path):
    from semantic_retrieval.full_data.training import save_state, restore_state
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.runtime.distributed import configure_fp32
    configure_fp32(); torch.manual_seed(713); np.random.seed(713)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    identity = dict(source_sha='a'*40, recipe_sha256='b'*64, input_sha256='c'*64, method='clip4clip', world_size=1)
    def step():
        optimizer.zero_grad()
        output = model(torch.randn(4, 3))
        (output.square().mean()*float(np.random.uniform())).backward()
        optimizer.step()
        return output.detach().clone()
    step()
    (tmp_path/'checkpoints').mkdir()
    store = ArtifactArrayStore(tmp_path/'arrays')
    path = tmp_path/'checkpoints/step-1.pt'
    save_state(path, model, optimizer, identity, dict(epoch=0, batch_cursor=1, update=1), store)
    expected = step(); expected_state = copy.deepcopy(model.state_dict())
    progress = restore_state(path, model, optimizer, identity)
    assert progress == dict(epoch=0, batch_cursor=1, update=1)
    torch.testing.assert_close(step(), expected, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected_state[name], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_state(path, model, optimizer, identity, progress, store)
    with pytest.raises(ValueError, match='identity'):
        restore_state(path, model, optimizer, dict(identity, input_sha256='d'*64))
    with path.open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='hash'):
        restore_state(path, model, optimizer, identity)


def test_checkpoint_carry_is_self_contained_across_two_successors(tmp_path):
    import shutil
    from semantic_retrieval.full_data.training import save_state, restore_state, carry_checkpoint
    from experiment_artifacts import ArtifactArrayStore
    model = torch.nn.Linear(513, 513)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 513)).sum().backward(); optimizer.step()
    identity = dict(source_sha='a'*40, recipe_sha256='b'*64, input_sha256='c'*64, method='clip4clip', world_size=1)
    original = tmp_path/'first/checkpoints/epoch-01.pt'; original.parent.mkdir(parents=True)
    save_state(original, model, optimizer, identity, dict(epoch=1, batch_cursor=0, update=1), ArtifactArrayStore(tmp_path/'first/arrays'))
    previous = original
    for name in ('second', 'third'):
        target = tmp_path/name/'checkpoints/epoch-01.pt'; target.parent.mkdir(parents=True)
        carry_checkpoint(previous, target, ArtifactArrayStore(tmp_path/name/'arrays'))
        shutil.rmtree(previous.parent.parent)
        assert restore_state(target, model, optimizer, identity)['epoch'] == 1
        previous = target


@pytest.mark.parametrize('defect', ['nonfinite', 'shape', 'param_order'])
def test_checkpoint_rejects_invalid_optimizer_before_mutating_model(tmp_path, defect):
    import hashlib
    import json
    from semantic_retrieval.full_data.training import save_state, restore_state
    from experiment_artifacts import ArtifactArrayStore
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(torch.ones(1, 3)).sum().backward(); optimizer.step()
    identity = dict(source_sha='a'*40, recipe_sha256='b'*64, input_sha256='c'*64, method='clip4clip', world_size=1)
    (tmp_path/'checkpoints').mkdir()
    path = tmp_path/'checkpoints/step-1.pt'
    save_state(path, model, optimizer, identity, dict(epoch=0, batch_cursor=1, update=1), ArtifactArrayStore(tmp_path/'arrays'))
    payload = torch.load(path, weights_only=True)
    if defect == 'nonfinite':
        payload['optimizer']['state'][0]['exp_avg'][0, 0] = float('nan')
    elif defect == 'shape':
        payload['optimizer']['state'][0]['exp_avg'] = torch.ones(1)
    else:
        payload['optimizer']['param_groups'][0]['params'].reverse()
    torch.save(payload, path)
    meta = json.loads(path.with_suffix('.json').read_bytes())
    meta['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.json').write_text(json.dumps(meta))
    with torch.no_grad():
        model.weight.add_(3)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='optimizer'):
        restore_state(path, model, optimizer, identity)
    for k, v in before.items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
