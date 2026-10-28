import copy
import datetime
import pytest
import torch
import torch.distributed as dist


def trace_wti(model):
    observations = []
    frame_tokens = []
    encode = model.encode_video
    def trace_encode(*args):
        output = encode(*args)
        output['tokens'].retain_grad()
        frame_tokens.append(output['tokens'])
        return output
    model.encode_video = trace_encode
    model._traced_frame_tokens = frame_tokens
    original = model.core.weighted_token_wise_intersection
    def trace(text, frames, text_mask, frame_mask):
        with torch.no_grad():
            pair = torch.einsum('atd,bvd->abtv', text, frames)
            pair = pair*text_mask[:, None, :, None]*frame_mask[None, :, None, :]
            observations.append((pair.argmax(-1), pair.argmax(-2)))
        return original(text, frames, text_mask, frame_mask)
    model.core.weighted_token_wise_intersection = trace
    return observations


class AuxiliaryOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, *args):
        output = self.model(*args)
        return output['aqs'], output['logits'].detach()


def _worker(rank, init_file, output, cfg, initial, batch, mode):
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.model import AqsBaselineModel
    import semantic_retrieval.aqs.model as adapter
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+init_file, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        config = AqsConfig(not (mode == 'config' and rank == 1), 2, .2)
        digest = ('b' if mode == 'mapping' and rank == 1 else 'a')*64
        clip_state = {k.removeprefix('core.clip.'): v for k, v in initial.items() if k.startswith('core.clip.')}
        try:
            model = AqsBaselineModel(cfg, clip_state, aqs=config, id_mapping_sha256=digest, temporal_layers=1)
            model.load_state_dict(initial, strict=True)
            local = [x[rank*2:rank*2+2].clone() for x in batch]
            if mode == 'duplicate' and rank == 1: local[-1][0] = 0
            if mode == 'input' and rank == 1: local[-1] = local[-1].float()
            if mode == 'training':
                from semantic_retrieval.aqs.training import aqs_step, save_aqs_state, restore_aqs_state, CheckpointBinding, model_recipe
                from semantic_retrieval.full_data.training import make_optimizer
                from semantic_retrieval.full_data.qualification import optimizer_fingerprint
                from semantic_retrieval.runtime.checkpoint import rng_state, model_sha256
                from experiment_artifacts import ArtifactArrayStore, artifact_json_bytes
                import hashlib
                from pathlib import Path
                recipe = dict(model=model_recipe(model), sampler=dict(seed=713, policy='unique_video_all_captions_v1'))
                identity = dict(source_sha='a'*40, input_sha256='b'*64, method='uatvr', world_size=2,
                                recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest())
                binding = CheckpointBinding(identity, recipe, digest)
                optimizer = make_optimizer(model, learning_rate=5e-5, total_steps=10)
                ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
                first = aqs_step(model, ddp, optimizer, local)
                root = Path(output).parent/'resume'
                (root/'checkpoints').mkdir(parents=True, exist_ok=True)
                checkpoint = root/'checkpoints/step1.pt'
                save_aqs_state(checkpoint, model, optimizer, binding, dict(epoch=0, batch_cursor=1, update=1), ArtifactArrayStore(root/'arrays'))
                second = aqs_step(model, ddp, optimizer, local)
                expected = (model_sha256(model), optimizer_fingerprint(optimizer.state_dict()), optimizer_fingerprint(rng_state()))
                fresh = AqsBaselineModel(cfg, clip_state, aqs=config, id_mapping_sha256=digest, temporal_layers=1)
                fresh_optimizer = make_optimizer(fresh, learning_rate=5e-5, total_steps=10)
                fresh_ddp = torch.nn.parallel.DistributedDataParallel(fresh, find_unused_parameters=True)
                restore_aqs_state(checkpoint, fresh, fresh_optimizer, binding)
                repeated = aqs_step(fresh, fresh_ddp, fresh_optimizer, local)
                assert repeated == second
                assert (model_sha256(fresh), optimizer_fingerprint(fresh_optimizer.state_dict()), optimizer_fingerprint(rng_state())) == expected
                torch.save(dict(trained=True, first=first, second=second), output+f'-{rank}.pt')
                return
            traced = trace_wti(model)
            selected = []
            original = adapter.select_negatives
            def capture(*args, **kwargs):
                result = original(*args, **kwargs); selected.append(result.selected_ids); return result
            adapter.select_negatives = capture
            ddp = torch.nn.parallel.DistributedDataParallel(AuxiliaryOnly(model), find_unused_parameters=True)
            torch.manual_seed(101)
            loss, logits = ddp(*local)
            loss.backward()
            if mode != 'normal': raise AssertionError('expected synchronized refusal')
            torch.save(dict(loss=loss.detach(), logits=logits, frame_grad=model._traced_frame_tokens[0].grad, frame_value=model._traced_frame_tokens[0].detach(), traced=traced[0], selected=selected[0],
                            gradients={k:p.grad for k,p in model.named_parameters() if p.grad is not None}), output+f'-{rank}.pt')
        except (ValueError, RuntimeError) as exc:
            if mode in ('normal', 'training'): raise
            expected = dict(config='configuration', mapping='mapping', duplicate='unique', input='contract')[mode]
            assert expected in str(exc), str(exc)
            torch.save(dict(refused=True), output+f'-{rank}.pt')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('mode', ['normal', 'training', 'duplicate', 'mapping', 'config', 'input'])
def test_global_auxiliary_gradients_or_collective_refusal(tmp_path, tiny_clip, paired_batch, mode, monkeypatch):
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.model import AqsBaselineModel
    model = AqsBaselineModel(tiny_clip.config, tiny_clip.state_dict(), aqs=AqsConfig(True, 2, .2),
                             id_mapping_sha256='a'*64, temporal_layers=1)
    initial = copy.deepcopy(model.state_dict())
    batch = tuple(torch.cat([x,x]) for x in paired_batch[:-1])+(torch.arange(4),)
    batch[0][2, 1] = 8; batch[0][3, 1] = 9
    batch[2][2:] += .1
    import semantic_retrieval.aqs.model as adapter
    captured = []
    original = adapter.select_negatives
    def capture(*args, **kwargs):
        result = original(*args, **kwargs); captured.append(result.selected_ids); return result
    monkeypatch.setattr(adapter, 'select_negatives', capture)
    traced = trace_wti(model)
    torch.manual_seed(101)
    output = model(*batch)
    output['aqs'].backward()
    expected = {k:p.grad for k,p in model.named_parameters() if p.grad is not None}
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path/'group'), str(tmp_path/'result'),
        tiny_clip.config, initial, batch, mode), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path/f'result-{rank}.pt', weights_only=True)
        if mode == 'training':
            assert actual['trained'] and actual['second']['aqs_active'] == 1; continue
        if mode != 'normal':
            assert actual['refused']; continue
        torch.testing.assert_close(actual['logits'], output['logits'], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual['loss'], output['aqs'], atol=1e-6, rtol=1e-6)
        assert all(torch.equal(x,y) for x,y in zip(actual['traced'],traced[0])), [(x!=y).sum().item() for x,y in zip(actual['traced'],traced[0])]
        assert torch.equal(actual['selected'], captured[0]), (actual['selected'], captured[0])
        torch.testing.assert_close(actual['frame_value'], model._traced_frame_tokens[0].detach()[rank*2:rank*2+2], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual['frame_grad']/2, model._traced_frame_tokens[0].grad[rank*2:rank*2+2], atol=2e-6, rtol=2e-4)
        assert actual['gradients'].keys() == expected.keys()
        for name, gradient in actual['gradients'].items():
            torch.testing.assert_close(gradient, expected[name], atol=2e-6, rtol=2e-4, msg=name)
    if mode == 'normal':
        a = torch.load(tmp_path/'result-0.pt', weights_only=True)
        b = torch.load(tmp_path/'result-1.pt', weights_only=True)
        assert torch.equal(a['selected'], b['selected'])
