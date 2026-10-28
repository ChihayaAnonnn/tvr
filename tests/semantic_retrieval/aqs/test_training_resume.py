import copy
from dataclasses import replace
import hashlib
import json
import random
import numpy as np
import pytest
import torch
from experiment_artifacts import ArtifactArrayStore, artifact_json_bytes


def make_model(clip, config=None, digest=None):
    from semantic_retrieval.aqs.model import AqsBaselineModel
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.aqs.training import video_id_mapping
    _, mapping = video_id_mapping(['v1', 'v0'])
    return AqsBaselineModel(clip.config, clip.state_dict(), aqs=config or AqsConfig(True, 2, .2),
                           id_mapping_sha256=digest or mapping, temporal_layers=1)


def binding_for(model):
    from semantic_retrieval.aqs.training import model_recipe, CheckpointBinding
    recipe = dict(model=model_recipe(model), sampler=dict(policy='unique_video_all_captions_v1', seed=713))
    identity = dict(source_sha='a'*40, recipe_sha256=hashlib.sha256(artifact_json_bytes(recipe)).hexdigest(),
                    input_sha256='b'*64, method='uatvr', world_size=1)
    return CheckpointBinding(identity, recipe, model.id_mapping_sha256)


def test_real_step_and_exact_restore(tmp_path, tiny_clip, paired_batch, monkeypatch):
    from semantic_retrieval.aqs.training import aqs_step, save_aqs_state, restore_aqs_state
    from semantic_retrieval.full_data.training import make_optimizer
    from semantic_retrieval.full_data.qualification import optimizer_fingerprint
    from semantic_retrieval.runtime.checkpoint import rng_state
    import semantic_retrieval.aqs.model as adapter
    selected = []
    original = adapter.select_negatives
    def capture(*args, **kwargs):
        result = original(*args, **kwargs); selected.append(result.selected_ids.clone()); return result
    monkeypatch.setattr(adapter, 'select_negatives', capture)
    random.seed(713); np.random.seed(713)
    model = make_model(tiny_clip)
    binding = binding_for(model)
    optimizer = make_optimizer(model, learning_rate=5e-5, total_steps=10)
    before = model.clip.text_projection.detach().clone()
    frozen = model.clip.visual.conv1.weight.detach().clone()
    first = aqs_step(model, model, optimizer, paired_batch)
    assert first['aqs_active'] == 1 and first['aqs_weighted'] == pytest.approx(.2*first['aqs'])
    (tmp_path/'checkpoints').mkdir()
    store = ArtifactArrayStore(tmp_path/'arrays')
    path = tmp_path/'checkpoints/step1.pt'
    save_aqs_state(path, model, optimizer, binding, dict(epoch=0, batch_cursor=1, update=1), store)
    recipe_record = json.loads(path.with_suffix('.aqs.json').read_text())
    assert recipe_record['recipe']['aqs'] == model.aqs_config.recipe()
    assert recipe_record['recipe']['id_mapping_sha256'] == model.id_mapping_sha256
    expected = aqs_step(model, model, optimizer, paired_batch)
    state = copy.deepcopy(model.state_dict())
    opt = optimizer_fingerprint(optimizer.state_dict()); rng = optimizer_fingerprint(rng_state())
    expected_ids = selected[-1].clone()
    assert not torch.equal(before, model.clip.text_projection)
    assert torch.equal(frozen, model.clip.visual.conv1.weight)
    fresh = make_model(tiny_clip)
    fresh_optimizer = make_optimizer(fresh, learning_rate=5e-5, total_steps=10)
    assert restore_aqs_state(path, fresh, fresh_optimizer, binding) == dict(epoch=0, batch_cursor=1, update=1)
    assert aqs_step(fresh, fresh, fresh_optimizer, paired_batch) == expected
    assert torch.equal(selected[-1], expected_ids)
    for key, value in state.items(): torch.testing.assert_close(fresh.state_dict()[key], value, atol=0, rtol=0)
    assert optimizer_fingerprint(fresh_optimizer.state_dict()) == opt
    assert optimizer_fingerprint(rng_state()) == rng
    with pytest.raises(FileExistsError):
        save_aqs_state(path, fresh, fresh_optimizer, binding, dict(epoch=0, batch_cursor=1, update=1), store)


@pytest.mark.parametrize('defect', ['negatives', 'weight', 'mapping', 'source', 'world', 'recipe', 'model'])
def test_restore_mismatch_rejected_before_mutation(tmp_path, tiny_clip, paired_batch, defect):
    from semantic_retrieval.aqs.training import aqs_step, save_aqs_state, restore_aqs_state, CheckpointBinding
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.full_data.training import make_optimizer
    from semantic_retrieval.full_data.qualification import optimizer_fingerprint
    from semantic_retrieval.runtime.checkpoint import rng_state, model_sha256
    model = make_model(tiny_clip); binding = binding_for(model)
    optimizer = make_optimizer(model, learning_rate=5e-5, total_steps=10)
    aqs_step(model, model, optimizer, paired_batch)
    (tmp_path/'checkpoints').mkdir(); path = tmp_path/'checkpoints/one.pt'
    save_aqs_state(path, model, optimizer, binding, dict(epoch=0, batch_cursor=1, update=1), ArtifactArrayStore(tmp_path/'arrays'))
    if defect in {'negatives', 'weight'}:
        model.aqs_config = AqsConfig(True, 3 if defect == 'negatives' else 2, .3 if defect == 'weight' else .2)
    if defect == 'mapping': model.id_mapping_sha256 = 'c'*64
    if defect == 'model': model.batching = 'caption_shuffle_v1'
    if defect in {'source', 'world', 'recipe'}:
        ident, recipe = binding.base_identity, binding.base_recipe
        if defect == 'source': ident['source_sha'] = 'd'*40
        if defect == 'world': ident['world_size'] = 2
        if defect == 'recipe':
            recipe['sampler']['seed'] = 999
            ident['recipe_sha256'] = hashlib.sha256(artifact_json_bytes(recipe)).hexdigest()
        binding = CheckpointBinding(ident, recipe, binding.id_mapping_sha256)
    before = (model_sha256(model), optimizer_fingerprint(optimizer.state_dict()), optimizer_fingerprint(rng_state()))
    with pytest.raises(ValueError): restore_aqs_state(path, model, optimizer, binding)
    after = (model_sha256(model), optimizer_fingerprint(optimizer.state_dict()), optimizer_fingerprint(rng_state()))
    assert before == after


def test_binding_immutable_mapping_and_complete_sampler(tiny_clip):
    from semantic_retrieval.aqs.training import video_id_mapping, CheckpointBinding
    from semantic_retrieval.full_data.training import unique_video_batches
    model = make_model(tiny_clip); binding = binding_for(model)
    old = binding.base_recipe
    old['sampler']['seed'] = 4
    assert binding.base_recipe['sampler']['seed'] == 713
    mapping, digest = video_id_mapping(['v2', 'v0', 'v1'])
    assert mapping == {'v0':0, 'v1':1, 'v2':2} and len(digest) == 64
    with pytest.raises(ValueError): video_id_mapping(['v0', 'v0'])
    ids = [f'v{i}' for i in range(13) for _ in range(4)]
    batches = unique_video_batches(ids, global_batch=8, world_size=2, seed=713, epoch=0)
    assert sorted(i for b in batches for i in b) == list(range(len(ids)))
    assert all(len(b)%2 == 0 and len({ids[i] for i in b}) == len(b) for b in batches)
    assert batches[1:] == unique_video_batches(ids, global_batch=8, world_size=2, seed=713, epoch=0)[1:]


def test_disabled_step_reports_not_run(tiny_clip, paired_batch):
    from semantic_retrieval.aqs.training import aqs_step
    from semantic_retrieval.aqs.config import AqsConfig
    from semantic_retrieval.full_data.training import make_optimizer
    model = make_model(tiny_clip, AqsConfig(False, 2, .2))
    result = aqs_step(model, model, make_optimizer(model, learning_rate=5e-5, total_steps=10), paired_batch)
    assert result['aqs_active'] == 0 and result['aqs'] == 0 and result['mean_negatives'] == 0


def test_step_rejects_foreign_optimizer_before_update(tiny_clip, paired_batch):
    from semantic_retrieval.aqs.training import aqs_step
    from semantic_retrieval.full_data.training import make_optimizer
    model, other = make_model(tiny_clip), make_model(tiny_clip)
    optimizer = make_optimizer(other, learning_rate=5e-5, total_steps=10)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='optimizer'):
        aqs_step(model, model, optimizer, paired_batch)
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('defect', ['missing', 'weight', 'extra'])
def test_checkpoint_recipe_sidecar_tamper_refused_before_mutation(tmp_path, tiny_clip, paired_batch, defect):
    from semantic_retrieval.aqs.training import aqs_step, save_aqs_state, restore_aqs_state
    from semantic_retrieval.full_data.training import make_optimizer
    from semantic_retrieval.full_data.qualification import optimizer_fingerprint
    from semantic_retrieval.runtime.checkpoint import rng_state, model_sha256
    model = make_model(tiny_clip); binding = binding_for(model)
    optimizer = make_optimizer(model, learning_rate=5e-5, total_steps=10)
    aqs_step(model, model, optimizer, paired_batch)
    (tmp_path/'checkpoints').mkdir(); path = tmp_path/'checkpoints/one.pt'
    save_aqs_state(path, model, optimizer, binding, dict(epoch=0, batch_cursor=1, update=1), ArtifactArrayStore(tmp_path/'arrays'))
    sidecar = path.with_suffix('.aqs.json')
    if defect == 'missing': sidecar.unlink()
    else:
        recipe = json.loads(sidecar.read_text())
        if defect == 'weight': recipe['recipe']['aqs']['weight'] = .9
        if defect == 'extra': recipe['extra'] = 1
        sidecar.write_text(json.dumps(recipe))
    before = (model_sha256(model), optimizer_fingerprint(optimizer.state_dict()), optimizer_fingerprint(rng_state()))
    with pytest.raises((ValueError, FileNotFoundError)):
        restore_aqs_state(path, model, optimizer, binding)
    assert (model_sha256(model), optimizer_fingerprint(optimizer.state_dict()), optimizer_fingerprint(rng_state())) == before
