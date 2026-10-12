import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def source_fixture():
    ids = ['a', 'b', 'c', 'v', 'sealed']
    videos = [dict(video_id=i, url='https://www.youtube.com/watch?v='+str(n).zfill(11),
                   **{'start time': 0, 'end time': 2}) for n, i in enumerate(ids)]
    videos[1]['url'] = videos[0]['url']
    sentences = [dict(video_id=i, sen_id=f'{i}-{n}', caption=f'{i} caption {n}')
                 for i in ids for n in range(3)]
    raw = json.dumps(dict(videos=videos, sentences=sentences)).encode()
    content = {i: hashlib.sha256(i.encode()).hexdigest() for i in ids[:-1]}
    content['v'] = content['b']  # Transitive source -> content -> validation.
    trusted = dict(train_video_ids=['a', 'b', 'c'], val_video_ids=['v'])
    return raw, set(ids[:-1]), trusted, content


def test_compiler_all_captions_transitive_quarantine_and_sealed_values():
    from semantic_retrieval.full_data.data import compile_sources
    raw, allowed, trusted, content = source_fixture()
    decoded = []
    def decode(value):
        assert b'sealed' not in value
        decoded.append(value)
        return json.loads(value)
    result = compile_sources(raw, allowed, trusted, content, decode_caption=decode)
    assert result['excluded_train_ids'] == ['a', 'b']
    assert result['train_video_ids'] == ['c']
    assert result['validation_video_ids'] == ['v']
    assert len(result['captions']) == 6
    assert {r['official_video_id'] for r in result['captions']} == {'c', 'v'}
    assert len(decoded) == 12 and result['privacy']['decoded_excluded_caption_values'] == 0
    assert result['components']['a'] == result['components']['b'] == result['components']['v']


def test_compiler_rejects_missing_captions_or_mismatched_inventory():
    from semantic_retrieval.full_data.data import compile_sources
    raw, allowed, trusted, content = source_fixture()
    parsed = json.loads(raw)
    parsed['sentences'] = [r for r in parsed['sentences'] if r['video_id'] != 'c']
    with pytest.raises(ValueError, match='caption coverage'):
        compile_sources(json.dumps(parsed).encode(), allowed, trusted, content)
    with pytest.raises(ValueError, match='membership'):
        compile_sources(raw, allowed | {'sealed'}, trusted, content)


def test_bidirectional_ranks_use_best_relevant_caption_and_canonical_ties():
    from semantic_retrieval.full_data.metrics import retrieval_metrics
    scores = np.array([[1., 1.], [0.8, 0.9], [0.7, 0.6], [0.5, 0.4]])
    ids, positives, gallery = ['q0', 'q1', 'q2', 'q3'], ['a', 'a', 'b', 'b'], ['a', 'b']
    result = retrieval_metrics(scores, positives, gallery, query_ids=ids)
    np.testing.assert_array_equal(result['t2v_ranks'], [1, 2, 2, 2])
    np.testing.assert_array_equal(result['v2t_ranks'], [1, 3])
    assert result['t2v']['R1'] == 25 and result['v2t']['R1'] == 50
    assert result['v2t']['MnR'] == 2
    qi, vi = [3, 1, 0, 2], [1, 0]
    other = retrieval_metrics(scores[np.ix_(qi, vi)], [positives[i] for i in qi],
                              [gallery[i] for i in vi], query_ids=[ids[i] for i in qi])
    np.testing.assert_array_equal(other['t2v_ranks'], result['t2v_ranks'][qi])
    np.testing.assert_array_equal(other['v2t_ranks'], result['v2t_ranks'][vi])
    assert other['t2v'] == result['t2v'] and other['v2t'] == result['v2t']


def test_metrics_reject_nonfinite_incomplete_duplicate_and_ambiguous_ids():
    from semantic_retrieval.full_data.metrics import retrieval_metrics
    for scores, positives, gallery, queries in [
        ([[float('nan')]], ['a'], ['a'], ['q']),
        ([[1, 2]], ['a'], ['a', 'b'], ['q']),  # No caption for video b.
        ([[1, 2]], ['a'], ['a', 'a'], ['q']),
        ([[1], [2]], ['a', 'a'], ['a'], ['q', 'q']),
        ([[1]], ['missing'], ['a'], ['q']),
    ]:
        with pytest.raises(ValueError):
            retrieval_metrics(np.array(scores), positives, gallery, query_ids=queries)


def test_complete_export_decodes_real_short_videos_and_roundtrips_all_records(tmp_path):
    import cv2
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.full_data.data import compile_sources
    from semantic_retrieval.full_data.preparation import export_complete
    raw, allowed, trusted, _ = source_fixture()
    content = {}
    videos = tmp_path/'videos'
    videos.mkdir()
    for n, i in enumerate(sorted(allowed)):
        path = videos/(i+'.mp4')
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (16, 16))
        assert writer.isOpened()
        for frame in range(3):
            writer.write(np.full((16, 16, 3), n*35+frame*12, np.uint8))
        writer.release()
        content[i] = file_sha256(path)
    (videos/'v.mp4').write_bytes((videos/'b.mp4').read_bytes())
    content['v'] = content['b']
    result = compile_sources(raw, allowed, trusted, content)
    (tmp_path/'provenance').mkdir()
    out = tmp_path/'provenance/inputs'
    exported = export_complete(out, result, videos, image_size=16)
    assert exported['decoded_video_count'] == 2
    assert exported['caption_count'] == 6
    records = json.loads((out/'videos-000.json').read_bytes())['rows']
    assert {r['split'] for r in records} == {'train', 'val'}
    assert all(r['frame_count'] == 3 and r['fps'] == 5 for r in records)
    captions = json.loads((out/'captions-000.json').read_bytes())['rows']
    assert captions == result['captions']
    from semantic_retrieval.data.video import decode_video
    standard, valid = decode_video(videos/'c.mp4', [0, 2], 16)
    bounded, bounded_valid = decode_video(videos/'c.mp4', [0, 2], 16, decoder_threads=1)
    np.testing.assert_array_equal(standard.numpy(), bounded.numpy())
    np.testing.assert_array_equal(valid.numpy(), bounded_valid.numpy())
    from semantic_retrieval.evidence.frame_cache import publish_json
    from semantic_retrieval.full_data.loader import read_complete, CaptionDataset
    from semantic_retrieval.runtime.input_bindings import manifest_input_references
    from semantic_retrieval.full_data.models import BaselineModel
    from semantic_retrieval.models.clip import ClipConfig, ClipBackbone
    import torch
    report = tmp_path/'provenance/complete-preparation.json'
    publish_json(report, exported)
    refs = manifest_input_references(tmp_path, dict(input_checkpoints=[], dataset_splits=exported['input_indices']))
    refs.update({str(report): file_sha256(report), **{str(videos/(i+'.mp4')): h for i, h in content.items()}})
    loaded, splits = read_complete(report, file_sha256(report), refs, expected_train=1, expected_val=1, captions_per_video=3)
    dataset = CaptionDataset(loaded, splits['train']+splits['val'], frames=3, image_size=16, text_length=8)
    left, right = dataset[0], dataset[3]
    batch = tuple(torch.stack([left[i], right[i]]) for i in range(5))
    assert batch[3].sum().item() == 2  # Actual 1 FPS decode + padding of two short videos.
    base = ClipBackbone(ClipConfig(64, 16, 64, 1, 4, 32, 49408, 64, 1, 1, fixture=True))
    model = BaselineModel('clip4clip', base.config, base.state_dict())
    loss = model(*batch)['loss']; loss.backward()
    assert torch.isfinite(loss) and model.clip.text_projection.grad.abs().sum() > 0
    from semantic_retrieval.full_data.baseline import train_epoch, evaluate_complete
    from semantic_retrieval.full_data.training import make_optimizer, unique_video_batches
    batches = unique_video_batches([r['official_video_id'] for r in dataset.captions],
                                  global_batch=2, world_size=1, seed=713, epoch=0)
    optimizer = make_optimizer(model, learning_rate=1e-4, total_steps=len(batches))
    import copy
    original_model = copy.deepcopy(model.state_dict())
    before_training_rng = torch.get_rng_state().clone()
    losses = train_epoch(model, model, optimizer, dataset, batches, torch.device('cpu'))
    serial_rng = torch.get_rng_state().clone()
    serial_model = copy.deepcopy(model.state_dict())
    model.load_state_dict(original_model, strict=True)
    optimizer = make_optimizer(model, learning_rate=1e-4, total_steps=len(batches))
    torch.set_rng_state(before_training_rng)
    parallel_losses = train_epoch(model, model, optimizer, dataset, batches, torch.device('cpu'), loader_workers=4)
    np.testing.assert_array_equal(parallel_losses, losses)
    assert torch.equal(torch.get_rng_state(), serial_rng)
    assert all(torch.equal(value, serial_model[key]) for key, value in model.state_dict().items())
    assert losses.shape == (3, 4) and np.isfinite(losses).all()
    validation = CaptionDataset(loaded, splits['val'], frames=3, image_size=16, text_length=8)
    scores, measured = evaluate_complete(model, validation, torch.device('cpu'))
    assert scores.shape == (3, 1) and measured['t2v']['R1'] == measured['v2t']['R1'] == 100
    loaded['v']['split'] = 'test'
    with pytest.raises(ValueError): evaluate_complete(model, validation, torch.device('cpu'))
    test_scores, test_metrics = evaluate_complete(model, validation, torch.device('cpu'), expected_split='test')
    np.testing.assert_array_equal(test_scores, scores)
    assert test_metrics['t2v']['R1'] == test_metrics['v2t']['R1'] == 100
    loaded['v']['split'] = 'val'
    # Even a re-authenticated source shard must satisfy its cross-record contract.
    source = out/'sources-000.json'
    payload = json.loads(source.read_bytes())
    for row in payload['rows']:
        if row['video_id'] == 'c': row['source_component'] = '0'*64
    source.write_text(json.dumps(payload)); refs[str(source)] = file_sha256(source)
    index_path = Path(exported['input_indices'][0]['path'])
    index = json.loads(index_path.read_bytes())
    for row in index['files']:
        if row['path'] == str(source): row['sha256'] = refs[str(source)]
    index_path.write_text(json.dumps(index)); refs[str(index_path)] = file_sha256(index_path)
    exported['input_indices'][0]['sha256'] = refs[str(index_path)]
    report.write_text(json.dumps(exported)); refs[str(report)] = file_sha256(report)
    with pytest.raises(ValueError, match='source'):
        read_complete(report, file_sha256(report), refs, expected_train=1, expected_val=1, captions_per_video=3)
    with pytest.raises(FileExistsError):
        export_complete(out, result, videos, image_size=16)
    (videos/'c.mp4').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='video'):
        export_complete(tmp_path/'failure', result, videos, image_size=16)


def test_official_one_fps_then_uniform_frame_policy():
    from semantic_retrieval.full_data.loader import official_frame_indices
    assert official_frame_indices(9, 5.0, 12) == [0, 5]
    assert official_frame_indices(50, 5.0, 3) == [0, 20, 45]
    with pytest.raises(ValueError):
        official_frame_indices(1, 0.5, 12)
