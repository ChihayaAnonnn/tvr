import hashlib
import json

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
    exported = export_complete(tmp_path/'export', result, videos, image_size=16)
    assert exported['decoded_video_count'] == 2
    assert exported['caption_count'] == 6
    records = json.loads((tmp_path/'export/videos-000.json').read_bytes())['rows']
    assert {r['split'] for r in records} == {'train', 'val'}
    assert all(r['frame_count'] == 3 and r['fps'] == 5 for r in records)
    captions = json.loads((tmp_path/'export/captions-000.json').read_bytes())['rows']
    assert captions == result['captions']
    with pytest.raises(FileExistsError):
        export_complete(tmp_path/'export', result, videos, image_size=16)
    (videos/'c.mp4').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='video'):
        export_complete(tmp_path/'failure', result, videos, image_size=16)
