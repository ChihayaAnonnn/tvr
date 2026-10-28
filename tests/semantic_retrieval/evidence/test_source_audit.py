import json

import pytest


def metadata(video, source, start=0, end=10):
    return {'video_id': video, 'url': 'https://www.youtube.com/watch?v='+source,
            'start time': start, 'end time': end}


def test_source_metadata_selection_precedes_decoding_excluded_values():
    from semantic_retrieval.evidence.source_audit import training_source_metadata
    raw = b'{"videos":[{"url":BROKEN,"video_id":"sealed"},'+json.dumps(metadata('a','abcdefghijk')).encode()+b'],"sentences":[]}'
    # Excluded JSON still must be syntactically valid. Its value decoder must never run.
    raw = raw.replace(b'BROKEN', b'"SEALED"')
    def decode(value):
        assert b'SEALED' not in value
        return json.loads(value)
    rows, counts = training_source_metadata(raw, {'a'}, decode_value=decode)
    assert rows == [metadata('a','abcdefghijk')]
    assert counts == {'selected':1, 'excluded':1, 'excluded_values_decoded':0, 'captions_decoded':0}


def test_source_audit_distinguishes_shared_source_overlap_and_exact_bytes():
    from semantic_retrieval.evidence.source_audit import summarize_sources
    rows = [metadata('a','abcdefghijk',0,10),metadata('b','abcdefghijk',5,12),
            metadata('c','abcdefghijk',20,30),metadata('d','12345678901',0,10)]
    out = summarize_sources(rows, {'a','d'}, {'b','c'}, {'a':'1'*64,'b':'2'*64,'c':'3'*64,'d':'1'*64})
    assert out['cross_train_validation_source_groups'] == 1
    assert out['train_videos_in_cross_source_groups'] == 1
    assert out['validation_videos_in_cross_source_groups'] == 2
    assert out['cross_partition_temporally_overlapping_pairs'] == 1
    assert out['overlapping_pair_examples'] == [{'train_video_id':'a','val_video_id':'b','overlap_seconds':5.0}]
    assert out['exact_duplicate_content_groups'] == 1
    assert out['cross_partition_exact_duplicate_groups'] == 0


@pytest.mark.parametrize('mutation', ['missing','overlap','nan','url','duplicate'])
def test_invalid_source_metadata_fails_closed(mutation):
    from semantic_retrieval.evidence.source_audit import summarize_sources
    rows = [metadata('a','abcdefghijk'),metadata('b','12345678901')]
    train,val = {'a'},{'b'}
    if mutation=='missing':rows.pop()
    if mutation=='overlap':train.add('b')
    if mutation=='nan':rows[0]['end time']=float('nan')
    if mutation=='url':rows[0]['url']='https://example.org/?v=abcdefghijk'
    if mutation=='duplicate':rows.append(rows[0])
    with pytest.raises(ValueError):
        summarize_sources(rows, train, val, {'a':'1'*64,'b':'2'*64})
