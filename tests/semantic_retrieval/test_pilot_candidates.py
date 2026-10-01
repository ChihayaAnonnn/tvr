import copy

import numpy as np
import pytest


def test_clip_order_is_cosine_and_has_canonical_ties():
    from semantic_retrieval.runtime.pilot_candidates import fixed_clip_orders

    ids = ('a', 'b', 'c')
    query = np.array([[1., 0.], [0., 1.]])
    video = np.array([[10., 0.], [1., 0.], [0., 2.]])
    assert fixed_clip_orders(query, video, ids).tolist() == [[0, 1, 2], [2, 0, 1]]
    with pytest.raises(ValueError, match='finite|zero'):
        fixed_clip_orders(query, np.zeros_like(video), ids)


def test_panel_masks_preserve_natural_order_and_balance_each_k():
    from semantic_retrieval.runtime.pilot_candidates import panel_groups

    ids = tuple(f'c{i:03}' for i in range(129))
    # q1 is a natural miss at every registered K, including 128.
    order = np.stack([np.arange(129), np.arange(129)[::-1]])
    groups = panel_groups(('q0', 'q1'), ids, order, {'q0': ('c000',), 'q1': ('c000',)})
    assert len(groups) == 12
    for group in groups:
        k, track, arrays = group['K'], group['track'], group['arrays']
        masks = arrays['candidate_mask']
        assert (masks.sum(1) == k).all()
        if track == 'natural':
            assert arrays['query_ids'].tolist() == ['q0', 'q1']
            assert masks[:, 0].tolist() == [True, False]
        else:
            assert arrays['query_ids'].tolist() == ['q0', 'q0', 'q1', 'q1']
            assert masks[:, 0].tolist() == [True, False, True, False]
            assert (arrays['initial_candidate_mask'].sum(1) == 4).all()


def test_ranked_training_inventory_contains_real_top_negatives_and_gt():
    from semantic_retrieval.runtime.pilot_candidates import ranked_episode_payload

    ids = tuple(f'c{i:03}' for i in range(140))
    videos = [dict(canonical_id=i, alias_id='v'+i, video_path='/synthetic/'+i,
                   content_sha256='a'*64, frame_count=8) for i in ids]
    payload = dict(schema_version=2, gallery_version='train-v2', preprocessing={},
                   relevance_space='benchmark_annotation', label_protocol={'binding': 'kept'},
                   label_references={'relations': {'path': '/r', 'sha256': 'b'*64}},
                   episodes=[dict(query_id='q', canonical_ids=['c000'], query_text='text',
                                  query_source_ids=['v-source'], query_component_id='component')])
    original = copy.deepcopy(payload)
    ranked = np.array([np.arange(140)[::-1]])
    result, orders = ranked_episode_payload(payload, {'videos': videos}, ranked, {'q': ('c000',)})
    row = result['episodes'][0]
    assert len(row['canonical_ids']) == 129
    assert row['canonical_ids'] == sorted(['c000'] + list(ids[-128:]))
    assert orders['q'] == list(ids[-128:][::-1]) + ['c000']
    assert row['query_source_ids'] == ['v-source']
    assert row['query_component_id'] == 'component'
    assert result['label_protocol'] == payload['label_protocol']
    assert payload == original


def test_locked_ranking_configuration_is_phase_specific():
    from semantic_retrieval.runtime.pilot_candidates import validate_candidate_config

    cfg = dict(schema_version=1, clip={}, weights={'path': '/clip', 'sha256': 'a'*64},
               splits={name: dict(episodes={}, gallery={}) for name in ('train', 'val_development', 'val_calibration')})
    validate_candidate_config(cfg, phase_id='E0')
    cfg['splits']['val_locked'] = dict(episodes={}, gallery={})
    with pytest.raises(ValueError, match='locked|splits'):
        validate_candidate_config(cfg, phase_id='E0')
    cfg['splits'] = {'val_locked': cfg['splits']['val_locked']}
    validate_candidate_config(cfg, phase_id='V0E')
    with pytest.raises(ValueError, match='phase'):
        validate_candidate_config(cfg, phase_id='P1a')


def test_actual_clip_producer_connects_v2_inventory_to_score_panels(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    import json
    from pathlib import Path
    import torch
    from test_pilot_qualification import qualification_context
    from test_runtime_artifacts import reservation, commit_reservation
    from semantic_retrieval.runtime.artifacts import prepare_run,start_run
    from semantic_retrieval.runtime.pilot_candidates import candidate_manifest_config,file_reference,produce_candidate_artifacts,write_json

    synthetic,cfg = qualification_context(tmp_path)
    source = cfg.data.episode_manifest.parent
    payload = json.loads((source/'val_calibration.json').read_text())
    payload['gallery_version'] = 'synthetic-development'
    for row in payload['episodes']:
        row.update(split='val_development',gallery_version='synthetic-development')
    dev_path = synthetic.paths.provenance/'development.json'; write_json(dev_path,payload)
    gallery = json.loads((source/'val_calibration-gallery.json').read_text()); gallery['gallery_version']='synthetic-development'
    dev_gallery = synthetic.paths.provenance/'development-gallery.json'; write_json(dev_gallery,gallery)
    from dataclasses import asdict
    body = dict(schema_version=1,clip=asdict(cfg.model.clip),weights=file_reference(cfg.model.clip_weights),splits={
        'train':dict(episodes=file_reference(source/'train.json'),gallery=file_reference(source/'train-gallery.json')),
        'val_development':dict(episodes=file_reference(dev_path),gallery=file_reference(dev_gallery)),
        'val_calibration':dict(episodes=file_reference(source/'val_calibration.json'),gallery=file_reference(source/'val_calibration-gallery.json'))})
    case=tmp_path/'candidates'; case.mkdir()
    refs=synthetic.manifest['dataset_splits']+[file_reference(dev_path),file_reference(dev_gallery)]
    repo,rt=reservation.__wrapped__(case,SimpleNamespace(param=dict(
        config=candidate_manifest_config(body,phase_id='E0'),input_checkpoints=synthetic.manifest['input_checkpoints'],
        dataset_splits=refs,selection_lineage=[dict(kind='pilot_phase',protocol_version=1,phase_id='E0',evidence={})])))
    rt=replace(rt,eval_split='val_development'); commit_reservation(repo,rt)
    context=prepare_run(rt,repo_root=repo); start_run(context)
    output=produce_candidate_artifacts(context,body,device=torch.device('cpu'))
    assert set(output['splits']) == {'train','val_development','val_calibration'}
    train=json.loads(Path(output['splits']['train']['episodes']['path']).read_text())
    assert all(len(row['canonical_ids']) == 129 for row in train['episodes'])
    panel_ref=output['splits']['val_development']['panels'][0]['panels']
    panels=json.loads(Path(panel_ref['path']).read_text())
    assert panels['split_sha256'] == output['splits']['val_development']['episodes']['sha256']
    assert len(panels['groups']) == 12
    assert len(output['splits']['val_development']['panels']) == 15
    assert (context.paths.provenance/'pilot-candidates.json').is_file()
