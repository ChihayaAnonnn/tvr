from pathlib import Path

import pytest


def test_training_configuration_uses_ranked_v2_data_and_fixed_arm_protocol(tmp_path):
    from semantic_retrieval.runtime.pilot_workflow import training_configuration
    from semantic_retrieval.runtime.pilot_candidates import file_reference, write_json
    from semantic_retrieval.models.clip import ClipConfig
    from semantic_retrieval.runtime.pilot import pilot_trainable_prefixes
    from dataclasses import asdict
    protocol = dict(protocol_id='bound-protocol', **{k:'a'*64 for k in (
        'annotation_sha256','relation_sha256','id_map_sha256','source_report_sha256','allowlist_sha256','source_annotation_sha256')})
    episode=tmp_path/'episodes.json'; write_json(episode,dict(gallery_version='train-hashed',label_protocol=protocol))
    orders=tmp_path/'orders.json'; write_json(orders,{'q':['positive','negative']})
    weights=tmp_path/'clip.npz'; weights.write_bytes(b'bound fixture')
    result=dict(clip=asdict(ClipConfig(512,224,768,12,32,77,49408,512,8,12)),clip_weights=file_reference(weights),
        splits={'train':dict(episodes=file_reference(episode),orders=file_reference(orders))})
    for arm,steps in (('A0',240),('P1',120)):
        cfg=training_configuration(arm,seed=2,candidates=result)
        assert cfg.model.pilot_arm == arm
        assert cfg.data.relevance_space == 'benchmark_annotation'
        assert cfg.data.episode_manifest == episode
        assert cfg.train.scheduler_steps == steps and cfg.train.scheduler_name == 'constant'
        assert cfg.train.trainable_prefixes == pilot_trainable_prefixes(arm)
        assert cfg.train.pilot.sampler_seed == 2
    with pytest.raises(ValueError,match='parent'):
        training_configuration('A3',seed=2,candidates=result)


def test_indexed_inputs_keep_a_large_training_gallery_out_of_manifest(tmp_path):
    from semantic_retrieval.runtime.pilot_workflow import compact_input_references
    # Input-index creation is controller metadata, before reservation and workers.
    refs=[dict(path=f'/canonical/video-{i:05}.avi',sha256='a'*64) for i in range(9000)]
    with pytest.raises(ValueError,match='index'):
        compact_input_references(refs,index_path=None)


def test_large_inventory_uses_real_shared_authenticated_index_contract(tmp_path):
    from semantic_retrieval.runtime.pilot_workflow import compact_input_references
    from semantic_retrieval.runtime.pilot_candidates import file_reference
    from semantic_retrieval.runtime.input_bindings import manifest_input_references
    refs=[]
    for i in range(901):
        path=tmp_path/f'video-{i}.avi'; path.write_bytes(str(i).encode()); refs.append(file_reference(path))
    indices=compact_input_references(refs,index_path=tmp_path/'indices'/'train.input-index.json')
    assert len(indices)==2
    expanded=manifest_input_references(tmp_path,dict(dataset_splits=indices,input_checkpoints=[]))
    assert all(expanded[r['path']]==r['sha256'] for r in refs)
    assert all(Path(r['path']).stat().st_size < 1024*1024 for r in indices)


@pytest.mark.parametrize('phase', ['C0', 'V0'])
def test_analysis_factory_requires_complete_archived_predecessors(tmp_path, phase):
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates
    with pytest.raises(ValueError, match='archive-derived'):
        workflow_templates(tmp_path, phase, evidence={}, environment={},
            python_executable='/python', index_directory=tmp_path/'configs')


def test_score_panel_binds_exact_completed_parent_and_frozen_expansion_inputs(tmp_path):
    from semantic_retrieval.runtime.pilot_workflow import score_panel_document
    from semantic_retrieval.runtime.pilot_candidates import file_reference, write_json
    panel = tmp_path/'panels.json'
    write_json(panel, dict(schema_version=1, arm='A3', seed=2, groups=[]))
    parent = dict(phase='P2b', summary=dict(path='experiments/p/summaries/P2b.json', sha256='a'*64, commit='b'*40),
        run_id='parent', config=dict(path='/config',sha256='c'*64),
        checkpoint=dict(path='/checkpoint',sha256='d'*64), metadata=dict(path='/metadata',sha256='e'*64))
    orders, frozen = dict(path='/orders',sha256='f'*64), dict(path='/frozen',sha256='0'*64)
    document = score_panel_document(file_reference(panel), arm='A3', seed=2, parent=parent,
        candidate_orders=orders, frozen=frozen)
    assert document['training_parent'] == parent
    assert document['candidate_orders'] == orders and document['frozen'] == frozen
    with pytest.raises(ValueError, match='arm/seed'):
        score_panel_document(file_reference(panel), arm='A2', seed=2, parent=parent, candidate_orders=orders)


def test_q0_preparation_factory_seals_cpu_only_command_without_borrowed_inputs(tmp_path):
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates
    run = workflow_templates(tmp_path, 'Q0P', evidence={}, environment={'gpu_count':4},
        python_executable='/python', index_directory=tmp_path/'unused')[0]
    assert run['environment']['gpu_count'] == 0
    assert 'CUDA_VISIBLE_DEVICES=' in run['command'] and '--mode' in run['command']
    assert 'prepare-production' in run['command']
    assert run['input_checkpoints'] == run['dataset_splits'] == []
    assert run['config']['evaluation'] == dict(eval_split='val', qualification_scenario='pilot_q0_prep_v2')


@pytest.mark.parametrize('phase', ['Q0', 'E0', 'V0E'])
def test_producer_factory_requires_archived_inputs_before_materializing(tmp_path, phase):
    from semantic_retrieval.runtime.pilot_workflow import workflow_templates
    with pytest.raises(ValueError, match='prerequisite'):
        workflow_templates(tmp_path, phase, evidence={}, environment={},
            python_executable='/python', index_directory=tmp_path/'unused')
