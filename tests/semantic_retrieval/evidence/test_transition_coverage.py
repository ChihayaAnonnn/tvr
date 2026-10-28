import json
from dataclasses import replace
import pytest
from test_episodes import episode_manifest


@pytest.fixture
def coverage_case(config_factory,episode_manifest,tmp_path):
    import copy
    from types import SimpleNamespace
    from experiment_artifacts import ArtifactArrayStore
    from semantic_retrieval.evidence.config import SplitBinding,InputReference
    from semantic_retrieval.evidence.data import read_split_binding
    from semantic_retrieval.evidence.transitions import TransitionProducerIdentity,TransitionReferences
    from semantic_retrieval.data.episodes import file_sha256
    cfg=config_factory(stage='transitions');payload,write,_=episode_manifest
    payload.update(schema_version=3,relevance_space='reviewed_semantic')
    row=payload['episodes'][0];row.update(split='val_development',query_source_ids=['query-source'],query_component_id='d'*64)
    other=copy.deepcopy(row);other.update(query_id='q1',caption_group_id='caption-1',query_component_id='e'*64,relevance=['unknown','negative'],answer_in_candidates='unknown')
    payload['episodes'].append(other);path=write();binding=SplitBinding('V_development',InputReference(path,file_sha256(path)),'reviewed_semantic')
    index=tmp_path/'index.json';index.write_text(json.dumps(dict(roles={'V_development':['a','b']})))
    cfg=replace(cfg,splits=(binding,),index=replace(cfg.index,reference=InputReference(index,file_sha256(index))))
    episodes=read_split_binding(binding);store=ArtifactArrayStore(tmp_path/'run'/'arrays')
    producer=TransitionProducerIdentity('a'*64,'b'*64,cfg.index.reference.sha256,binding.manifest.sha256,*(['c'*64]*4))
    index_path=store.run_root/'provenance/transitions.json';index_path.parent.mkdir(parents=True);index_path.write_text('{}')
    refs=TransitionReferences(index_path,file_sha256(index_path),producer,())
    return SimpleNamespace(cfg=cfg,episodes=episodes,store=store,producer=producer,refs=refs)


def test_transition_coverage_preserves_full_cohort_unknown_and_selection(coverage_case):
    from experiment_artifacts import verify_array_reference
    from semantic_retrieval.evidence.execution import publish_transition_coverage
    from semantic_retrieval.evidence.types import digest
    c=coverage_case
    path=publish_transition_coverage(c.cfg,'V_development',c.episodes,c.producer,c.refs,c.store)
    report=json.loads(path.read_text());arrays={a['key']:verify_array_reference(c.store.run_root,a) for a in report['array_references']}
    assert report['cohort_complete'] is True and report['query_count']==2
    assert arrays['candidate_presence'].tolist()==[1,-1]
    expected=[[[ 'a','b'].index(cid) for cid in w.episode.canonical_ids] for w in c.episodes]
    assert arrays['selected_gallery_indices'].tolist()==expected
    assert report['query_order_sha256']==digest([w.episode.query_id for w in c.episodes])
    assert report['component_order_sha256']==digest([w.episode.query_component_id for w in c.episodes])
    assert 'candidate_presence' not in report


@pytest.mark.parametrize('corruption',['missing-query','presence','candidate','producer'])
def test_transition_coverage_rejects_incomplete_or_inconsistent_cohort(coverage_case,corruption):
    from semantic_retrieval.evidence.execution import publish_transition_coverage
    c=coverage_case;episodes=c.episodes;producer=c.producer
    if corruption=='missing-query':episodes=episodes[:1]
    elif corruption=='presence':episodes=(replace(episodes[0],episode=replace(episodes[0].episode,answer_in_candidates=0)),*episodes[1:])
    elif corruption=='candidate':episodes=(replace(episodes[0],episode=replace(episodes[0].episode,canonical_ids=('outside','a'))),*episodes[1:])
    else:producer=replace(producer,label_sha256='f'*64)
    with pytest.raises(ValueError):publish_transition_coverage(c.cfg,'V_development',episodes,producer,c.refs,c.store)
    assert not (c.store.run_root/'provenance/transition-coverage.json').exists()
