import pytest
from evidence.test_source_audit import metadata


def test_source_roles_quarantine_transitive_content_and_original_source_crossings():
    from semantic_retrieval.evidence.source_preparation import partition_sources
    rows=[metadata('t0','abcdefghijk'),metadata('t1','abcdefghijk'),metadata('v0','12345678901')]
    rows += [metadata('t'+str(i),str(i).zfill(11)) for i in range(2,12)]
    rows += [metadata('v'+str(i),str(100+i).zfill(11)) for i in range(1,11)]
    train={r['video_id'] for r in rows if r['video_id'].startswith('t')}
    val={r['video_id'] for r in rows if r['video_id'].startswith('v')}
    import hashlib
    content={i:hashlib.sha256(i.encode()).hexdigest() for i in train|val}
    content['v0']=content['t1']
    roles,components,excluded=partition_sources(rows,train,val,content,seed=0)
    assert excluded == {'t0','t1'}
    assert components['t0']==components['t1']==components['v0']
    assert set(roles)==(train|val)-excluded
    assert set(roles.values())=={'T_match','T_value','V_development','V_calibration','V_locked'}
    seen={}
    for i,role in roles.items():
        assert components[i] not in seen or seen[components[i]]==role
        seen[components[i]]=role
    assert partition_sources(list(reversed(rows)),train,val,content,seed=0)==(roles,components,excluded)


def test_pilot_sampling_keeps_exact_component_count_and_never_splits_source_group():
    from semantic_retrieval.evidence.source_preparation import select_pilot
    roles={'a':'T_match','b':'T_match','c':'T_match','d':'T_match'}
    components={'a':'x','b':'x','c':'y','d':'z'}
    selected=select_pilot(roles,components,per_role=2,seed=713)
    assert len(selected['T_match'])==2
    assert len({components[i] for i in selected['T_match']})==2
    with pytest.raises(ValueError):select_pilot(roles,components,per_role=4,seed=713)


def test_export_real_decoded_videos_roundtrips_five_production_role_readers(tmp_path):
    import cv2
    import numpy as np
    import hashlib
    from semantic_retrieval.evidence.source_preparation import export_pilot, preparation_spec
    from semantic_retrieval.data.episodes import file_sha256
    from semantic_retrieval.data.msrvtt_snapshot import inspect_training_video
    from semantic_retrieval.evidence.config import ROLES
    videos=tmp_path/'videos';videos.mkdir()
    selected={r:[f'video{n*2}',f'video{n*2+1}'] for n,r in enumerate(sorted(ROLES))}
    content,frames,components,captions={},{},{},[]
    for n,i in enumerate(i for ids in selected.values() for i in ids):
        path=videos/(i+'.mp4')
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(16,16))
        assert writer.isOpened()
        for k in range(3):writer.write(np.full((16,16,3),n*17+k,np.uint8))
        writer.release()
        content[i]=file_sha256(path);components[i]=hashlib.sha256(i.encode()).hexdigest()
        frames[i]=inspect_training_video(path,expected_sha256=content[i],image_size=16,max_frames=2)
        captions.append(dict(query_id=i+':caption',official_video_id=i,query_text='a small video'))
    spec=dict(preparation_spec(),video_directory=str(videos))
    result=export_pilot(tmp_path/'export',spec,captions,selected,components,content,frames,dict(test_authorized=False))
    assert result['roles_disjoint'] and result['gallery_coverage_verified']
    assert result['selected_video_count']==10 and len(result['splits'])==5
    with pytest.raises(FileExistsError):
        export_pilot(tmp_path/'export',spec,captions,selected,components,content,frames,{})


def test_source_export_permission_is_cpu_only_and_cannot_authorize_scientific_entry(config_factory,tmp_path):
    import yaml
    from evidence.archive_fixture import reserve_evidence_fixture
    from semantic_retrieval.evidence.authorization import assert_evidence_authorized
    from semantic_retrieval.evidence.source_preparation import preparation_spec,preparation_scenario
    from semantic_retrieval.evidence.runtime import prepare_evidence_run
    from test_runtime_artifacts import git
    repo,cfg=reserve_evidence_fixture(tmp_path,config_factory(stage='cpu_fixture'))
    context=prepare_evidence_run(cfg,repo_root=repo)
    m=dict(context.manifest,variant='source-role-input',dataset='msrvtt',command=['python','scripts/prepare_evidence_sources.py'],
           config=dict(model={},loss={},optimizer={},scheduler={},evaluation=dict(eval_split='val',qualification_scenario=preparation_scenario())))
    def check(manifest=m):
        assert_evidence_authorized(repo_root=repo,registry_path=cfg.runtime.registry_path,manifest=manifest,operation='reserve')
    with pytest.raises(ValueError):check()
    study=yaml.safe_load(cfg.runtime.study_path.read_text())
    study['source_preparation']=dict(authorized=True,scope='cpu_training_allowlist_export_and_source_roles',spec=preparation_spec())
    cfg.runtime.study_path.write_text(yaml.safe_dump(study));git(repo,'add','.');git(repo,'commit','-qm','source preparation')
    check()
    with pytest.raises(ValueError):check(dict(m,environment=dict(m['environment'],gpu_count=1)))
    with pytest.raises(ValueError):check(dict(m,command=['python','scripts/train_evidence_retrieval.py']))
    with pytest.raises(ValueError):check(dict(m,test_authorized=True))
