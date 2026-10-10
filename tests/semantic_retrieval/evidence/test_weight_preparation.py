import hashlib
from pathlib import Path

import pytest


def test_verified_download_never_publishes_bad_or_oversized_bytes(tmp_path):
    from semantic_retrieval.evidence.weight_preparation import download_verified
    source = tmp_path/'source';source.write_bytes(b'known immutable bytes')
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path/'weights.pt'
    download_verified(source.as_uri(),target,sha,max_bytes=100,max_seconds=10)
    assert target.read_bytes() == b'known immutable bytes'
    with pytest.raises(FileExistsError):
        download_verified(source.as_uri(),target,sha,max_bytes=100,max_seconds=10)
    bad = tmp_path/'bad.pt'
    with pytest.raises(ValueError,match='SHA'):
        download_verified(source.as_uri(),bad,'0'*64,max_bytes=100,max_seconds=10)
    assert not bad.exists()
    small = tmp_path/'small.pt'
    with pytest.raises(ValueError,match='limit'):
        download_verified(source.as_uri(),small,sha,max_bytes=2,max_seconds=10)
    assert not small.exists()


def test_cpu_weight_permission_is_separate_from_scientific_gates(config_factory,tmp_path):
    import yaml
    from evidence.archive_fixture import reserve_evidence_fixture
    from semantic_retrieval.evidence.authorization import assert_evidence_authorized
    from semantic_retrieval.evidence.weight_preparation import preparation_spec, preparation_scenario
    from semantic_retrieval.runtime.artifacts import prepare_run
    from semantic_retrieval.evidence.runtime import prepare_evidence_run
    from test_runtime_artifacts import git
    repo,cfg=reserve_evidence_fixture(tmp_path,config_factory(stage='cpu_fixture'))
    context=prepare_evidence_run(cfg,repo_root=repo)
    manifest=dict(context.manifest,variant='pretrained-clip-input',dataset='synthetic',
                  command=['python','scripts/prepare_evidence_weights.py'],
                  config=dict(model={},loss={},optimizer={},scheduler={},
                              evaluation=dict(eval_split='val',qualification_scenario=preparation_scenario())))
    def check(m=manifest):
        assert_evidence_authorized(repo_root=repo,registry_path=cfg.runtime.registry_path,manifest=m,operation='reserve')
    with pytest.raises(ValueError):check()
    study=yaml.safe_load(cfg.runtime.study_path.read_text())
    study['weight_preparation']=dict(authorized=True,scope='cpu_public_pretrained_input_only',spec=preparation_spec())
    cfg.runtime.study_path.write_text(yaml.safe_dump(study));git(repo,'add','.');git(repo,'commit','-qm','weight preparation')
    check()
    with pytest.raises(ValueError):check(dict(manifest,environment=dict(manifest['environment'],gpu_count=1)))
    with pytest.raises(ValueError):check(dict(manifest,command=['python','scripts/train_evidence_retrieval.py']))
    with pytest.raises(ValueError):check(dict(manifest,config=dict(manifest['config'],model={'anything':True})))
    with pytest.raises(ValueError):check(dict(manifest,test_authorized=True))
