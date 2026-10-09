import pytest
from semantic_retrieval.evidence.authorization import assert_evidence_authorized

def test_evidence_never_inherits_old_study_authorization():
    manifest = {
        "study": "semantic-answerability-pilot-v1",
        "registry_path": "experiments/semantic-answerability-pilot-v1/registry.jsonl",
        "config": {"evaluation": {"config_kind": "evidence-retrieval-v1"}},
        "test_authorized": False,
    }
    with pytest.raises(ValueError):
        assert_evidence_authorized(repo_root=None, registry_path=None,
                                   manifest=manifest, operation="start")

def test_missing_marker_with_evidence_command_rejected():
    manifest = {'study':'unrelated', 'command':['python','scripts/train_evidence_retrieval.py'],
                'config':{'evaluation':{}}, 'test_authorized':False}
    with pytest.raises(ValueError):
        assert_evidence_authorized(repo_root=None,registry_path=None,manifest=manifest,operation='reserve')

def test_scientific_cpu_cannot_skip_four_gpu_gate(config_factory,tmp_path):
    from evidence.archive_fixture import reserve_evidence_fixture
    with pytest.raises(ValueError,match='four-GPU'):
        reserve_evidence_fixture(tmp_path,config_factory(stage='reference'))


def test_true_test_flag_still_rejected(config_factory,tmp_path):
    from evidence.archive_fixture import reserve_evidence_fixture
    from semantic_retrieval.evidence.runtime import prepare_evidence_run
    repo,cfg = reserve_evidence_fixture(tmp_path,config_factory(stage='cpu_fixture'))
    context = prepare_evidence_run(cfg,repo_root=repo)
    manifest = {**context.manifest,'test_authorized':True}
    with pytest.raises(ValueError,match='sealed'):
        assert_evidence_authorized(repo_root=repo,registry_path=cfg.runtime.registry_path,manifest=manifest,operation='start')


def test_gpu_qualification_requires_registered_random_weight_bytes(config_factory,tmp_path):
    import json
    import yaml
    from evidence.archive_fixture import reserve_evidence_fixture
    from test_runtime_artifacts import git
    from semantic_retrieval.evidence.authorization import assert_evidence_authorized
    repo,cfg = reserve_evidence_fixture(tmp_path,config_factory(stage='cpu_fixture'))
    study_path = cfg.runtime.study_path
    study = yaml.safe_load(study_path.read_text());study['software_authorization']['four_gpu'] = True
    study_path.write_text(yaml.safe_dump(study));git(repo,'add',str(study_path));git(repo,'commit','-qm','authorize synthetic GPU qualification')
    manifest = json.loads((cfg.runtime.run_dir/'manifest.json').read_text())
    manifest['config']['evaluation']['evidence_stage'] = 'gpu_qualification'
    manifest['config']['model']['base']['clip']['fixture'] = False
    manifest['environment'].update(gpu_count=4,distributed_topology='torchrun-4')
    with pytest.raises(ValueError,match='registered random'):
        assert_evidence_authorized(repo_root=repo,registry_path=cfg.runtime.registry_path,manifest=manifest,operation='reserve')


def test_value_qualification_binds_capacity_and_each_head(config_factory):
    from dataclasses import replace
    from semantic_retrieval.evidence.authorization import validate_value_qualification
    from semantic_retrieval.config import config_payload
    cfg = config_factory()
    qualified = config_payload(cfg.value)
    report = {'qualified_value_kinds':['distribution','geometry','mean_regressor']}
    validate_value_qualification(qualified,config_payload(cfg.value),report)
    with pytest.raises(ValueError,match='value'):
        validate_value_qualification(qualified,config_payload(replace(cfg.value,hidden_dim=cfg.value.hidden_dim+1)),report)
    with pytest.raises(ValueError,match='value'):
        validate_value_qualification(qualified,config_payload(replace(cfg.value,kind='geometry')),{'qualified_value_kinds':['distribution']})
