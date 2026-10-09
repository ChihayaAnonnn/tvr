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
