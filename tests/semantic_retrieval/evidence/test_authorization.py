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


@pytest.mark.parametrize('operation', ['reserve', 'start'])
def test_science_split_scope_cannot_unlock_locked_with_development_gate(config_factory,tmp_path,monkeypatch,operation):
    import copy,json,hashlib,yaml
    from evidence.archive_fixture import reserve_evidence_fixture
    from test_runtime_artifacts import git
    import semantic_retrieval.evidence.authorization as authorization
    repo,cfg = reserve_evidence_fixture(tmp_path,config_factory(stage='cpu_fixture'))
    control = cfg.runtime.study_path.parent
    evidence = control/'scope-evidence.json';evidence.write_text('{"completed":true}\n')
    study = yaml.safe_load(cfg.runtime.study_path.read_text())
    study['science'] = dict(authorized=True,authorized_stages=['evaluation','transitions','calibration'],
        stage_eval_splits={'evaluation':['val_development'],'transitions':['train','val_development','val_calibration'],'calibration':['val_calibration']},
        gates={stage:dict(passed=True,reports=[dict(path=str(evidence.relative_to(repo)),sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())]) for stage in ['evaluation','transitions','calibration']})
    cfg.runtime.study_path.write_text(yaml.safe_dump(study));git(repo,'add',str(control));git(repo,'commit','-qm','scope test protocol')
    manifest=json.loads((cfg.runtime.run_dir/'manifest.json').read_text())
    # Only the independent GPU prerequisite is isolated here; scope authorization
    # itself consumes a real committed study/report. Production scope is also
    # witnessed using the completed real four-rank qualification before launch.
    monkeypatch.setattr(authorization,'validate_qualification',lambda *args:None)
    def check(stage,split):
        candidate=copy.deepcopy(manifest);candidate['config']['evaluation'].update(evidence_stage=stage,eval_split=split)
        return authorization.assert_evidence_authorized(repo_root=repo,registry_path=cfg.runtime.registry_path,manifest=candidate,operation=operation)
    for stage,allowed in study['science']['stage_eval_splits'].items():
        for split in allowed:check(stage,split)
        for split in ['val_locked','test']:
            with pytest.raises(ValueError,match='split|sealed'):check(stage,split)
    with pytest.raises(ValueError,match='split'):check('calibration','val_development')
    study['science']['stage_eval_splits']['evaluation'].append('val_locked')
    cfg.runtime.study_path.write_text(yaml.safe_dump(study));git(repo,'add',str(cfg.runtime.study_path));git(repo,'commit','-qm','locked scope alone is insufficient')
    with pytest.raises(ValueError,match='locked'):check('evaluation','val_locked')
    study['science']['gates']['locked_validation'] = copy.deepcopy(study['science']['gates']['evaluation'])
    cfg.runtime.study_path.write_text(yaml.safe_dump(study));git(repo,'add',str(cfg.runtime.study_path));git(repo,'commit','-qm','explicit frozen locked evidence')
    check('evaluation','val_locked')
    study['science']['stage_eval_splits']['calibration'].append('val_development')
    cfg.runtime.study_path.write_text(yaml.safe_dump(study));git(repo,'add',str(cfg.runtime.study_path));git(repo,'commit','-qm','invalid calibration scope')
    with pytest.raises(ValueError,match='split'):check('calibration','val_development')
    study['science'].pop('stage_eval_splits');cfg.runtime.study_path.write_text(yaml.safe_dump(study));git(repo,'add',str(cfg.runtime.study_path));git(repo,'commit','-qm','missing scope fails closed')
    with pytest.raises(ValueError,match='split'):check('evaluation','val_development')
