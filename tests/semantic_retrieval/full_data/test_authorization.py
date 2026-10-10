import copy
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def permission_fixture(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    control = repo / 'experiments/full-data-retrieval-v1'
    control.mkdir(parents=True)
    origin = Path(__file__).resolve().parents[3] / 'experiments/full-data-retrieval-v1/study.yaml'
    (control / 'study.yaml').write_bytes(origin.read_bytes())
    (control / 'registry.jsonl').write_text('')
    for command in [('init', '-q'), ('add', '.'), ('-c', 'user.name=test', '-c', 'user.email=test@example.org', 'commit', '-qm', 'protocol')]:
        subprocess.run(['git', *command], cwd=repo, check=True)
    manifest = dict(study='full-data-retrieval-v1', registry_path='experiments/full-data-retrieval-v1/registry.jsonl',
                    artifact_path=str(tmp_path/'artifacts/full-data-retrieval-v1/msrvtt/baseline/run'),
                    dataset='msrvtt', stage='baseline', variant='complete-source-input',
                    test_authorized=False, validation_authorized=True,
                    environment=dict(precision='fp32', gpu_count=0, distributed_topology='single-process'),
                    config=dict(model={}, loss={}, optimizer={}, scheduler={},
                                evaluation=dict(eval_split='val', qualification_scenario='full_data_prepare:v1')))
    return repo, control/'registry.jsonl', manifest


@pytest.mark.parametrize('operation', ['reserve', 'start'])
def test_permission_exact_cpu_only_and_no_study_retarget(permission_fixture, operation):
    from semantic_retrieval.full_data.authorization import assert_full_data_authorized
    repo, registry, manifest = permission_fixture
    def check(m, argv=None):
        assert_full_data_authorized(repo_root=repo, registry_path=registry, manifest=m, operation=operation,
                                   command=argv or ['python', 'scripts/prepare_full_data_retrieval.py'])
    check(manifest)
    for mutation in [dict(test_authorized=True), dict(study='evidence-conditioned-retrieval-v1'),
                     dict(environment=dict(precision='fp32', gpu_count=1, distributed_topology='single-process')),
                     dict(config=dict(manifest['config'], model={'train': True})),
                     dict(artifact_path=str(repo/'artifacts/other/msrvtt/baseline/run'))]:
        with pytest.raises(ValueError):
            check(dict(manifest, **mutation))
    with pytest.raises(ValueError):
        check(manifest, ['python', 'scripts/train_full_data_retrieval.py'])
    (registry.with_name('study.yaml')).write_text('study: changed\n')
    with pytest.raises(ValueError, match='committed'):
        check(manifest)


def test_missing_marker_and_unqualified_science_fail_closed(permission_fixture):
    from semantic_retrieval.full_data.authorization import assert_full_data_authorized
    repo, registry, manifest = permission_fixture
    for evaluation in [{}, {'config_kind': 'full-data-retrieval-v1', 'eval_split': 'val'}]:
        candidate = copy.deepcopy(manifest)
        candidate['config']['evaluation'] = evaluation
        with pytest.raises(ValueError):
            assert_full_data_authorized(repo_root=repo, registry_path=registry, manifest=candidate,
                                       operation='reserve', command=['python', 'scripts/train_full_data_retrieval.py'])


def test_unrelated_study_is_untouched():
    from semantic_retrieval.full_data.authorization import assert_full_data_authorized
    assert_full_data_authorized(repo_root=None, registry_path=None, manifest={'study': 'unrelated'}, operation='reserve')
