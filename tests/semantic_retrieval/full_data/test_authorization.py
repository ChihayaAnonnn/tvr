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


@pytest.mark.parametrize('operation', ['reserve', 'start'])
def test_synthetic_fixture_permission_and_formal_approved_two_card_identity(permission_fixture, operation):
    from semantic_retrieval.full_data.authorization import assert_full_data_authorized
    from semantic_retrieval.full_data.runtime import FullDataSettings, manifest_config
    from full_data.test_runtime import fixture_clip
    from semantic_retrieval.models.clip import ClipConfig
    repo, registry, manifest = permission_fixture
    cfg = FullDataSettings('cpu_fixture', 'both', repo/'tiny.npz', 'a'*64, fixture_clip(), 713)
    candidate = dict(manifest, dataset='synthetic', variant='full-data-cpu-fixture', seed=713,
                     input_checkpoints=[dict(path=str(cfg.clip_weights), sha256=cfg.clip_sha256)], dataset_splits=[],
                     config=manifest_config(cfg))
    command = ['python', 'scripts/qualify_full_data_retrieval.py']
    def check(m):
        return assert_full_data_authorized(repo_root=repo, registry_path=registry, manifest=m, operation=operation, command=command)
    check(candidate)
    with pytest.raises(ValueError): check(dict(candidate, dataset='msrvtt'))
    formal = ClipConfig(512, 224, 768, 12, 32, 77, 49408, 512, 8, 12)
    q = FullDataSettings('gpu_qualification', 'both', repo/'random.npz',
                        '4186a61a25330fbc2b567b948f55c8e5660484d393e70817fc3e95b7de72faa8', formal, 713, world_size=2)
    four = dict(candidate, variant='full-data-q0', config=manifest_config(q),
                input_checkpoints=[dict(path=str(q.clip_weights), sha256=q.clip_sha256)],
                environment=dict(precision='fp32', gpu_count=2, distributed_topology='torchrun-2'))
    check(four)
    from dataclasses import replace
    serial = dict(four, config=manifest_config(replace(q, loader_workers=0)))
    with pytest.raises(ValueError, match='loader'):
        check(serial)
    with pytest.raises(ValueError): check(dict(four, environment=dict(four['environment'], gpu_count=4)))
    bad = copy.deepcopy(four); bad['config']['model']['clip_sha256'] = 'b'*64
    with pytest.raises(ValueError): check(bad)
