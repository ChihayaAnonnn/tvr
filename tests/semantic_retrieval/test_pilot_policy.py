"""The new namespace cannot borrow old study authorization or bypass CPU-only gates."""
from pathlib import Path
import pytest
import yaml

NAME = 'semantic-answerability-pilot-v1'


def fixture(tmp_path, status='design_approved'):
    control = tmp_path / 'experiments' / NAME
    control.mkdir(parents=True)
    (control / 'study.yaml').write_text(yaml.safe_dump(dict(
        study=NAME, status=status, scientific_authorized=False, stage0_passed=False,
        control_plane=f'experiments/{NAME}', data_plane=f'artifacts/{NAME}',
        registry=f'experiments/{NAME}/registry.jsonl')))
    manifest = dict(study=NAME, dataset='msrvtt', stage='baseline', variant='source_inventory',
                    environment={'gpu_count': 0}, test_authorized=False,
                    artifact_path=str(tmp_path / 'artifacts' / NAME / 'msrvtt' / 'baseline' / 'run'),
                    registry_path=f'experiments/{NAME}/registry.jsonl')
    return control / 'registry.jsonl', manifest


@pytest.mark.parametrize('operation', ['reserve', 'start'])
def test_cpu_preparation_allowed(tmp_path, operation):
    from semantic_retrieval.runtime.pilot_policy import assert_pilot_authorized
    registry, manifest = fixture(tmp_path)
    assert_pilot_authorized(repo_root=tmp_path, registry_path=registry, manifest=manifest, operation=operation)
    manifest['variant'] = 'benchmark_compile'
    assert_pilot_authorized(repo_root=tmp_path, registry_path=registry, manifest=manifest, operation=operation)


@pytest.mark.parametrize('operation', ['reserve', 'start'])
@pytest.mark.parametrize('defect', ['draft', 'gpu', 'test', 'dataset', 'stage', 'unknown', 'A0', 'A1', 'A2', 'A3', 'A4', 'P1', 'renamed', 'path'])
def test_unauthorized_runs_rejected(tmp_path, operation, defect):
    from semantic_retrieval.runtime.pilot_policy import assert_pilot_authorized
    registry, manifest = fixture(tmp_path, 'draft' if defect == 'draft' else 'design_approved')
    if defect == 'gpu':
        manifest['environment']['gpu_count'] = 4
    elif defect == 'test':
        manifest['test_authorized'] = True
    elif defect == 'dataset':
        manifest['dataset'] = 'msvd'
    elif defect == 'stage':
        manifest['stage'] = 'final'
    elif defect == 'renamed':
        manifest['study'] = 'fixture-study'
    elif defect == 'path':
        manifest['artifact_path'] = str(tmp_path / 'artifacts' / 'wrong' / 'msrvtt' / 'baseline' / 'run')
    elif defect != 'draft':
        manifest['variant'] = defect
    with pytest.raises(ValueError):
        assert_pilot_authorized(repo_root=tmp_path, registry_path=registry, manifest=manifest, operation=operation)
