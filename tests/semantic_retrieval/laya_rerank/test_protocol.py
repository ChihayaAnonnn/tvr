from pathlib import Path
import pytest
from semantic_retrieval.laya_rerank.protocol import assert_laya_authorized, load_fixed

ROOT=Path(__file__).resolve().parents[3]


def test_fixed_and_closed_science():
    fixed=load_fixed(ROOT)
    assert fixed['top_k']==10 and fixed['frames']==4 and fixed['world_size_qualification']==4
    with pytest.raises(ValueError,match='four'):
        assert_laya_authorized(repo_root=ROOT,registry_path=ROOT/'experiments/laya-rerank-validation-v1/registry.jsonl',
            manifest={'study':'laya-rerank-validation-v1','test_authorized':False},operation='reserve',command=['scripts/evaluate_laya_rerank.py'])


def test_alias_cannot_escape_gate():
    with pytest.raises(ValueError,match='independent study'):
        assert_laya_authorized(repo_root=ROOT,registry_path=ROOT/'experiments/contextual-risk-v2/registry.jsonl',
            manifest={'study':'contextual-risk-v2'},operation='start',command=['python','scripts/evaluate_laya_rerank.py'])


def test_unrelated_study_untouched():
    assert_laya_authorized(repo_root=ROOT,registry_path=ROOT/'elsewhere',manifest={'study':'other'},operation='reserve',command=['python','unrelated.py'])


def test_missing_manifest_input_binding_fails():
    from semantic_retrieval.laya_rerank.protocol import validate_input_bindings
    fixed=load_fixed(ROOT)
    with pytest.raises(ValueError,match='bound'):
        validate_input_bindings(ROOT,{'dataset_splits':[],'input_checkpoints':[]},fixed,{'clip_weights':'/clip','clip_sha256':'a'})
