"""Test-only mutations caught: duplicate/drop test rows, mixed split, loose checkpoint load."""
import csv
import json
from pathlib import Path

import pytest
import torch


def test_jsfusion_membership_is_complete_unique_and_disjoint(tmp_path):
    from semantic_retrieval.full_data.final import read_test_csv
    p=tmp_path/'test.csv'
    def write(rows):
        with p.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['video_id','sentence']);w.writeheader();w.writerows(rows)
    write([dict(video_id='v1',sentence='a'),dict(video_id='v2',sentence='b')])
    rows=read_test_csv(p,expected_count=2,forbidden_ids={'v3'})
    assert [(r['official_video_id'],r['query_text']) for r in rows]==[('v1','a'),('v2','b')]
    with pytest.raises(ValueError):read_test_csv(p,expected_count=3,forbidden_ids=set())
    with pytest.raises(ValueError):read_test_csv(p,expected_count=2,forbidden_ids={'v2'})
    write([dict(video_id='v1',sentence='a'),dict(video_id='v1',sentence='b')])
    with pytest.raises(ValueError):read_test_csv(p,expected_count=2,forbidden_ids=set())


def test_final_checkpoint_is_strict_model_only_and_does_not_restore_rng(tmp_path):
    from semantic_retrieval.full_data.final import load_selected_model
    from semantic_retrieval.full_data.training import save_state
    from experiment_artifacts import ArtifactArrayStore
    model=torch.nn.Linear(2,2)
    optimizer=torch.optim.Adam(model.parameters())
    root=tmp_path/'run';(root/'checkpoints').mkdir(parents=True)
    path=root/'checkpoints/epoch-01.pt'
    identity=dict(source_sha='a'*40,recipe_sha256='b'*64,input_sha256='c'*64,method='uatvr',world_size=1)
    save_state(path,model,optimizer,identity,dict(epoch=1,batch_cursor=0,update=1),ArtifactArrayStore(root/'arrays'))
    clone=torch.nn.Linear(2,2)
    before=torch.get_rng_state().clone()
    load_selected_model(path,clone,identity,expected_epoch=1)
    assert torch.equal(torch.get_rng_state(),before)
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):load_selected_model(path,clone,dict(identity,method='clip4clip'),expected_epoch=1)
    with pytest.raises(ValueError):load_selected_model(path,clone,identity,expected_epoch=2)
    with pytest.raises(ValueError):load_selected_model(path,torch.nn.Linear(3,2),identity,expected_epoch=1)


def test_successful_test_cannot_be_repeated_but_other_seed_is_independent():
    from semantic_retrieval.full_data.final import reject_duplicate_test
    records=[dict(event='completed',run_id='prior',stage='final',variant='test-uatvr',seed=713)]
    # Archive records do not carry stage/variant; real identifiers encode these.
    records[0]['run_id']='msrvtt-final-test-uatvr-seed713-20261012T000000Z-aaaaaaaaaaaa'
    with pytest.raises(ValueError):reject_duplicate_test(records,method='uatvr',seed=713)
    reject_duplicate_test(records,method='uatvr',seed=714)
    reject_duplicate_test(records,method='clip4clip',seed=713)
