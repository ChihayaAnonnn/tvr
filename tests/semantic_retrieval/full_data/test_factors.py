import numpy as np
import pytest
import torch


def test_offset_sampling_preserves_budget_short_video_and_replay():
    from semantic_retrieval.full_data.factors import offset_frame_indices
    from semantic_retrieval.full_data.loader import official_frame_indices
    for n in (1, 8, 29, 30, 31, 1000):
        base = official_frame_indices(n, 30., 12)
        for epoch in range(5):
            idx = offset_frame_indices(n, 30., 12, seed=713, epoch=epoch, query_id='v:q')
            assert len(idx) == len(base) and len(set(idx)) == len(idx)
            assert all(0 <= i < n for i in idx)
            assert idx == offset_frame_indices(n, 30., 12, seed=713, epoch=epoch, query_id='v:q')
    variants = {tuple(offset_frame_indices(1000,30.,12,seed=713,epoch=e,query_id='v:q')) for e in range(20)}
    assert len(variants)>1


def test_caption_shuffle_keeps_exact_coverage_steps_and_duplicate_exposure():
    from semantic_retrieval.full_data.factors import caption_batches
    from semantic_retrieval.full_data.training import unique_video_batches
    ids = [f'v{i}' for i in range(16) for _ in range(20)]
    kwargs = dict(global_batch=12, world_size=2, seed=713, epoch=0)
    shuffled = caption_batches(ids, **kwargs)
    unique = unique_video_batches(ids, **kwargs)
    assert [len(b) for b in shuffled] == [len(b) for b in unique]
    assert sorted(i for b in shuffled for i in b) == list(range(len(ids)))
    assert any(len({ids[i] for i in b}) < len(b) for b in shuffled)
    assert shuffled == caption_batches(ids, **kwargs)
    assert shuffled != caption_batches(ids, **dict(kwargs,epoch=1))
    with pytest.raises(ValueError):caption_batches(ids[:-1],**kwargs)


@pytest.mark.parametrize('method',['uatvr','clip4clip'])
def test_registered_caption_policy_retains_official_diagonal_loss(method):
    from semantic_retrieval.models.clip import ClipBackbone, ClipConfig
    from semantic_retrieval.full_data.models import BaselineModel
    clip=ClipBackbone(ClipConfig(64,16,64,1,4,32,128,64,1,1,fixture=True))
    model=BaselineModel(method,clip.config,clip.state_dict(),temporal_layers=1,batching='caption_shuffle_v1')
    ids=torch.tensor([[126,4,127,0],[126,5,127,0]])
    frames=torch.randn(1,3,3,16,16).repeat(2,1,1,1,1)
    out=model(ids,ids!=0,frames,torch.ones(2,3,dtype=torch.bool),torch.zeros(2,dtype=torch.long))
    ce=(torch.nn.functional.cross_entropy(out['logits'],torch.arange(2))+torch.nn.functional.cross_entropy(out['logits'].T,torch.arange(2)))/2
    torch.testing.assert_close(ce,out['contrastive']);out['loss'].backward()
    assert torch.isfinite(model.clip.text_projection.grad).all()


def test_factor_settings_roundtrip_and_scope():
    from dataclasses import replace
    from pathlib import Path
    from semantic_retrieval.full_data.runtime import FullDataSettings,manifest_config,settings_from_manifest
    from semantic_retrieval.models.clip import ClipConfig
    from semantic_retrieval.full_data.factors import validate_factor_permission
    cfg=FullDataSettings('cpu_fixture','both',Path('/tmp/clip.npz'),'a'*64,
        ClipConfig(64,16,64,1,4,32,128,64,1,1,fixture=True),713,factor_arm='caption_shuffle')
    assert settings_from_manifest(dict(config=manifest_config(cfg)))==cfg
    with pytest.raises(ValueError):replace(cfg,factor_arm='both')
    with pytest.raises(ValueError):validate_factor_permission({},cfg,'full-data-cpu-fixture')


def test_real_decoder_augmented_pixels_and_validation_guard(tmp_path):
    import cv2,hashlib
    from semantic_retrieval.full_data.loader import CaptionDataset
    p=tmp_path/'clip.avi'; w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),30.,(32,32))
    assert w.isOpened()
    for i in range(90):w.write(np.full((32,32,3),i*2,dtype=np.uint8))
    w.release()
    row=dict(split='train',path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),frame_count=90,fps=30.)
    captions=[dict(query_id='v:q',official_video_id='v',query_text='a changing image')]
    a=CaptionDataset({'v':row},captions,temporal_offset=True,seed=713)
    b=CaptionDataset({'v':row},captions)
    observed=[]
    for epoch in range(4):
        a.epoch=epoch; sample=a[0];assert sample[3].sum()==b[0][3].sum()==3
        assert torch.equal(sample[2],a[0][2]); observed.append(sample[2])
    assert any(not torch.equal(x,b[0][2]) for x in observed)
    with pytest.raises(ValueError,match='training-only'):
        CaptionDataset({'v':dict(row,split='val')},captions,temporal_offset=True)


def test_terminal_binding_rejects_manifest_and_inventory_tampering():
    from semantic_retrieval.full_data.factors import validate_terminal_binding
    from experiment_artifacts import manifest_sha256
    m=dict(run_id='msrvtt-baseline-reference-seed000-20260814T020304Z-000000000000',
        study='contextual-risk-v2',dataset='msrvtt',stage='baseline',variant='reference',seed=0,
        artifact_path='/tmp/artifacts/contextual-risk-v2/msrvtt/baseline/example',registry_path='experiments/contextual-risk-v2/registry.jsonl',
        git_sha='0'*40,tracked_clean=True,source_patch_sha256=None,command_sha256='0'*64,
        environment=dict(python='3.10',pytorch='2.1',cuda='12.1',gpu_model='test',gpu_count=1,precision='fp32',distributed_topology='single-process'),
        input_checkpoints=[],dataset_splits=[],config=dict(model={},loss={},optimizer={},scheduler={},evaluation={}),
        parent_run_ids=[],selection_lineage=[],validation_authorized=True,test_authorized=False,expected_scalar_outputs=[],array_references=[])
    h=manifest_sha256(m)
    r=dict(run_id=m['run_id'],event='completed',manifest_sha256=h)
    s=dict(state='completed',output_hashes={'metrics.json':'a'*64})
    summary=dict(run_id=m['run_id'],state='completed',manifest_sha256=h,output_hashes=s['output_hashes'])
    validate_terminal_binding(r,m,s,summary)
    with pytest.raises(ValueError):validate_terminal_binding(r,dict(m,config=dict(arm='other')),s,summary)
    with pytest.raises(ValueError):validate_terminal_binding(r,m,dict(s,output_hashes={}),summary)


def test_campaign_lock_rejects_duplicate_owner(tmp_path):
    from semantic_retrieval.full_data.controller import campaign_lock
    with campaign_lock(tmp_path/'factor.lock'):
        with pytest.raises(RuntimeError):
            with campaign_lock(tmp_path/'factor.lock'):pass


def test_factor_reservation_rejects_duplicate_and_cap():
    from semantic_retrieval.full_data.factors import validate_factor_slot
    validate_factor_slot([],('uatvr','control',713),6)
    with pytest.raises(ValueError):validate_factor_slot([('uatvr','control',713)],('uatvr','control',713),6)
    with pytest.raises(ValueError):validate_factor_slot([('uatvr','control',713)],('uatvr','temporal_offset',713),1)
