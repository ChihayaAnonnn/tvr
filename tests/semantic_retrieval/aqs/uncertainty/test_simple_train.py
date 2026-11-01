import copy
import json

import pytest
import torch


def test_cohort_preserves_split_and_rejects_overlap(tmp_path):
    from semantic_retrieval.aqs.fusion_train import read_cohort
    videos = {str(i): dict(video_id=str(i), split='train' if i < 2 else 'val',
                          source_component=str(i), path=str(tmp_path/(str(i)+'.mp4')),
                          sha256='a'*64, frame_count=12, fps=24.) for i in range(4)}
    for row in videos.values():
        from pathlib import Path
        Path(row['path']).touch()
    captions = {split: [dict(query_id=f'{i}:0', caption_id='0', official_video_id=str(i), query_text='a video')
                       for i in indices] for split, indices in [('train', range(2)), ('val', range(2, 4))]}
    path = tmp_path/'cohort.json'
    path.write_text(json.dumps(dict(videos=videos, captions=captions)))
    actual, rows = read_cohort(path, train_videos=2, val_videos=2, captions_per_video=1)
    assert actual == videos and rows == captions
    captions['val'][0]['official_video_id'] = '0'
    path.write_text(json.dumps(dict(videos=videos, captions=captions)))
    with pytest.raises(ValueError):
        read_cohort(path, train_videos=2, val_videos=2, captions_per_video=1)


def test_simple_checkpoint_restores_next_update_and_rejects_changed_scoring(tmp_path, fusion_factory, paired_batch):
    from semantic_retrieval.aqs.fusion_train import save_checkpoint, load_checkpoint
    from semantic_retrieval.aqs.fusion_training import fusion_step
    from semantic_retrieval.full_data.training import make_optimizer
    from semantic_retrieval.full_data.qualification import optimizer_fingerprint
    from semantic_retrieval.runtime.checkpoint import model_sha256
    config = dict(eta=.1, seed=713, epochs=5, global_batch=2, output=str(tmp_path))
    model = fusion_factory()
    optimizer = make_optimizer(model, learning_rate=5e-5, total_steps=10)
    fusion_step(model, model, optimizer, paired_batch)
    fusion_step(model, model, optimizer, paired_batch)
    progress = dict(epoch=0, batch_cursor=2, update=2)
    path = tmp_path/'step2.pt'
    path.with_suffix('.pt.tmp').write_bytes(b'interrupted old write')
    save_checkpoint(path, model, optimizer, config, progress, [])
    expected = fusion_step(model, model, optimizer, paired_batch)
    expected_model, expected_opt = model_sha256(model), optimizer_fingerprint(optimizer.state_dict())
    fresh = fusion_factory()
    opt = make_optimizer(fresh, learning_rate=5e-5, total_steps=10)
    assert load_checkpoint(path, fresh, opt, config)['progress'] == progress
    assert fusion_step(fresh, fresh, opt, paired_batch) == expected
    assert model_sha256(fresh) == expected_model
    assert optimizer_fingerprint(opt.state_dict()) == expected_opt
    before = model_sha256(fresh)
    changed = dict(config, eta=.2)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(path, fresh, opt, changed)
    assert model_sha256(fresh) == before
    changed = dict(config, output='/different/run')
    assert load_checkpoint(path, fresh, opt, changed)['progress'] == progress


def test_checkpoint_retry_preserves_existing_file(tmp_path):
    from semantic_retrieval.aqs.fusion_train import available_checkpoint
    path = tmp_path/'epoch-01.pt'; path.write_bytes(b'previous checkpoint')
    replacement = available_checkpoint(path)
    assert replacement != path and not replacement.exists()
    assert path.read_bytes() == b'previous checkpoint'


def test_diagnostic_reduction_preserves_rank():
    from semantic_retrieval.aqs.fusion_train import global_losses
    row = dict(loss=1., contrastive=.7, aqs=.2, mil=.1, kl=.16, aqs_weighted=.04,
               diagnostic_rank=3., video_variance_update_norm=.005)
    saved = copy.deepcopy(row)
    reduced = global_losses(row, torch.device('cpu'))
    assert row == saved
    assert reduced['loss'] == 1. and 'diagnostic_rank' not in reduced
