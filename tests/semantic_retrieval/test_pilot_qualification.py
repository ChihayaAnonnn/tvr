from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_runtime_artifacts import commit_reservation, reservation


def qualification_context(tmp_path):
    from semantic_retrieval.config import effective_manifest_config
    from semantic_retrieval.runtime.artifacts import prepare_run, start_run
    from semantic_retrieval.runtime.pilot_qualification import build_pilot_fixture_inputs

    repo, runtime = reservation.__wrapped__(tmp_path, SimpleNamespace(param='success'))
    commit_reservation(repo, runtime)
    prep = prepare_run(runtime, repo_root=repo)
    start_run(prep)
    cfg, refs = build_pilot_fixture_inputs(prep, tiny=True)
    case = tmp_path / 'chain'; case.mkdir()
    repo, runtime = reservation.__wrapped__(case, SimpleNamespace(param=dict(
        config=effective_manifest_config(cfg, eval_split='train'), variant='new-pilot-chain',
        input_checkpoints=refs[:1], dataset_splits=refs[1:],
        environment={**prep.manifest['environment'], 'distributed_topology': 'torchrun-4'})))
    runtime = replace(runtime, eval_split='train')
    cfg = replace(cfg, runtime=runtime)
    commit_reservation(repo, runtime)
    # The scientific protocol has four ranks. CPU unit fixtures explicitly use its global batch.
    import os
    previous = os.environ.get('WORLD_SIZE')
    os.environ['WORLD_SIZE'] = '4'
    try:
        context = prepare_run(runtime, repo_root=repo, system_config=cfg)
        start_run(context)
    finally:
        if previous is None: os.environ.pop('WORLD_SIZE')
        else: os.environ['WORLD_SIZE'] = previous
    return context, cfg


def test_new_raw_v2_fixture_exercises_c2_and_c3_chain(tmp_path):
    from semantic_retrieval.data.episodes import read_episodes
    from semantic_retrieval.runtime.pilot_protocol import validate_q0_report
    from semantic_retrieval.runtime.pilot_qualification import qualify_pilot_chain

    context, cfg = qualification_context(tmp_path)
    episodes = read_episodes(cfg.data.episode_manifest, gallery_version=cfg.data.gallery_version, split='train')
    assert len(episodes) == 16
    assert len(episodes[0].canonical_ids) == 129
    assert all(e.relevance_space == 'benchmark_annotation' and len(e.benchmark_positive_ids) == 1 for e in episodes)
    report = qualify_pilot_chain(context, cfg, device=torch.device('cpu'))
    assert all(report['checks'].values())
    assert report['resume_parameter_max_abs_delta'] == 0
    assert report['resume_optimizer_max_abs_delta'] == 0
    assert report['gradient_probes']['A2']['presence_gradient_norm'] == 0
    assert report['gradient_probes']['A3']['presence_gradient_norm'] > 0
    assert report['gradient_probes']['A4']['presence_gradient_norm'] == 0
    assert report['calibration']['natural']['presence']['status'] == 'available'
    assert report['calibration']['controlled']['presence']['positive_prevalence'] == .5
    assert (context.paths.provenance/'pilot-training-qualification.json').is_file()
    assert (context.paths.provenance/'pilot-qualification.json').is_file()
    with pytest.raises(ValueError, match='four-GPU'):
        validate_q0_report(report, source_sha=context.manifest['git_sha'])


def _cpu_rank(rank, init, context, cfg):
    import torch.distributed as dist
    from semantic_retrieval.runtime.pilot_qualification import qualify_pilot_chain

    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=4)
    try:
        qualify_pilot_chain(context, cfg, device=torch.device('cpu'))
    finally:
        dist.destroy_process_group()


def test_new_chain_four_rank_probe_hooks_and_sampler_restore(tmp_path):
    context, cfg = qualification_context(tmp_path)
    torch.multiprocessing.spawn(_cpu_rank, args=(f'file://{tmp_path / "gloo-init"}', context, cfg), nprocs=4, join=True)
    import json
    report = json.loads((context.paths.provenance/'pilot-qualification.json').read_text())
    assert report['world_size'] == 4
    assert report['checks']['unique_rank_queries']
    assert report['checks']['global_known_counts']
    assert report['checks']['rank_parameter_consistency']


@pytest.mark.parametrize('defect', ['orders', 'relations'])
def test_new_chain_refuses_modified_v2_inputs_before_training(tmp_path, defect):
    from semantic_retrieval.runtime.pilot_qualification import qualify_pilot_chain
    context, cfg = qualification_context(tmp_path)
    path = cfg.train.pilot.candidate_order_path if defect == 'orders' else cfg.data.episode_manifest.parent/'relations.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='hash|SHA'):
        qualify_pilot_chain(context, cfg, device=torch.device('cpu'))
    assert not list(context.paths.checkpoints.glob('*.pt'))


def test_later_calibration_failure_retains_evidence_and_cannot_claim_pass(tmp_path,monkeypatch):
    from semantic_retrieval.runtime import pilot_qualification
    context,cfg=qualification_context(tmp_path)
    def failed_consumer(*args,**kwargs): raise ValueError('planned consumer rejection')
    monkeypatch.setattr(pilot_qualification,'_calibration_probe',failed_consumer)
    with pytest.raises(ValueError,match='after archived training evidence'):
        pilot_qualification.qualify_pilot_chain(context,cfg,device=torch.device('cpu'))
    assert (context.paths.provenance/'pilot-training-qualification.json').is_file()
    assert (context.paths.provenance/'pilot-gradient-qualification.json').is_file()
    assert not (context.paths.provenance/'pilot-qualification.json').exists()
