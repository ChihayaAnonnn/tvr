from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from test_runtime_artifacts import reservation, commit_reservation
from semantic_retrieval.evidence.checkpoint import EvidenceProgress, evidence_config_sha256, save_evidence_checkpoint, load_evidence_checkpoint


@pytest.fixture
def checkpoint_context(tmp_path,config_factory):
    from evidence.archive_fixture import reserve_evidence_fixture
    from semantic_retrieval.evidence.runtime import prepare_evidence_run
    from semantic_retrieval.runtime.artifacts import start_run
    repo,cfg = reserve_evidence_fixture(tmp_path,config_factory(stage='cpu_fixture'))
    context = prepare_evidence_run(cfg,repo_root=repo)
    start_run(context)
    return context,cfg


def test_resume_next_noise_sampler_and_parameters_exact(checkpoint_context):
    from semantic_retrieval.evidence.sampling import SamplingState
    context,cfg = checkpoint_context
    torch.manual_seed(4)
    model = torch.nn.Linear(2,1)
    opt = torch.optim.AdamW(model.parameters(),lr=.01)
    sched = torch.optim.lr_scheduler.StepLR(opt,1)
    sampling = SamplingState(cfg.sampling,mode='train',model_sha256='a'*64)
    def step():
        opt.zero_grad(); loss = model(torch.randn(2,2)).square().sum(); loss.backward(); opt.step(); sched.step()
    step()
    progress = EvidenceProgress('cpu_fixture',1,0,1,{'order':(0,1),'cursor':1},sampling.state_dict(),'a'*64,None,None)
    path = save_evidence_checkpoint(context,{'matcher':model},opt,sched,cfg,progress)
    step()
    expected = {k:v.clone() for k,v in model.state_dict().items()}
    next_noise = sampling.noise((('q','a'),),device='cpu',dtype=torch.float32)
    recovered = load_evidence_checkpoint(path,{'matcher':model},expected_config_sha256=evidence_config_sha256(cfg),expected_parents=cfg.parents)
    opt.load_state_dict(recovered.optimizer_state); sched.load_state_dict(recovered.scheduler_state)
    sampling.load_state_dict(recovered.progress.mc_state)
    assert recovered.progress.sampler_state['cursor'] == 1
    step()
    assert all(torch.equal(v,expected[k]) for k,v in model.state_dict().items())
    assert torch.equal(sampling.noise((('q','a'),),device='cpu',dtype=torch.float32),next_noise)


def test_partial_sidecar_and_wrong_producer_rejected(checkpoint_context):
    context,cfg = checkpoint_context
    model = torch.nn.Linear(2,1); opt = torch.optim.AdamW(model.parameters()); sched = torch.optim.lr_scheduler.StepLR(opt,1)
    progress = EvidenceProgress('cpu_fixture',0,0,0,{'cursor':0},{'identity':'a'*64},'a'*64,None,None)
    path = save_evidence_checkpoint(context,{'matcher':model},opt,sched,cfg,progress)
    with pytest.raises(ValueError):
        load_evidence_checkpoint(path,{'matcher':model},expected_config_sha256='b'*64,expected_parents=cfg.parents)
    path.with_suffix('.json').unlink()
    with pytest.raises(ValueError):
        load_evidence_checkpoint(path,{'matcher':model},expected_config_sha256=evidence_config_sha256(cfg),expected_parents=cfg.parents)


def test_parent_contract_rejects_shape_compatible_wrong_science(config_factory):
    import copy
    import pytest
    from semantic_retrieval.evidence.config import evidence_manifest_config
    from semantic_retrieval.evidence.execution import validate_parent_contract
    cfg = config_factory()
    parent = {'config':evidence_manifest_config(cfg,eval_split='train')}
    parent['config']['evaluation']['evidence_stage'] = 'matcher'
    validate_parent_contract(cfg,'matcher',parent)
    for field,value in (('scale_floor',.2),('variant','fixed_scale')):
        wrong = copy.deepcopy(parent);wrong['config']['model'][field] = value
        with pytest.raises(ValueError,match='parent'):
            validate_parent_contract(cfg,'matcher',wrong)
    wrong = copy.deepcopy(parent);wrong['config']['model']['base']['clip_sha256'] = 'f'*64
    with pytest.raises(ValueError,match='parent'):
        validate_parent_contract(cfg,'matcher',wrong)
