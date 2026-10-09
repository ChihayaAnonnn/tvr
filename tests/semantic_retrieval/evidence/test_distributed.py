import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _loss_worker(rank, rendezvous):
    from semantic_retrieval.evidence.types import LossTerm
    from semantic_retrieval.evidence.training import global_loss
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        x = torch.tensor(2., requires_grad=True)
        count = torch.tensor(float(rank == 1))
        loss, active = global_loss({"pc": LossTerm(count*x*x, count)}, {"pc": 1.})
        assert active
        loss.backward()
        # DDP平均前的本地梯度：rank0为0，rank1为8；平均为单进程4。
        assert x.grad.item() == (0. if rank == 0 else 8.)
    finally:
        dist.destroy_process_group()

def test_one_rank_unknown_does_not_skip_collectives(tmp_path):
    mp.spawn(_loss_worker, args=("file://" + str(tmp_path/"rendezvous"),), nprocs=2, join=True)

def test_all_unknown_does_not_step():
    from semantic_retrieval.evidence.types import LossTerm
    from semantic_retrieval.evidence.training import EvidenceStepper
    model = torch.nn.Linear(1,1)
    opt = torch.optim.AdamW(model.parameters(),lr=.1)
    sched = torch.optim.lr_scheduler.StepLR(opt,1)
    stepper = EvidenceStepper(model,opt,sched,weights={'pc':1.})
    before = {k:v.clone() for k,v in model.state_dict().items()}
    x = model(torch.ones(1,1)).sum()*0
    result = stepper.step({'pc':LossTerm(x,torch.tensor(0.))})
    assert not result.stepped and result.step == 0
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert not opt.state

def _ddp_weight_worker(rank,rendezvous):
    from torch.nn.parallel import DistributedDataParallel
    from semantic_retrieval.evidence.types import LossTerm
    from semantic_retrieval.evidence.training import EvidenceStepper
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=2)
    try:
        model = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():
            model.weight.fill_(2.)
        wrapped = DistributedDataParallel(model)
        opt = torch.optim.SGD(model.parameters(),lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(opt,100)
        stepper = EvidenceStepper(wrapped,opt,scheduler,weights={'pc':1.})
        output = wrapped(torch.ones(1,1)).squeeze()
        count = torch.tensor(float(rank == 1))
        result = stepper.step({'pc':LossTerm(count*output.square(),count)})
        assert result.stepped and result.step == 1
        torch.testing.assert_close(model.weight,torch.tensor([[1.6]]))
        assert count.item() == float(rank == 1)
    finally:
        dist.destroy_process_group()


def test_real_ddp_matches_global_effective_mean(tmp_path):
    mp.spawn(_ddp_weight_worker,args=('file://'+str(tmp_path/'ddp-rendezvous'),),nprocs=2,join=True)
