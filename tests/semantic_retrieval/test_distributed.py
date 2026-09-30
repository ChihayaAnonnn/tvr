import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world, rendezvous, mismatch):
    from semantic_retrieval.runtime.distributed import gather_metadata, gather_with_grad

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    try:
        if mismatch:
            with pytest.raises(ValueError, match="shape"):
                gather_with_grad(torch.ones(rank + 1, 2, requires_grad=True))
            return
        value = torch.tensor([[float(rank + 1), float(rank + 2)]], requires_grad=True)
        gathered = gather_with_grad(value)
        torch.testing.assert_close(gathered, torch.tensor([[1.0, 2.0], [2.0, 3.0]]))
        # Each rank contributes 1/world of the replicated global objective.
        (gathered.square().sum() / world).backward()
        torch.testing.assert_close(value.grad, 2 * value.detach())
        ids = gather_metadata(torch.tensor([rank]))
        mask = gather_metadata(torch.tensor([[True, rank == 0]]))
        assert ids.tolist() == [0, 1] and mask.tolist() == [[True, True], [True, False]]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mismatch", [False, True])
def test_two_rank_gloo_gradient_and_shape_contract(tmp_path, mismatch):
    mp.spawn(_worker, args=(2, "file://" + str(tmp_path / "rendezvous"), mismatch), nprocs=2, join=True)


def test_single_process_metadata_has_no_gradient():
    from semantic_retrieval.runtime.distributed import gather_metadata, gather_with_grad

    value = torch.ones(2, 3, requires_grad=True)
    assert gather_with_grad(value) is value
    assert not gather_metadata(value).requires_grad
