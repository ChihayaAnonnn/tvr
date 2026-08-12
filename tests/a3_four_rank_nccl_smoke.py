"""Bounded real-NCCL A3 smoke; invoke only through four-rank torchrun."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_task_retrieval as retrieval
from prob_models.contextual_retrieval_risk import (
    ContextualRetrievalRiskSidecar,
    contextual_retrieval_risk_objective,
)


class SmokeModel(torch.nn.Module):
    def __init__(self, sidecar):
        super().__init__()
        self.retriever_anchor = torch.nn.Parameter(
            torch.tensor([1.0]), requires_grad=False
        )
        self.contextual_risk_sidecar = sidecar


def _all_gather(tensor):
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def main():
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("A3 NCCL smoke requires exactly four ranks")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(100 + rank)

    local_text = torch.nn.functional.normalize(
        torch.arange(rank * 16, rank * 16 + 16, device=device, dtype=torch.float32)
        .reshape(2, 8)
        .add(1.0),
        dim=-1,
    )
    local_video = torch.roll(local_text, shifts=1, dims=1)
    global_text = _all_gather(local_text)
    global_video = _all_gather(local_video)

    registered_logits = torch.zeros(8, 8, device=device, dtype=torch.float32)
    registered_logits[torch.arange(8), torch.arange(8)] = 5.0
    for first in (0, 2):
        registered_logits[first, first] = 3.0
        registered_logits[first + 1, first + 1] = 3.0
        registered_logits[first, first + 1] = 4.0
        registered_logits[first + 1, first] = 4.0
    local_logits = registered_logits[rank * 2 : rank * 2 + 2]
    global_logits = _all_gather(local_logits)
    positive_mask = torch.eye(8, device=device, dtype=torch.bool)

    sidecar = ContextualRetrievalRiskSidecar(
        embed_dim=8,
        projection_dim=8,
        hidden_dim=8,
        topk=4,
        temperature=1.0,
        dropout=0.1,
    ).to(device=device, dtype=torch.float32)
    ddp_sidecar = torch.nn.parallel.DistributedDataParallel(
        sidecar, device_ids=[local_rank], broadcast_buffers=False
    )
    model = SmokeModel(sidecar)
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=1e-4)
    scheduler = retrieval._build_warmup_cosine_scheduler(
        optimizer, warmup_proportion=0.1, total_steps=5
    )

    optimizer.zero_grad()
    loss, terms = contextual_retrieval_risk_objective(
        ddp_sidecar,
        global_logits,
        global_text,
        global_video,
        positive_mask,
        fit_weight=0.25,
        rank_weight=0.25,
    )
    loss.backward()
    sidecar_grad_norm = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in sidecar.parameters()
            if parameter.grad is not None
        )
    )
    if not torch.isfinite(loss) or not torch.isfinite(sidecar_grad_norm):
        raise RuntimeError("non-finite A3 smoke loss/gradient")
    if float(sidecar_grad_norm) <= 0:
        raise RuntimeError("A3 smoke produced zero sidecar gradient")
    if model.retriever_anchor.grad is not None:
        raise RuntimeError("A3 smoke leaked gradient outside the sidecar")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("A3 smoke model is not exact FP32")
    optimizer.step()
    scheduler.step()

    rng_by_rank = retrieval._gather_distributed_rng_states()
    dist.barrier()
    if rank == 0:
        output_dir = Path(os.environ["A3_SMOKE_OUTPUT_DIR"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
        retrieval.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
        args = SimpleNamespace(
            output_dir=str(output_dir),
            contextual_risk_sidecar_only=True,
            csd_logvar_sidecar_only=False,
        )
        model_path = Path(
            retrieval.save_model(
                0,
                args,
                model,
                optimizer,
                tr_loss=float(loss),
                scheduler=scheduler,
                global_step=1,
                scheduler_horizon_epochs=5,
                scheduler_total_steps=5,
                optimizer_steps_per_epoch=1,
                rng_state_by_rank=rng_by_rank,
            )
        )
        optimizer_path = Path(retrieval._optimizer_checkpoint_path(0, args))
        marker = retrieval._validate_checkpoint_publication(
            retrieval._checkpoint_publication_marker_path(0, args),
            epoch=0,
            model_path=model_path,
            optimizer_path=optimizer_path,
        )
        payload = torch.load(optimizer_path, map_location="cpu")
        if sorted(int(item) for item in payload["rng_state_by_rank"]) != [0, 1, 2, 3]:
            raise RuntimeError("A3 smoke checkpoint omitted rank RNG state")
        print(json.dumps({
            "world_size": dist.get_world_size(),
            "gathered_logits_shape": list(global_logits.shape),
            "gathered_text_shape": list(global_text.shape),
            "gathered_video_shape": list(global_video.shape),
            "loss": float(loss),
            "sidecar_grad_norm": float(sidecar_grad_norm),
            "non_a3_gradients": 0,
            "parameter_dtype": "float32",
            "t2v_failure_rate": float(terms["t2v"]["failure_rate"]),
            "v2t_failure_rate": float(terms["v2t"]["failure_rate"]),
            "publication_model_sha256": marker["model_sha256"],
            "publication_optimizer_sha256": marker["optimizer_sha256"],
            "rng_ranks": sorted(int(item) for item in rng_by_rank),
            "jsfusion_invocations": 0,
        }, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
