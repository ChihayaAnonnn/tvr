"""Measure how much gradient each loss term actually delivers to the backbone.

The A0/A1/A3-fixed runs produced identical DSA loss trajectories, which has two
very different explanations:

  (a) the RSPR terms reach the shared trunk but are too weak to move it
      (a weight-calibration problem), or
  (b) the RSPR terms never reach the shared trunk at all
      (a broken-graph implementation bug).

This probe runs one real training step and reports, per loss term, the gradient
norm w.r.t. the CLIP trunk parameters and w.r.t. the trunk activations that feed
RSPR. A term with a nonzero-but-small trunk norm is case (a); an exactly zero
norm is case (b).

Run with a single GPU, e.g.

    CUDA_VISIBLE_DEVICES=1 python scripts/probe_rspr_gradient_flow.py \
        --init_model <ckpt> --rspr_mode stochastic ...
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main_task_retrieval as mtr  # noqa: E402
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer  # noqa: E402


def _grad_norm(loss, tensors):
    """L2 norm of d(loss)/d(tensors); None means the graph never connects."""

    tensors = [t for t in tensors if t is not None and t.requires_grad]
    if not tensors or not torch.is_tensor(loss) or not loss.requires_grad:
        return float("nan"), 0
    grads = torch.autograd.grad(
        loss, tensors, retain_graph=True, allow_unused=True
    )
    total = torch.zeros((), device=loss.device, dtype=torch.float64)
    connected = 0
    for grad in grads:
        if grad is not None:
            total = total + grad.double().square().sum()
            connected += 1
    return float(total.sqrt()), connected


def main() -> None:
    args = mtr.get_args()
    args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.11")
        os.environ.setdefault("MASTER_PORT", "29611")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl")
    args = mtr.set_seed_logger(args)
    device, n_gpu = mtr.init_device(args, args.local_rank)
    args.pin_memory = torch.cuda.is_available()

    tokenizer = ClipTokenizer()
    model = mtr.init_model(args, device, n_gpu, args.local_rank)
    model = model.module if hasattr(model, "module") else model
    model.train()

    train_bundle, _eval, _len, _split = mtr.prepare_requested_dataloaders(
        args, tokenizer
    )
    train_dataloader = train_bundle[0]

    # Capture the un-detached loss components that _assemble_training_loss folds
    # into a single scalar.
    captured = {}
    original = type(model)._assemble_training_loss

    def spy(self, dsa_loss, probability_loss, rank_loss, rspr_output, **kwargs):
        captured.update(
            dsa=dsa_loss,
            prob=probability_loss,
            rank=rank_loss,
            anchor=rspr_output.anchor_kl,
            scales=kwargs,
        )
        return original(self, dsa_loss, probability_loss, rank_loss, rspr_output, **kwargs)

    type(model)._assemble_training_loss = spy

    # Trunk activations feeding RSPR; hooked so we can see the split point.
    activations = {}
    original_similarity = type(model).get_similarity_logits

    def similarity_spy(self, sequence_output, text_token, visual_output, *rest, **kw):
        text_token.retain_grad()
        visual_output.retain_grad()
        activations["text_token"] = text_token
        activations["visual_output"] = visual_output
        return original_similarity(self, sequence_output, text_token, visual_output, *rest, **kw)

    type(model).get_similarity_logits = similarity_spy

    batch = next(iter(train_dataloader))
    batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
    input_ids, input_mask, segment_ids, video, video_mask, group_ids = (
        mtr._unpack_train_batch(batch)
    )
    print(f"[probe] batch rows={input_ids.shape[0]} group_ids={'yes' if group_ids is not None else 'no'}")

    total = model(
        input_ids,
        segment_ids,
        input_mask,
        video,
        video_mask,
        group_ids=group_ids,
        rspr_prob_scale=1.0,
        rspr_rank_scale=1.0,
        rspr_anchor_scale=1.0,
    )
    print(f"[probe] total loss={float(total):.4f}  scales={captured.get('scales')}")

    trainable_trunk = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("clip.") and parameter.requires_grad
    ]
    rspr_params = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("rspr.") and parameter.requires_grad
    ]
    print(
        f"[probe] trainable clip tensors={len(trainable_trunk)}  "
        f"rspr tensors={len(rspr_params)}"
    )

    weights = {
        "dsa": 1.0,
        "prob": args.rspr_prob_weight,
        "rank": args.rspr_rank_weight,
        "anchor": args.rspr_anchor_weight,
    }

    print("\n=== gradient norm delivered to the CLIP trunk (weighted) ===")
    print(f"  {'term':<8} {'weight':>8} {'raw loss':>10} "
          f"{'|d/d clip|':>13} {'connected':>10} {'|d/d text_tok|':>15} {'|d/d vis_out|':>14}")
    trunk_norms = {}
    for name, weight in weights.items():
        loss = captured.get(name)
        if loss is None:
            continue
        weighted = weight * loss
        clip_norm, connected = _grad_norm(weighted, trainable_trunk)
        text_norm, _ = _grad_norm(weighted, [activations.get("text_token")])
        video_norm, _ = _grad_norm(weighted, [activations.get("visual_output")])
        trunk_norms[name] = clip_norm
        print(
            f"  {name:<8} {weight:>8.4g} {float(loss):>10.4f} "
            f"{clip_norm:>13.4e} {connected:>6}/{len(trainable_trunk):<4} "
            f"{text_norm:>15.4e} {video_norm:>14.4e}"
        )

    dsa_norm = trunk_norms.get("dsa", float("nan"))
    print("\n=== RSPR share of the trunk gradient ===")
    for name in ("prob", "rank", "anchor"):
        if name in trunk_norms:
            print(f"  |grad_{name}| / |grad_dsa| = {trunk_norms[name] / dsa_norm:.4e}")
    rspr_sum = sum(trunk_norms.get(k, 0.0) for k in ("prob", "rank", "anchor"))
    print(f"  (prob+rank+anchor) / dsa        = {rspr_sum / dsa_norm:.4e}")

    print("\n=== same terms measured on RSPR's own parameters (sanity) ===")
    for name, weight in weights.items():
        loss = captured.get(name)
        if loss is None:
            continue
        norm, connected = _grad_norm(weight * loss, rspr_params)
        print(f"  {name:<8} |d/d rspr| = {norm:.4e}  connected={connected}/{len(rspr_params)}")


if __name__ == "__main__":
    main()
