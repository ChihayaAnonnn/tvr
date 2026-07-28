"""Offline probe: does the learned RSPR uncertainty carry retrievable signal?

Loads one checkpoint, encodes the requested eval split once, then reports
(a) how heteroscedastic the learned sigma is, (b) how large the probabilistic
rerank term is next to the deterministic logits, and (c) whether sigma^2 /
U_pair separate correct from incorrect Top-1 retrievals (AUROC).

Usage mirrors eval.sh; pass the same RSPR flags the checkpoint was trained
with, e.g.

    python scripts/probe_rspr_uncertainty.py --init_model <ckpt> ... --rspr_mode stochastic
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main_task_retrieval as mtr  # noqa: E402
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer  # noqa: E402


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (ties = 0.5)."""

    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks over ties so exact duplicates score 0.5
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(unique.size)
    np.add.at(tie_sum, inverse, ranks)
    ranks = (tie_sum / counts)[inverse]
    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - positives.size * (positives.size + 1) / 2)
        / (positives.size * negatives.size)
    )


def _describe(name: str, values: np.ndarray) -> None:
    print(
        f"  {name:<24} mean={values.mean():.6g}  std={values.std():.6g}  "
        f"cv={values.std() / (abs(values.mean()) + 1e-12):.4f}  "
        f"min={values.min():.6g}  max={values.max():.6g}  "
        f"p05={np.percentile(values, 5):.6g}  p95={np.percentile(values, 95):.6g}"
    )


def main() -> None:
    args = mtr.get_args()
    args.local_rank = 0
    torch.cuda.set_device(args.local_rank)
    # set_seed_logger reads the world size, so stand up a single-rank group.
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.11")
        os.environ.setdefault("MASTER_PORT", "29573")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl")
    args = mtr.set_seed_logger(args)
    device, n_gpu = mtr.init_device(args, args.local_rank)
    args.pin_memory = torch.cuda.is_available()

    tokenizer = ClipTokenizer()
    model = mtr.init_model(args, device, n_gpu, args.local_rank)
    model = model.module if hasattr(model, "module") else model
    model.eval()

    _train, eval_dataloader, _length, split = mtr.prepare_requested_dataloaders(
        args, tokenizer
    )
    print(f"[probe] split={split} batches={len(eval_dataloader)}")

    sequence_outputs, text_tokens, input_masks = [], [], []
    visual_outputs, video_masks = [], []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            if len(batch) == 5:
                input_ids, input_mask, segment_ids, video, video_mask = batch
            else:
                (
                    input_ids,
                    input_mask,
                    segment_ids,
                    *_attrs,
                    video,
                    video_mask,
                ) = batch
            sequence_output, text_token, visual_output = (
                model.get_sequence_visual_output(
                    input_ids, segment_ids, input_mask, video, video_mask
                )
            )
            sequence_outputs.append(sequence_output.cpu())
            text_tokens.append(text_token.cpu())
            input_masks.append(input_mask.cpu())
            visual_outputs.append(visual_output.cpu())
            video_masks.append(video_mask.cpu())

    sequence_output = torch.cat(sequence_outputs).to(device)
    text_token = torch.cat(text_tokens).to(device)
    input_mask = torch.cat(input_masks).to(device)
    visual_output = torch.cat(visual_outputs).to(device)
    video_mask = torch.cat(video_masks).to(device)
    n_text, n_video = sequence_output.size(0), visual_output.size(0)
    print(f"[probe] encoded #text={n_text} #video={n_video}")

    chunk = 64
    with torch.no_grad():
        deterministic_rows = []
        for t_start in range(0, n_text, chunk):
            t_end = min(t_start + chunk, n_text)
            row = []
            for v_start in range(0, n_video, chunk):
                v_end = min(v_start + chunk, n_video)
                logits, *_ = model.get_similarity_logits(
                    sequence_output[t_start:t_end],
                    text_token[t_start:t_end],
                    visual_output[v_start:v_end],
                    input_mask[t_start:t_end],
                    video_mask[v_start:v_end],
                    loose_type=model.loose_type,
                )
                row.append(logits.float().cpu())
            deterministic_rows.append(torch.cat(row, dim=1))
        deterministic = torch.cat(deterministic_rows)

        text_means, text_logvars, text_samples = [], [], []
        for start in range(0, n_text, chunk):
            end = min(start + chunk, n_text)
            dist = model.get_rspr_text_distribution(
                text_token[start:end], input_mask[start:end]
            )
            text_means.append(dist.mean.float().cpu())
            text_logvars.append(dist.logvar.float().cpu())
            text_samples.append(dist.samples.float().cpu())
        video_means, video_logvars, video_samples = [], [], []
        for start in range(0, n_video, chunk):
            end = min(start + chunk, n_video)
            dist = model.get_rspr_video_distribution(
                visual_output[start:end], video_mask[start:end]
            )
            video_means.append(dist.mean.float().cpu())
            video_logvars.append(dist.logvar.float().cpu())
            video_samples.append(dist.samples.float().cpu())

    text_mean = torch.cat(text_means)
    video_mean = torch.cat(video_means)
    text_variance = torch.cat(text_logvars).exp()
    video_variance = torch.cat(video_logvars).exp()
    text_sample = torch.cat(text_samples)
    video_sample = torch.cat(video_samples)

    print("\n=== 1. learned variance: is it input dependent? ===")
    _describe("text sigma^2 (per-dim)", text_variance.numpy().ravel())
    _describe("video sigma^2 (per-dim)", video_variance.numpy().ravel())
    _describe("text sigma^2 (per-sample)", text_variance.mean(dim=-1).numpy())
    _describe("video sigma^2 (per-sample)", video_variance.mean(dim=-1).numpy())

    matcher = model.rspr.matcher
    mean_logits = (
        torch.nn.functional.normalize(text_mean, dim=-1)
        @ torch.nn.functional.normalize(video_mean, dim=-1).T
    )
    top1_by_mean = mean_logits.argmax(dim=1)
    top1_by_det = deterministic.argmax(dim=1)

    with torch.no_grad():
        gt = torch.arange(min(n_text, n_video))
        pair_batches = {
            "gt": (gt, gt),
            "det_top1": (torch.arange(n_text), top1_by_det),
        }
        pair_stats = {}
        for name, (t_idx, v_idx) in pair_batches.items():
            match = matcher.score_pairs(
                text_sample[t_idx].to(device), video_sample[v_idx].to(device)
            )
            pair_stats[name] = (
                match.logits.float().cpu().numpy(),
                match.pair_uncertainty.float().cpu().numpy(),
            )

    print("\n=== 2. rerank term magnitude vs deterministic logits ===")
    det_np = deterministic.numpy()
    top10 = np.sort(det_np, axis=1)[:, -10:]
    det_gap = (top10[:, -1] - top10[:, 0]).mean()
    prob_gt, unc_gt = pair_stats["gt"]
    prob_top1, unc_top1 = pair_stats["det_top1"]
    all_probs = np.concatenate([prob_gt, prob_top1])
    print(f"  deterministic logits         range=[{det_np.min():.3f}, {det_np.max():.3f}]")
    print(f"  deterministic top1-top10 gap mean={det_gap:.4f}")
    print(f"  matcher probability          range=[{all_probs.min():.4f}, {all_probs.max():.4f}] std={all_probs.std():.5f}")
    weight = getattr(args, "rspr_rerank_weight", 0.1)
    print(f"  rerank weight                {weight}")
    print(
        f"  weighted prob spread         {weight * all_probs.std():.6f}  "
        f"vs deterministic gap {det_gap:.4f}  "
        f"=> ratio {weight * all_probs.std() / max(det_gap, 1e-9):.2e}"
    )

    print("\n=== 3. does uncertainty predict retrieval failure? ===")
    n = min(n_text, n_video)
    correct = (top1_by_det[:n] == gt).numpy().astype(int)
    print(f"  deterministic Top-1 accuracy {correct.mean():.4f} over {n} queries")
    signals = {
        "text sigma^2 (per-sample)": text_variance.mean(dim=-1)[:n].numpy(),
        "video sigma^2 (per-sample)": video_variance.mean(dim=-1)[:n].numpy(),
        "U_pair(gt pair)": unc_gt[:n],
        "U_pair(det top1 pair)": unc_top1[:n],
    }
    for name, signal in signals.items():
        # AUROC for "high signal => wrong"; 0.5 means no information.
        score = _auroc(signal, 1 - correct)
        _describe(name, signal)
        print(f"    -> AUROC(predicting a WRONG top-1) = {score:.4f}")

    print("\n=== 4. pure RSPR-mean retrieval vs deterministic ===")
    for label, matrix in (("deterministic (DSA)", deterministic), ("rspr mean", mean_logits)):
        ranks = (matrix >= matrix[torch.arange(n), gt].unsqueeze(1)).sum(dim=1)[:n]
        r1 = (ranks <= 1).float().mean().item() * 100
        r5 = (ranks <= 5).float().mean().item() * 100
        r10 = (ranks <= 10).float().mean().item() * 100
        print(f"  {label:<22} R@1={r1:5.1f}  R@5={r5:5.1f}  R@10={r10:5.1f}")
    agree = (top1_by_mean[:n] == top1_by_det[:n]).float().mean().item()
    print(f"  top-1 agreement (mean vs deterministic) = {agree:.4f}")


if __name__ == "__main__":
    main()
