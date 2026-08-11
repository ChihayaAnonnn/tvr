"""Diagnose whether CRF-trained u_video correlates with real retrieval failure.

Correct implementation: reuse `main_task_retrieval.eval_epoch` for the sim
matrix (was previously reimplementing it and getting wrong R@1s), and run
a second dedicated pass over the same test loader to collect `u_video` per
video via `model.score_video_risk`.

Goal-driven decision rule (from /goal):
  - max Spearman(u_video, rank) >= 0.20 -> keep uncertainty claim
  - < 0.20                              -> drop uncertainty claim
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-csv", default="/data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv")
    p.add_argument("--features-path", default="/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps/")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def _configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import main_task_retrieval as mtr
    mtr.logger = logging.getLogger("diag")


def _build_test_ns(cli_args, restored_args):
    """Merge restored training args with eval-only overrides."""
    ns_dict = dict(vars(restored_args))
    ns_dict.update({
        "test_csv": cli_args.test_csv,
        "features_path": cli_args.features_path,
        "batch_size_val": cli_args.batch_size,
        "num_thread_reader": 4,
        "prefetch_factor": 2,
        "pin_memory": False,
        "use_attributes": False,
        "local_rank": 0,
        "n_gpu": 1,
        "world_size": 1,
        "DSL": False,
        "eval_vid_chunk_size": 128,
        "log_mus_scores": False,
        "do_train": False,
        "do_eval": True,
        "eval_split": "test",
    })
    return SimpleNamespace(**ns_dict)


def main(argv=None):
    args = parse_args(argv)
    _configure_logging()
    ckpt_dir = Path(args.checkpoint).parent
    training_args_path = ckpt_dir / "training_args.json"
    if not training_args_path.exists():
        raise FileNotFoundError(f"missing {training_args_path}")

    from dataloaders.data_dataloaders import dataloader_msrvtt_test
    from main_task_retrieval import eval_epoch
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
    from scripts.evaluate_crf_ladder import _load_uatvr_for_eval, _restore_task_config

    device = torch.device(args.device)
    print(f"[diag] loading ckpt {args.checkpoint}")
    cli_shim = SimpleNamespace(max_frames=None, feature_framerate=None, image_resolution=224)
    model = _load_uatvr_for_eval(args.checkpoint, cli_shim, device)
    model.eval()

    restored = _restore_task_config(training_args_path, cli_shim)
    tokenizer = ClipTokenizer()
    ns = _build_test_ns(args, restored)
    loader, n_pairs = dataloader_msrvtt_test(ns, tokenizer, subset="test")
    print(f"[diag] test pairs: {n_pairs} | batches: {len(loader)}")

    # ---- Pass 1: retrieval metrics + sim_matrix via the REAL eval_epoch ----
    # Monkey-patch _run_on_single_gpu to capture the sim matrix; that's the
    # only clean way since eval_epoch doesn't expose it and metrics['cols']
    # is tie-inflated for bf16 numerics.
    import main_task_retrieval as _mtr
    _real_run = _mtr._run_on_single_gpu
    _captured = {"sim_rows": None}

    def _capturing_run(*a, **kw):
        rows = _real_run(*a, **kw)
        _captured["sim_rows"] = rows
        return rows
    _mtr._run_on_single_gpu = _capturing_run

    t0 = time.time()
    metrics = eval_epoch(ns, model, loader, device, n_gpu=1)
    _mtr._run_on_single_gpu = _real_run
    r1_t2v = float(metrics["t2v"]["R1"])
    r5_t2v = float(metrics["t2v"]["R5"])
    r10_t2v = float(metrics["t2v"]["R10"])
    mdr_t2v = float(metrics["t2v"]["MR"])
    r1_v2t = float(metrics["v2t"]["R1"])
    r5_v2t = float(metrics["v2t"]["R5"])
    r10_v2t = float(metrics["v2t"]["R10"])
    mdr_v2t = float(metrics["v2t"]["MR"])
    # Build correct diagonal-based ranks from the captured sim matrix
    sim = np.concatenate(_captured["sim_rows"], axis=0).astype(np.float32)
    assert sim.shape == (n_pairs, n_pairs), f"sim shape {sim.shape}"
    diag = np.diag(sim)
    t2v_rank = (sim > diag[:, None]).sum(axis=1)  # 0=perfect
    v2t_rank = (sim > diag[None, :]).sum(axis=0)
    print(f"[diag] eval_epoch done in {time.time()-t0:.1f}s")
    print(f"[diag] T2V R@1={r1_t2v:.1f} R@5={r5_t2v:.1f} R@10={r10_t2v:.1f} MdR={mdr_t2v:.1f}")
    print(f"[diag] V2T R@1={r1_v2t:.1f} R@5={r5_v2t:.1f} R@10={r10_v2t:.1f} MdR={mdr_v2t:.1f}")

    # ---- Pass 2: u_video per video via score_video_risk ----
    u_video_all = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for bid, batch in enumerate(loader):
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            if len(batch) == 5:
                _ids, _mask, _seg, video, video_mask = batch
            elif len(batch) == 8:
                _ids, _mask, _seg, _a1, _a2, _a3, video, video_mask = batch
            else:
                raise ValueError(f"unexpected batch shape: {len(batch)}")
            _, u_video = model.score_video_risk(video, video_mask)
            u_video_all.append(u_video.detach().float().cpu().numpy())
    u_video = np.concatenate(u_video_all)
    assert u_video.shape == (n_pairs,), f"u_video shape {u_video.shape}"
    print(f"[diag] u_video stats: mean={u_video.mean():.4f} std={u_video.std():.4f} "
          f"min={u_video.min():.4f} max={u_video.max():.4f}")

    # ---- Spearman(u_video, retrieval rank) ----
    assert t2v_rank.shape == (n_pairs,), f"t2v_rank shape {t2v_rank.shape}"
    assert v2t_rank.shape == (n_pairs,), f"v2t_rank shape {v2t_rank.shape}"
    rho_t2v, p_t2v = spearmanr(u_video, t2v_rank)
    rho_v2t, p_v2t = spearmanr(u_video, v2t_rank)
    hard_rank = np.maximum(t2v_rank, v2t_rank)
    rho_hard, p_hard = spearmanr(u_video, hard_rank)
    print(f"[diag] Spearman(u_video, T2V_rank) rho={rho_t2v:+.4f} p={p_t2v:.2e}")
    print(f"[diag] Spearman(u_video, V2T_rank) rho={rho_v2t:+.4f} p={p_v2t:.2e}")
    print(f"[diag] Spearman(u_video, max(T2V,V2T) rank) rho={rho_hard:+.4f} p={p_hard:.2e}")

    threshold = 0.20
    max_rho = max(rho_t2v, rho_v2t, rho_hard)
    verdict = "keep_uncertainty_claim" if max_rho >= threshold else "drop_uncertainty_claim"
    print(f"[diag] max_rho={max_rho:+.4f} threshold={threshold} => {verdict}")

    payload = {
        "checkpoint": str(args.checkpoint),
        "n_pairs": int(n_pairs),
        "retrieval": {
            "t2v": {"R@1": r1_t2v, "R@5": r5_t2v, "R@10": r10_t2v, "MdR": mdr_t2v},
            "v2t": {"R@1": r1_v2t, "R@5": r5_v2t, "R@10": r10_v2t, "MdR": mdr_v2t},
        },
        "u_video_stats": {
            "mean": float(u_video.mean()),
            "std": float(u_video.std()),
            "min": float(u_video.min()),
            "max": float(u_video.max()),
        },
        "spearman": {
            "u_vs_t2v_rank": {"rho": float(rho_t2v), "p": float(p_t2v)},
            "u_vs_v2t_rank": {"rho": float(rho_v2t), "p": float(p_v2t)},
            "u_vs_max_rank": {"rho": float(rho_hard), "p": float(p_hard)},
        },
        "decision": {"threshold": threshold, "max_rho": float(max_rho), "verdict": verdict},
    }
    out_path = args.output or str(Path(args.checkpoint).parent / "uncertainty_correlation.json")
    Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[diag] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
