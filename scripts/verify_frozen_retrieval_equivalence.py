"""Verify that sidecar-only training leaves the retrieval score path unchanged.

Motivation
----------
`PAPER.md` claims retrieval metrics are "byte-identical to the June baseline"
because A1/A3 train with the retriever frozen. The historical numbers cited in
support of that claim disagree (T2V R@1 = 52.6 / 52.1 / 52.9 for June / A1 / A3),
which reads as a refutation of the claim.

Static analysis says otherwise: the eval-time retrieval score is
`weighted_token_wise_intersection(text_token, visual_cls) * logit_scale`, whose
parameters (`clip.*`, `spatial_enhancer.*`, `text_weight_fc`, `video_weight_fc`,
`logit_scale`) are bitwise identical across the three checkpoints. The sidecar
heads that A1/A3 add are only read inside `if self.training:` branches.

The historical spread is therefore attributed to evaluator numerics, not to the
freeze leaking. `main_task_retrieval._eval_autocast_context` selects exact FP32 +
deterministic cuDNN when a *contextual* risk sidecar flag is present and bf16
autocast otherwise -- so A3 was evaluated in FP32 while June and A1 were
evaluated in bf16, on top of code-revision drift.

This script tests that attribution directly. It builds every checkpoint with one
canonical architecture config, runs one dataloader, forces one numeric path, and
compares the *raw pre-DSL* similarity matrices bitwise. Under a given precision,
identical matrices prove rank preservation far more strongly than identical R@1
(which could coincide by luck).

Scope
-----
Validation split only. The campaign seals the JSFUSION test set until Stages 4
and 5 pass, and rank preservation is a property of the model that holds on any
split, so no test access is spent here.

Usage
-----
    python scripts/verify_frozen_retrieval_equivalence.py \
        --config-from ckpts/ckpt_msrvtt_20260811_231959_risk_sidecar_a1/training_args.json \
        --checkpoint june=ckpts/.../pytorch_model.bin.4 \
        --checkpoint a1=ckpts/.../pytorch_model.bin.4 \
        --checkpoint a3=ckpts/.../pytorch_model.bin.4 \
        --precision fp32 --precision bf16 \
        --out-dir /tmp/frozen_equiv --device cuda:1
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Numeric-path control
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _forced_precision(precision: str):
    """Pin the evaluation numerics instead of letting sidecar flags pick them.

    `fp32` mirrors the A3 path: no autocast, TF32 off, deterministic cuDNN.
    `bf16` mirrors the June/A1 path: bf16 autocast with stock backend flags.
    """
    prior = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    if precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(precision == "bf16")
        ):
            yield
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        ) = prior


# ---------------------------------------------------------------------------
# Config / model construction
# ---------------------------------------------------------------------------

def _canonical_namespace(config_from: Path, overrides: dict) -> SimpleNamespace:
    """One architecture config for every checkpoint, so weights are the only variable."""
    from scripts.evaluate_crf_ladder import _restore_task_config

    cli_shim = SimpleNamespace(max_frames=None, feature_framerate=None, image_resolution=224)
    restored = _restore_task_config(config_from, cli_shim)
    merged = dict(vars(restored))
    merged.update(overrides)
    return SimpleNamespace(**merged)


def _build_model(checkpoint: Path, task_config: SimpleNamespace, device):
    """Construct UATVR from the canonical config and load one checkpoint into it.

    Checkpoints that predate the sidecars are missing those keys; the modules stay
    at their constructor init. That is exactly the condition under test -- if the
    missing heads mattered to retrieval, the similarity matrices would diverge.
    """
    if not getattr(task_config, "use_uatvr_full", False):
        raise ValueError("canonical config must set use_uatvr_full for the June architecture")
    from modules.modeling_mulit import UATVR

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cache_dir = getattr(task_config, "cache_dir", "") or os.path.join(
        os.environ.get("PYTORCH_PRETRAINED_BERT_CACHE", "/tmp"), "distributed"
    )
    cfg = SimpleNamespace(**dict(vars(task_config)))
    cfg.init_model = str(checkpoint)
    model = UATVR.from_pretrained(
        cfg.cross_model, cache_dir=cache_dir, state_dict=state_dict, task_config=cfg
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, ns, loader, device, precision):
    """Run one eval pass, returning metrics plus the raw pre-DSL similarity matrix.

    DSL is applied downstream of `_run_on_single_gpu`, so capturing its return
    value yields the raw WTI scores -- the actual model output, before any
    post-processing that could mask a difference.
    """
    import main_task_retrieval as mtr

    real_run = mtr._run_on_single_gpu
    real_ctx = mtr._eval_autocast_context
    captured = {"rows": None}

    def capturing_run(*a, **kw):
        rows = real_run(*a, **kw)
        captured["rows"] = rows
        return rows

    mtr._run_on_single_gpu = capturing_run
    mtr._eval_autocast_context = lambda args: _forced_precision(precision)
    try:
        t0 = time.time()
        metrics = mtr._eval_epoch_impl(ns, model, loader, device, n_gpu=1)
        elapsed = time.time() - t0
    finally:
        mtr._run_on_single_gpu = real_run
        mtr._eval_autocast_context = real_ctx

    if captured["rows"] is None:
        raise RuntimeError("similarity capture failed: _run_on_single_gpu was never called")
    sim = np.concatenate(captured["rows"], axis=0).astype(np.float64)
    return metrics, sim, elapsed


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def _summarize(metrics, direction):
    block = metrics.get(direction, {})
    return {k: float(block[k]) for k in ("R1", "R5", "R10", "MR") if k in block}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-from", required=True, type=Path,
                   help="training_args.json supplying the single canonical architecture config")
    p.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH",
                   help="repeatable; label is used in the report")
    p.add_argument("--precision", action="append", choices=["fp32", "bf16"],
                   help="repeatable; defaults to both")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dsl-temperature", type=float, default=2.0)
    p.add_argument("--repeat-first", action="store_true",
                   help="re-run the first checkpoint at the end as a determinism control")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    precisions = args.precision or ["fp32", "bf16"]
    ckpts = []
    for spec in args.checkpoint:
        if "=" not in spec:
            raise SystemExit(f"--checkpoint expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        resolved = Path(path)
        if not resolved.is_file():
            raise SystemExit(f"checkpoint not found: {resolved}")
        ckpts.append((label, resolved))

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    import main_task_retrieval as mtr
    mtr.logger = logging.getLogger("verify")

    from dataloaders.data_dataloaders import dataloader_msrvtt_val
    from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ns = _canonical_namespace(args.config_from, {
        "batch_size_val": args.batch_size,
        "num_thread_reader": args.num_workers,
        "prefetch_factor": 2,
        "pin_memory": False,
        "use_attributes": False,
        "local_rank": 0,
        "n_gpu": 1,
        "world_size": 1,
        "do_train": False,
        "do_eval": True,
        "eval_split": "val",
        "eval_vid_chunk_size": 128,
        "log_mus_scores": False,
        "DSL": True,
        "dsl_temperature": args.dsl_temperature,
    })

    tokenizer = ClipTokenizer()
    loader, n_pairs = dataloader_msrvtt_val(ns, tokenizer, subset="val")
    print(f"[verify] val pairs={n_pairs} batches={len(loader)} device={device}")
    print(f"[verify] canonical config from {args.config_from}")

    plan = [(label, path, prec) for prec in precisions for label, path in ckpts]
    if args.repeat_first:
        first_label, first_path = ckpts[0]
        plan += [(f"{first_label}__repeat", first_path, prec) for prec in precisions]

    runs = {}
    for label, path, prec in plan:
        print(f"\n[verify] === {label} @ {prec} ===")
        model = _build_model(path, ns, device)
        metrics, sim, elapsed = _evaluate(model, ns, loader, device, prec)
        del model
        torch.cuda.empty_cache()

        key = f"{label}@{prec}"
        sim_path = args.out_dir / f"sim_{key.replace('@', '_')}.npy"
        np.save(sim_path, sim)
        runs[key] = {
            "label": label,
            "precision": prec,
            "checkpoint": str(path),
            "sim_shape": list(sim.shape),
            "sim_sha256": _digest(sim),
            "sim_path": str(sim_path),
            "elapsed_sec": round(elapsed, 1),
            "t2v": _summarize(metrics, "t2v"),
            "v2t": _summarize(metrics, "v2t"),
        }
        r = runs[key]
        print(f"[verify] {key}: T2V R@1={r['t2v'].get('R1')} V2T R@1={r['v2t'].get('R1')} "
              f"sha={r['sim_sha256'][:16]} ({r['elapsed_sec']}s)")

    # Pairwise comparison within each precision -- the actual claim under test.
    comparisons = []
    for prec in precisions:
        keys = [k for k in runs if runs[k]["precision"] == prec]
        base = keys[0]
        base_sim = np.load(runs[base]["sim_path"])
        for other in keys[1:]:
            other_sim = np.load(runs[other]["sim_path"])
            same_shape = base_sim.shape == other_sim.shape
            identical = bool(same_shape and np.array_equal(base_sim, other_sim))
            max_abs = float(np.abs(base_sim - other_sim).max()) if same_shape else float("nan")
            comparisons.append({
                "precision": prec,
                "a": base,
                "b": other,
                "bitwise_identical": identical,
                "max_abs_diff": max_abs,
            })
            verdict = "IDENTICAL" if identical else f"DIFFERS (max |Δ|={max_abs:.3e})"
            print(f"[verify] {prec}: {base} vs {other} -> {verdict}")

    payload = {
        "purpose": "prove sidecar-only training preserves the retrieval score path",
        "split": "msrvtt trusted-v1 val",
        "n_pairs": int(n_pairs),
        "canonical_config": str(args.config_from),
        "dsl_temperature": args.dsl_temperature,
        "runs": runs,
        "comparisons": comparisons,
    }
    out_json = args.out_dir / "frozen_retrieval_equivalence.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\n[verify] wrote {out_json}")

    failed = [c for c in comparisons if not c["bitwise_identical"]]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
