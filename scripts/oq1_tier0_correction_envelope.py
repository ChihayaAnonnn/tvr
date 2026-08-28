"""OQ-1 Tier 0: capability envelope of the additive/affine per-candidate
correction family, measured on the frozen fp32 val similarity matrix.

Question (research/04-open-questions.md, OQ-1 Tier 0)
-----------------------------------------------------
How much retrieval gain -- and with what shape (A index, MnR, hubness
skewness) -- can query-independent per-candidate corrections produce on our
own frozen backbone?  Arms:

  label-free : DSL(T) / inverted softmax(T) / QB-Norm(beta) / Sinkhorn(T)
  oracle     : split-half MLE fits of additive b_j and affine (a_j, b_j)

If even the affine oracle cannot produce the DAB shape (R@1 up, R@5/R@10 up
more, MnR collapse), the thesis' diagnostic half is refuted at zero GPU cost.

Ground-truth rule (verified, not assumed)
-----------------------------------------
The val dataset enforces exactly 20 captions per video (dataloader raises
otherwise) and metric code reads the group diagonal, so caption i -> video
i // 20.  This script additionally re-derives the post-DSL T2V/V2T metrics
recorded in frozen_retrieval_equivalence.json from the raw matrix with the
repo's own functions and refuses to run the analysis if they do not match.

No GPU, no test-set access (val-only; PC-4 untouched).

Usage:  python scripts/oq1_tier0_correction_envelope.py
Output: artifacts/oq1_tier0/tier0_results.json  (+ stdout tables)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ART = REPO_ROOT / "artifacts/rescued_from_tmp/frozen_equiv"
OUT = REPO_ROOT / "artifacts/oq1_tier0"
EXPECTED_SHA = "8ebf27ae44468c67dd1b76c532f92bfdba65033faabcd5a758360cd0f48faf65"

N_CAP, N_VID, CPV = 10000, 500, 20
GT = np.arange(N_CAP) // CPV
CUT_OFF = list(range(CPV, N_CAP + 1, CPV))

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# Step 0: load + provenance check
# ---------------------------------------------------------------------------

def load_matrix() -> np.ndarray:
    sim = np.load(ART / "sim_june_fp32.npy")
    sha = hashlib.sha256(
        np.ascontiguousarray(sim, dtype=np.float64).tobytes()
    ).hexdigest()
    if sim.shape != (N_CAP, N_VID) or sha != EXPECTED_SHA:
        raise SystemExit(f"matrix provenance check failed: shape={sim.shape} sha={sha}")
    print(f"[tier0] matrix ok: shape={sim.shape} sha={sha[:12]} "
          f"range=[{sim.min():.3f},{sim.max():.3f}]")
    return sim.astype(np.float64)


# ---------------------------------------------------------------------------
# Step 1: alignment verification against the frozen-contract JSON
# ---------------------------------------------------------------------------

def verify_alignment(sim: np.ndarray) -> None:
    """Reproduce the post-DSL metrics stored in frozen_retrieval_equivalence.json
    using the repo's own eval functions.  Exact reproduction proves the assumed
    row/column ordering and the caption->video rule simultaneously."""
    from main_task_retrieval import _apply_dual_softmax, _reshape_multi_sentence_matrix
    from metrics import (compute_metrics, tensor_text_to_video_metrics,
                         tensor_video_to_text_sim)

    contract = json.loads((ART / "frozen_retrieval_equivalence.json").read_text())
    ref = contract["runs"]["june@fp32"]

    t2v_dsl, _ = _apply_dual_softmax(sim, contract["dsl_temperature"])
    m_t2v = tensor_text_to_video_metrics(
        _reshape_multi_sentence_matrix(t2v_dsl, CUT_OFF))
    # multi-sentence eval passes independent_directions=False, so V2T is
    # computed from the T2V DSL matrix (main_task_retrieval.py:4378-4384)
    m_v2t = compute_metrics(
        tensor_video_to_text_sim(
            torch.tensor(_reshape_multi_sentence_matrix(t2v_dsl, CUT_OFF))
        ).numpy())

    def close(a, b):
        return abs(float(a) - float(b)) < 1e-4

    checks = [
        ("t2v.R1", m_t2v["R1"], ref["t2v"]["R1"]),
        ("t2v.R5", m_t2v["R5"], ref["t2v"]["R5"]),
        ("t2v.R10", m_t2v["R10"], ref["t2v"]["R10"]),
        ("t2v.MR", m_t2v["MR"], ref["t2v"]["MR"]),
        ("v2t.R1", m_v2t["R1"], ref["v2t"]["R1"]),
        ("v2t.R5", m_v2t["R5"], ref["v2t"]["R5"]),
        ("v2t.R10", m_v2t["R10"], ref["v2t"]["R10"]),
    ]
    bad = [(n, a, b) for n, a, b in checks if not close(a, b)]
    for n, a, b in checks:
        print(f"[verify] {n}: recomputed={float(a):.10f} contract={float(b):.10f}"
              f" {'OK' if close(a, b) else 'MISMATCH'}")
    if bad:
        raise SystemExit(f"alignment verification FAILED: {bad}")

    # cross-check: flat ranks (gt = i // 20) must equal the repo's grouped path
    flat = flat_metrics(t2v_dsl)
    if not (close(flat["R1"], m_t2v["R1"]) and close(flat["MnR"], m_t2v["MeanR"])):
        raise SystemExit(f"flat-vs-grouped mismatch: {flat} vs {m_t2v}")
    print("[verify] alignment + flat-metric equivalence: PASS")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def flat_metrics(M: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """T2V metrics with gt video = row // 20.  rank = #{j: s_ij > s_i,gt}."""
    if rows is None:
        rows = np.arange(M.shape[0])
    sub = M[rows]
    gt = GT[rows]
    gt_scores = sub[np.arange(len(rows)), gt]
    ranks = (sub > gt_scores[:, None]).sum(axis=1)
    return {
        "R1": float((ranks == 0).mean() * 100),
        "R5": float((ranks < 5).mean() * 100),
        "R10": float((ranks < 10).mean() * 100),
        "MdR": float(np.median(ranks) + 1),
        "MnR": float(ranks.mean() + 1),
    }


def hubness(M: np.ndarray, k: int, rows: np.ndarray | None = None) -> dict:
    """k-occurrence distribution over videos and its Fisher moment skewness."""
    if rows is None:
        rows = np.arange(M.shape[0])
    sub = M[rows]
    topk = np.argpartition(-sub, k, axis=1)[:, :k]
    nk = np.bincount(topk.ravel(), minlength=N_VID).astype(np.float64)
    mu, sd = nk.mean(), nk.std()
    return {
        f"S_N{k}": float(((nk - mu) ** 3).mean() / sd**3) if sd > 0 else 0.0,
        f"maxN{k}": int(nk.max()),
        f"zeroN{k}": int((nk == 0).sum()),
        f"_N{k}": nk,
    }


def arm_report(name: str, M: np.ndarray, rows: np.ndarray | None = None) -> dict:
    rep = {"arm": name}
    rep.update(flat_metrics(M, rows))
    for k in (5, 10):
        h = hubness(M, k, rows)
        rep.update({kk: v for kk, v in h.items() if not kk.startswith("_")})
    return rep


# ---------------------------------------------------------------------------
# Label-free arms
# ---------------------------------------------------------------------------

def arm_dsl(S: np.ndarray, T: float) -> np.ndarray:
    from main_task_retrieval import _apply_dual_softmax
    return _apply_dual_softmax(S, T)[0].astype(np.float64)


def arm_inverted_softmax(S: np.ndarray, T: float) -> np.ndarray:
    """s'_ij = s_ij - T * logsumexp_i(s_ij / T): pure additive b_j."""
    x = S / T
    lse = np.logaddexp.reduce(x, axis=0)          # over queries, per video
    return S - T * lse[None, :]


def arm_qbnorm(S: np.ndarray, beta: float) -> np.ndarray:
    """QB-Norm / dynamic inverted softmax (Bogolin et al., CVPR'22), with the
    val query set itself as querybank (label-free, transductive -- same
    information DSL uses).  Activated set = videos that are top-1 for >=1 bank
    query; only queries whose raw top-1 is activated get renormalized."""
    top1 = S.argmax(axis=1)
    activated = np.zeros(N_VID, dtype=bool)
    activated[np.unique(top1)] = True
    x = beta * S
    lse = np.logaddexp.reduce(x, axis=0)
    normalized = x - lse[None, :]
    out = S.copy()
    hit = activated[top1]
    out[hit] = normalized[hit]
    return out


def arm_sinkhorn(S: np.ndarray, T: float, iters: int = 50) -> np.ndarray:
    """Sinkhorn on exp(S/T), uniform marginals.  Row scaling never changes a
    row's ranking, so for T2V this is exactly an additive b_j = T*log(v_j) --
    membership in the additive family is by construction."""
    logK = (S - S.max()) / T
    log_u = np.zeros(N_CAP)
    log_v = np.zeros(N_VID)
    log_r = np.log(1.0 / N_CAP)
    log_c = np.log(1.0 / N_VID)
    for _ in range(iters):
        log_u = log_r - np.logaddexp.reduce(logK + log_v[None, :], axis=1)
        log_v = log_c - np.logaddexp.reduce(logK + log_u[:, None], axis=0)
    return logK + log_u[:, None] + log_v[None, :]


# ---------------------------------------------------------------------------
# Oracle arms (split-half conditional-logit MLE; convex in (a, b))
# ---------------------------------------------------------------------------

FIT_ROWS = np.arange(N_CAP)[np.arange(N_CAP) % 2 == 0]
EVAL_ROWS = np.arange(N_CAP)[np.arange(N_CAP) % 2 == 1]


def fit_oracle(S: np.ndarray, tau: float, affine: bool,
               steps: int = 4000, lr: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    s = torch.tensor(S[FIT_ROWS] / tau, dtype=torch.float64)
    y = torch.tensor(GT[FIT_ROWS], dtype=torch.long)
    b = torch.zeros(N_VID, dtype=torch.float64, requires_grad=True)
    params = [b]
    if affine:
        a = torch.ones(N_VID, dtype=torch.float64, requires_grad=True)
        params.append(a)
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(steps):
        opt.zero_grad()
        logits = (a * s + b) if affine else (s + b)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    a_np = a.detach().numpy() if affine else np.ones(N_VID)
    return a_np, b.detach().numpy()


def oracle_matrix(S: np.ndarray, a: np.ndarray, b: np.ndarray, tau: float) -> np.ndarray:
    return a[None, :] * S + tau * b[None, :]


def run_oracle(S: np.ndarray, affine: bool, taus=(0.5, 1.0, 2.0, 5.0)) -> dict:
    """tau selected on the FIT half by R@1; reported metrics are EVAL-half only."""
    best = None
    for tau in taus:
        a, b = fit_oracle(S, tau, affine)
        M = oracle_matrix(S, a, b, tau)
        fit_r1 = flat_metrics(M, FIT_ROWS)["R1"]
        print(f"[oracle {'affine' if affine else 'additive'}] tau={tau} "
              f"fit-half R@1={fit_r1:.2f}")
        if best is None or fit_r1 > best["fit_R1"]:
            best = {"tau": tau, "a": a, "b": b, "fit_R1": fit_r1, "M": M}
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = load_matrix()
    verify_alignment(S)

    results = {"full": [], "eval_half": [], "grids": [], "fingerprints": {}}

    # --- baseline + label-free arms, default hyperparameters ---
    arms_default = {
        "raw": S,
        "dsl_T2": arm_dsl(S, 2.0),
        "isoftmax_T2": arm_inverted_softmax(S, 2.0),
        "qbnorm_b0.5": arm_qbnorm(S, 0.5),
        "sinkhorn_T2": arm_sinkhorn(S, 2.0),
    }
    for name, M in arms_default.items():
        results["full"].append(arm_report(name, M))
        results["eval_half"].append(arm_report(name, M, EVAL_ROWS))

    # --- label-free temperature grids (tuned-on-val => flagged, supplementary) ---
    for T in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        results["grids"].append(arm_report(f"dsl_T{T}", arm_dsl(S, T)))
        results["grids"].append(arm_report(f"isoftmax_T{T}", arm_inverted_softmax(S, T)))
        results["grids"].append(arm_report(f"sinkhorn_T{T}", arm_sinkhorn(S, T)))
        results["grids"].append(arm_report(f"qbnorm_b{1.0 / T}", arm_qbnorm(S, 1.0 / T)))

    # --- oracle arms (fit on even rows, evaluated on odd rows only) ---
    for affine in (False, True):
        tag = "oracle_affine" if affine else "oracle_additive"
        best = run_oracle(S, affine)
        rep = arm_report(f"{tag}_tau{best['tau']}", best["M"], EVAL_ROWS)
        rep["fit_half_R1"] = best["fit_R1"]
        results["eval_half"].append(rep)
        results["fingerprints"][tag] = {
            "tau": best["tau"],
            "b_mean": float(best["b"].mean()),
            "b_std": float(best["b"].std()),
            "a_mean": float(best["a"].mean()),
            "a_std": float(best["a"].std()),
            "b": best["b"].tolist(),
            "a": best["a"].tolist(),
        }

    # --- mechanism fingerprints: corr(b_j, N_k^baseline(j)) ---
    from scipy.stats import pearsonr, spearmanr
    nk_base = {k: hubness(S, k)[f"_N{k}"] for k in (5, 10)}
    is_b = -2.0 * np.logaddexp.reduce(S / 2.0, axis=0)   # isoftmax_T2 additive term
    fingerprint_bs = {
        "isoftmax_T2_b": is_b,
        "oracle_additive_b": np.array(results["fingerprints"]["oracle_additive"]["b"]),
        "oracle_affine_b": np.array(results["fingerprints"]["oracle_affine"]["b"]),
    }
    for bname, bvec in fingerprint_bs.items():
        for k in (5, 10):
            pr = pearsonr(bvec, nk_base[k])
            sr = spearmanr(bvec, nk_base[k])
            results["fingerprints"].setdefault("corr_b_vs_Nk", {})[
                f"{bname}_vs_N{k}"
            ] = {"pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
                 "spearman_rho": float(sr[0]), "spearman_p": float(sr[1])}

    # --- N_k of baseline for the record ---
    for k in (5, 10):
        nk = nk_base[k]
        results["fingerprints"][f"baseline_N{k}"] = {
            "skew": float(hubness(S, k)[f"S_N{k}"]),
            "max": int(nk.max()), "zeros": int((nk == 0).sum()),
            "top10_videos": np.argsort(-nk)[:10].tolist(),
            "top10_counts": np.sort(nk)[::-1][:10].tolist(),
        }

    # --- pretty print ---
    def show(rows, title):
        print(f"\n=== {title} ===")
        hdr = ["arm", "R1", "R5", "R10", "MdR", "MnR", "S_N5", "S_N10", "maxN10"]
        print(" | ".join(f"{h:>14}" for h in hdr))
        for r in rows:
            print(" | ".join(
                f"{r[h]:>14.3f}" if isinstance(r.get(h), float) else f"{str(r.get(h, '')):>14}"
                for h in hdr))

    show(results["full"], "FULL SET (10000 queries) -- label-free arms")
    show(results["eval_half"], "EVAL HALF (odd rows, 5000 queries) -- all arms incl. oracle")
    show(results["grids"], "GRIDS (tuned on val -- supplementary only)")
    print("\n=== fingerprints: corr(b_j, N_k^baseline) ===")
    for name, d in results["fingerprints"].get("corr_b_vs_Nk", {}).items():
        print(f"  {name}: pearson={d['pearson_r']:+.3f} (p={d['pearson_p']:.2e}) "
              f"spearman={d['spearman_rho']:+.3f}")

    out_path = OUT / "tier0_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[tier0] wrote {out_path}")


if __name__ == "__main__":
    main()
