"""
MUS（Mapping Uncertainty Score）批量计算工具。

移植自 UMIVR（ICCV 2025）Sec. 3.2，公式 (2)–(5)。
对每个文本查询，衡量其与候选视频相似度分布是否"尖锐"：
  - MUS ≈ 0：存在高置信度的单一最优视频，映射清晰
  - MUS ≈ 1：分数分布平坦，映射不确定性高

对外接口：compute_mus_batch(sim_row, k=10) -> np.ndarray [N_text]
"""

import numpy as np


def _custom_normalize(scores: np.ndarray, cutoff: float = None, p: float = 2.0) -> np.ndarray:
    """论文式 (2)：对 top-k 相似度做截断幂次归一化，得到分布 p。"""
    if cutoff is None:
        cutoff = float(np.mean(scores))
    s_max = float(scores[0])
    if abs(s_max - cutoff) < 1e-8:
        return np.ones_like(scores) / len(scores)
    new_scores = np.array([
        ((s - cutoff) / (s_max - cutoff)) ** p if s >= cutoff else 0.0
        for s in scores
    ])
    total = new_scores.sum()
    if total == 0:
        return np.ones_like(new_scores) / len(new_scores)
    return new_scores / total


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))


def _max_jsd(k: int) -> float:
    """理论上界：one-hot 与均匀分布之间的 JSD。"""
    q = np.zeros(k)
    q[0] = 1.0
    p = np.ones(k) / k
    m = 0.5 * (q + p)
    return 0.5 * (_kl_divergence(q, m) + _kl_divergence(p, m))


def _mus_single(row: np.ndarray, k: int = 10) -> float:
    """对单个查询行向量计算 MUS。

    Args:
        row: shape [N_video] 的相似度向量（numpy float）。
        k:   取 top-k 参与计算，默认 10。

    Returns:
        float: MUS ∈ [0, 1]。
    """
    k = min(k, len(row))
    top_k_idx = np.argsort(row)[::-1][:k]
    top_k_scores = row[top_k_idx]

    p_dist = _custom_normalize(top_k_scores)
    q = np.zeros(k)
    q[int(np.argmax(top_k_scores))] = 1.0

    jsd = _js_divergence(p_dist, q)
    jsd_max = _max_jsd(k)
    mus = jsd / jsd_max if jsd_max > 0 else 0.0
    return float(min(mus, 1.0))


def compute_mus_batch(sim_matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """对整个相似度矩阵批量计算每条查询的 MUS。

    Args:
        sim_matrix: shape [N_text, N_video] 的 numpy float 数组。
        k:          每个查询取 top-k 候选参与 JSD 计算。

    Returns:
        np.ndarray: shape [N_text]，每行对应一个查询的 MUS 值。
    """
    return np.array([_mus_single(sim_matrix[i], k=k) for i in range(sim_matrix.shape[0])])
