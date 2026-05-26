"""
TQFS（Temporal Quality-based Frame Sampler）实现。

移植自 UMIVR（ICCV 2025）Sec. 3.3，去除 VideoLLaVA 语义特征依赖，
改用 OpenCV Laplacian 方差 + 像素级 KMeans 聚类。

对外接口：select_tqfs_indices(raw_frames, num_frames) -> List[int]
"""

import numpy as np


def _compute_sharpness(frame_bgr: np.ndarray) -> float:
    """计算单帧清晰度（Laplacian 方差的 numpy 近似）。

    等效于 cv2.Laplacian(gray, cv2.CV_64F).var()，但用纯 numpy 实现，
    避免 OpenCV 4.11 的 cvtColor/Laplacian 兼容性问题。

    Args:
        frame_bgr: shape [H, W, 3] 的 uint8 数组（BGR 格式）。

    Returns:
        float: 清晰度分数，越大越清晰。
    """
    # cv2 VideoCapture 返回的是 cv2.Mat 子类，需要转为标准 numpy
    frame_bgr = np.asarray(frame_bgr)
    # BGR -> 灰度（与 cv2.COLOR_BGR2GRAY 一致：0.114*B + 0.587*G + 0.299*R）
    gray = (0.114 * frame_bgr[:, :, 0].astype(np.float64)
            + 0.587 * frame_bgr[:, :, 1].astype(np.float64)
            + 0.299 * frame_bgr[:, :, 2].astype(np.float64))
    # 二阶差分近似 Laplacian：d²I/dx² + d²I/dy²
    dxx = gray[:, 2:] - 2 * gray[:, 1:-1] + gray[:, :-2]  # [H, W-2]
    dyy = gray[2:, :] - 2 * gray[1:-1, :] + gray[:-2, :]  # [H-2, W]
    lap = dxx[1:-1, :] + dyy[:, 1:-1]  # [H-2, W-2]
    return float(lap.var())


def select_tqfs_indices(raw_frames, num_frames: int):
    """从已解码的原始帧序列中选出 num_frames 个高质量、语义多样的帧索引。

    流程与 UMIVR frame_selection 一致：
    1. 计算每帧 Laplacian 方差（清晰度）
    2. 时间分箱，每箱选质量最高帧作为候选（候选数 ≈ 3×目标帧数）
    3. 对候选帧做像素级 KMeans 聚类，保证时间多样性
    4. 每簇取质量最高帧作为最终选择

    Args:
        raw_frames: List[np.ndarray]，每个元素为 [H, W, 3] BGR uint8。
        num_frames: 目标帧数。

    Returns:
        List[int]: 按时间升序排列的帧索引列表，长度为 min(num_frames, N)。
    """
    N = len(raw_frames)
    if N <= num_frames:
        return list(range(N))

    # 步骤 1：计算每帧清晰度（Laplacian 方差）
    quality = np.array([_compute_sharpness(raw_frames[i]) for i in range(N)])

    # 步骤 2：时间分箱，每箱选质量最高帧作为候选
    # 候选数约为目标帧数的 3 倍，确保有足够多样性供聚类
    num_candidates = min(max(3 * num_frames, (N // 3)), N)
    bin_edges = np.linspace(0, N, num_candidates + 1, dtype=int)
    candidates = []
    for b in range(num_candidates):
        start, end = int(bin_edges[b]), int(bin_edges[b + 1])
        if start >= end:
            continue
        best = start + int(np.argmax(quality[start:end]))
        candidates.append(best)

    if len(candidates) <= num_frames:
        return sorted(candidates)

    # 步骤 3：对候选帧做像素级 KMeans，保证语义多样性
    try:
        from sklearn.cluster import KMeans  # noqa: PLC0415
        cand_arr = np.array(candidates)
        H, W = raw_frames[0].shape[0], raw_frames[0].shape[1]
        stride = max(1, H // 16)  # 下采样到约 16×16，降低计算量

        feats = []
        for idx in candidates:
            f = np.asarray(raw_frames[idx])  # [H, W, 3] uint8 BGR
            # BGR -> 灰度
            gray = (0.114 * f[:, :, 0].astype(np.float32)
                    + 0.587 * f[:, :, 1].astype(np.float32)
                    + 0.299 * f[:, :, 2].astype(np.float32))
            feats.append(gray[::stride, ::stride].flatten())
        feats = np.stack(feats).astype(np.float32)  # [num_candidates, D]

        km = KMeans(n_clusters=num_frames, random_state=42, n_init=5)
        labels = km.fit_predict(feats)

        # 每簇取质量最高帧
        selected = []
        for c in range(num_frames):
            mask = labels == c
            if not mask.any():
                continue
            cluster_cands = cand_arr[mask]
            selected.append(int(cluster_cands[np.argmax(quality[cluster_cands])]))

        selected = sorted(set(selected))

        # 数量不足时用均匀采样补足
        if len(selected) < num_frames:
            uniform = list(np.linspace(0, N - 1, num_frames, dtype=int))
            extras = [x for x in uniform if x not in set(selected)]
            selected = sorted(set(selected) | set(extras[: num_frames - len(selected)]))

        return selected[:num_frames]

    except Exception:
        # fallback：均匀采样，保证鲁棒性
        return list(np.linspace(0, N - 1, num_frames, dtype=int))
