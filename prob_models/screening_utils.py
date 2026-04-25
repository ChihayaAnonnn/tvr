import torch
import torch.nn as nn
import torch.nn.functional as F

def coarse_screening(q_uncert_dict, text_mean, pos_query_mask, neg_query_mask, logit_scale, config_topk_t, config_topk_v):
    """
    重构和双向优化的粗筛。
    - 1. 对视频查询进行不确定性建模。
    - 2. 使用均值（mu）进行 V2T 和 T2V 粗筛。
    - 3. 清理返回值。
    """
    
    # 1. 不确定性建模
    mean_queries = q_uncert_dict['mean']          # [B, Q, D]

    # 2. 归一化
    mean_queries_norm = F.normalize(mean_queries, dim=-1)
    text_norm = F.normalize(text_mean, dim=-1)     # [B, D]

    # 3. 构建 Query 掩码 (与原版相同)
    selected_query_mask = pos_query_mask | neg_query_mask  # [B, Q]
    selected_query_counts = selected_query_mask.sum(dim=1).clamp(min=1).float() # [B]

    # 4. 计算 V2T 粗筛 Logits [B, B]
    #    (视频 V 聚合 Q 个查询后，与所有文本 T 比较)
    
    # 4a. 计算所有查询与文本的相似度矩阵
    # [B_v, Q, D] x [B_t, D] -> [B_v, Q, B_t]
    coarse_sim_matrix_v2t = torch.einsum('bqd,td->bqt', mean_queries_norm, text_norm)
    
    # 4b. 仅使用选中的查询
    query_mask = selected_query_mask.unsqueeze(2)  # [B_v, Q, 1]

    coarse_sim_masked = coarse_sim_matrix_v2t * query_mask.float()
    
    # 4c. 聚合查询维度，得到视频-文本相似度
    # [B_v, Q, B_t] -> [B_v, B_t]
    coarse_scores_v2t = coarse_sim_masked.sum(dim=1) / selected_query_counts.unsqueeze(1)
    coarse_logits_v2t = logit_scale * coarse_scores_v2t

    # 5. 计算 T2V 粗筛 Logits [B_t, B_v]
    #    (文本 T 与所有视频 V 聚合后的 Q 个查询比较)
    #    在我们的例子中，这只是 V2T 矩阵的转置
    coarse_logits_t2v = coarse_logits_v2t.T  # [B_t, B_v]

    # 6. V2T Top-K: 为每个视频选择 Top-K 文本
    topk_t = min(config_topk_t, coarse_logits_v2t.size(1))
    v2t_topk_indices = get_topk_with_positive(coarse_logits_v2t, k=topk_t, dim=1)

    # 7. T2V Top-K: 为每个文本选择 Top-K 视频
    topk_v = min(config_topk_v, coarse_logits_t2v.size(1))
    t2v_topk_indices = get_topk_with_positive(coarse_logits_t2v, k=topk_v, dim=1)

    # 8. 返回清理后的必要信息
    return {
        'selected_query_mask': selected_query_mask, # 选中的查询掩码
        'coarse_logits_v2t': coarse_logits_v2t,    # 用于V2T粗筛损失
        'coarse_logits_t2v': coarse_logits_t2v,    # 用于T2V粗筛损失
        'v2t_topk_indices': v2t_topk_indices,  # [B_v, K_t] Top-K文本索引
        't2v_topk_indices': t2v_topk_indices   # [B_t, K_v] Top-K视频索引
    }

def get_topk_with_positive(logits, k, dim):
    """辅助函数：执行topk并确保对角线（正样本）被包含在内。"""
    B = logits.size(0)
    device = logits.device
    
    topk_scores, topk_indices = torch.topk(logits, k=k, dim=dim)
    
    pos_indices = torch.arange(B, device=device)
    pos_scores = logits[pos_indices, pos_indices]
    
    # 检查是否包含正样本
    contains_pos = (topk_indices == pos_indices.unsqueeze(1)).any(dim=1)
    
    if (~contains_pos).any():
        # 找到缺失正样本的行
        missing_idx = (~contains_pos).nonzero(as_tuple=False).squeeze(-1)
        # 将最后一列替换为正样本
        topk_indices[missing_idx, -1] = pos_indices[missing_idx]
        # (也可以选择更新分数，但对于索引来说这已足够)
        
    return topk_indices

# 粗筛对比损失
def calculate_coarse_loss(coarse_logits_v2t, coarse_logits_t2v):
    """计算 BxB 上的标准对称 InfoNCE 损失。"""
    device = coarse_logits_v2t.device
    B = coarse_logits_v2t.size(0)
    
    # 标签是 [0, 1, 2, ..., B-1]
    labels = torch.arange(B, device=device)
    
    loss_v2t = F.cross_entropy(coarse_logits_v2t, labels)
    loss_t2v = F.cross_entropy(coarse_logits_t2v, labels)
    
    return (loss_v2t + loss_t2v) / 2