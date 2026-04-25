import torch
import torch.nn as nn
import torch.nn.functional as F


class QuerySelectionHead(nn.Module):
    def __init__(
        self,
        pos_query_k: int = 4,  # 正样本选 K 个
        lambda_diversity: float = 0.5,  # MMR 多样性权重
    ):
        super().__init__()
        self.pos_query_k = pos_query_k
        self.lambda_diversity = lambda_diversity
        # Used as a very negative value to mask already-selected queries in greedy MMR.
        self.fill_value = -1e9

    def forward(self, query_text_sim, refined_queries):
        """
        query_text_sim: [B, Q, B]  (Batch Videos, Queries, Batch Texts)
                       or [B, Q, 1] (Pair-only selection: one text per video)
        refined_queries: [B, Q, D] (用于计算查询间冗余)
        """
        B, Q, B_text = query_text_sim.shape
        device = refined_queries.device

        # 准备查询间的相似度矩阵 (Redundancy)
        # [B, Q, Q]
        refined_queries_norm = F.normalize(refined_queries, dim=-1)
        sim_qq = torch.matmul(
            refined_queries_norm, refined_queries_norm.transpose(1, 2)
        )
        eye_mask = (
            torch.eye(Q, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        )
        sim_qq.masked_fill_(eye_mask, -1.0)  # 自身不计入冗余

        # =======================================================
        # Phase 1: 正样本构建 (Ground-Truth Guided MMR)
        # 目标：为文本 T_i 从 Video_i 中选出互补的查询组合
        # =======================================================

        # 1. 提取 Relevance
        if B == B_text:
            # 训练或对齐评估时使用对角线
            pos_relevance = torch.diagonal(query_text_sim, dim1=0, dim2=2).transpose(0, 1)
        else:
            # 评估时如果 Batch 不对齐（如 MSVD），取每个查询对当前文本 Batch 的最大响应作为代表
            pos_relevance = query_text_sim.max(dim=2).values

        # 2. 运行 MMR 选择 (保证多角度覆盖)
        pos_mask = self._run_mmr_selection(pos_relevance, sim_qq, self.pos_query_k)

        return {
            "pos_query_mask": pos_mask,  # 用于 WTI (正向拼图)
        }

    def _run_mmr_selection(self, relevance, sim_qq, k):
        """
        标准的 MMR 逻辑，用于选出互补的正样本集合
        """
        B, Q = relevance.shape
        device = relevance.device

        selected_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
        redundancy_scores = torch.zeros(B, Q, device=device)

        # NOTE:
        # - MMR is inherently greedy (each selection depends on previous picks),
        #   so we keep a small loop over k.
        # - We vectorize the redundancy update via batched matmul instead of gather/expand,
        #   which reduces Python overhead and uses efficient GPU kernels.
        k = int(min(k, Q))
        fill_value = torch.tensor(self.fill_value, device=device, dtype=relevance.dtype)

        for _ in range(k):
            # MMR Score = Relevance - lambda * Redundancy
            mmr_scores = relevance - (self.lambda_diversity * redundancy_scores)
            mmr_scores.masked_fill_(selected_mask, fill_value)

            best_idx = torch.argmax(mmr_scores, dim=1)  # [B]
            selected_mask.scatter_(1, best_idx.unsqueeze(1), True)

            # 更新冗余度：取 max
            # sim_to_new[b, q] = sim_qq[b, q, best_idx[b]]
            one_hot = F.one_hot(best_idx, num_classes=Q).to(dtype=sim_qq.dtype)  # [B, Q]
            sim_to_new = torch.bmm(sim_qq, one_hot.unsqueeze(-1)).squeeze(-1)  # [B, Q]
            redundancy_scores = torch.maximum(redundancy_scores, sim_to_new)

        return selected_mask
