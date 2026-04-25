import torch
import torch.nn as nn


class LearnedTemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()

        self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        B, N, D = x.shape
        # T: 时间步数
        # Q: 每帧查询数 (num_frame_queries)
        # 1. 计算出当前序列需要的位置编码长度
        current_T = N // self.pe.shape[1]  # 计算T

        # 2. 提取需要的 T 个位置编码
        pe_t = self.pe[:current_T, 0, :].unsqueeze(0)  # [1, T, D]

        # 3. 复制 num_frame_queries 次
        # 假设 Q_f = num_frame_queries
        Q_f = N // current_T
        # pe_t 形状 [1, T, D] -> [1, T*Q_f, D]
        # (这里假设Q_f恒定, 更好的方法是reshape)
        pe_t_expanded = pe_t.repeat_interleave(Q_f, dim=1)  # [1, T*Q_f, D]

        # 4. 广播到批次 B，并与输入相加
        # x [B, T*Q_f, D] + pe_t_expanded [1, T*Q_f, D]
        return x + pe_t_expanded


class QueryFormer(nn.Module):
    def __init__(self, d_model=512, num_queries=16, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # 1. 语义锚点 (Semantic Anchors)
        # 这些是学习出来的参数，目的是去"激活"视频中不同的语义部分
        self.query_tokens = nn.Parameter(torch.randn(num_queries, d_model))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # 2. 解码层 (Decoder Layer)
        # TransformerDecoderLayer 包含 Self-Attention (Refinement) 和 Cross-Attention (Extraction)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=0.1, activation="gelu", batch_first=True
        )

        # 增加层数以整合 Refinement 功能
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 3. 显式重要性打分 (Gating Mechanism)
        self.gate_fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )

    def forward(self, video_features, padding_mask=None,
                attr_features=None, attr_padding_mask=None,
                disable_gate=False):
        B = video_features.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)

        # Build multimodal memory: visual tokens (+ optional attribute tokens)
        if attr_features is not None:
            memory = torch.cat([video_features, attr_features], dim=1)
            if padding_mask is not None and attr_padding_mask is not None:
                memory_mask = torch.cat([padding_mask, attr_padding_mask], dim=1)
            elif padding_mask is not None:
                no_pad = torch.zeros(B, attr_features.size(1),
                                     device=video_features.device, dtype=torch.bool)
                memory_mask = torch.cat([padding_mask, no_pad], dim=1)
            elif attr_padding_mask is not None:
                no_pad = torch.zeros(B, video_features.size(1),
                                     device=video_features.device, dtype=torch.bool)
                memory_mask = torch.cat([no_pad, attr_padding_mask], dim=1)
            else:
                memory_mask = None
        else:
            memory = video_features
            memory_mask = padding_mask

        queries = self.decoder(tgt=queries, memory=memory,
                               memory_key_padding_mask=memory_mask)
        queries = self.norm(queries)

        gate_scores = self.gate_fc(queries).squeeze(-1)  # [B, Q]
        if disable_gate:
            gate_scores = torch.ones_like(gate_scores)

        return queries, gate_scores
