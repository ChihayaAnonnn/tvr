from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange


def rotate_every_two(x):
    """
    RoPE核心辅助函数：将特征向量按每两个元素分组，对每组执行旋转操作
    功能：实现复数平面的虚部取反（对应旋转矩阵的- sinθ项），为后续theta_shift做准备
    输入：x [B, N, L, D]（B=批次，N=头数，L=序列长度/像素数，D=单头特征维度）
    输出：旋转后的特征 [B, N, L, D]（每两个元素一组，第二元素取反并交换位置）
    数学对应：对于特征(x1, x2)，输出(-x2, x1)，模拟复数x1+ix2的90°旋转
    """
    # 按最后一维（特征维度）步长2拆分：x1取偶数索引，x2取奇数索引
    x1 = x[:, :, :, ::2]  # [B, N, L, D//2]（偶数位特征）
    x2 = x[:, :, :, 1::2]  # [B, N, L, D//2]（奇数位特征）
    # 堆叠(-x2, x1)，并沿最后一维展平（恢复原特征维度D）
    x = torch.stack([-x2, x1], dim=-1)  # [B, N, L, D//2, 2]
    return x.flatten(-2)  # [B, N, L, D]（展平后维度与输入一致）


def theta_shift(x, sin, cos):
    """
    RoPE核心计算函数：将位置编码（sin/cos）融入特征向量，实现旋转位置嵌入
    功能：通过特征向量与sin/cos的线性组合，编码相对位置信息（无需额外参数）
    输入：
        x: 待编码特征 [B, N, L, D]
        sin: 位置对应的正弦值 [L, D]（L=序列长度/像素数，D=单头特征维度）
        cos: 位置对应的余弦值 [L, D]
    输出：旋转编码后的特征 [B, N, L, D]
    数学原理：x_rot = x * cosθ + rotate_every_two(x) * sinθ，对应复数乘法的实部计算
    """
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE(nn.Module):
    """
    旋转位置编码（Rotary Position Embedding）模块
    功能：生成2D空间（图像H×W）中每个像素的位置编码参数（sin/cos），支持任意分辨率外推
    输入：slen (H, W)（图像高度、宽度，对应2D位置坐标）
    输出：(sin, cos) 元组，均为 [H×W, D]（D=单头特征维度）
    核心优势：1. 无额外可学习参数；2. 支持任意长度/分辨率序列外推；3. 编码相对位置信息
    """

    def __init__(self, embed_dim, num_heads):
        '''
        参数说明：
            embed_dim: 总特征维度（如64）
            num_heads: 注意力头数（如4），单头特征维度D = embed_dim // num_heads
        '''
        super().__init__()
        # 计算旋转角度基数：遵循10000^(-2(k-1)/d)规律，生成D//2个基数（因每2维一组旋转）
        # embed_dim//num_heads为单头维度D，//4是因代码中sin/cos分h/w两部分各占D//2
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        # 扩展为D维：每个基数重复2次（对应sin/cos的h/w分量），最终维度为D
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # 注册为缓冲区（不参与梯度更新），避免每次前向重新计算
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int]):
        '''
        参数说明：slen (H, W)，图像的高度和宽度（2D位置坐标）
        '''
        # 生成H×W像素的位置索引（h方向：0~H-1，w方向：0~W-1）
        index_h = torch.arange(slen[0]).to(self.angle)  # [H]（h方向位置索引）
        index_w = torch.arange(slen[1]).to(self.angle)  # [W]（w方向位置索引）

        # 计算h方向的sin值：[H] × [D//2] → [H, D//2]，再广播到[H, W, D//2]
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # [H, D//2]
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # [H, W, D//2]
        # 计算w方向的sin值：[W] × [D//2] → [W, D//2]，再广播到[H, W, D//2]
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # [W, D//2]
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # [H, W, D//2]
        # 合并h/w的sin值，得到完整的2D位置sin编码：[H, W, D]
        sin = torch.cat([sin_h, sin_w], -1)

        # 同理计算cos值（步骤与sin完全一致）
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # [H, D//2]
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # [H, W, D//2]
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # [W, D//2]
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # [H, W, D//2]
        cos = torch.cat([cos_h, cos_w], -1)

        # 展平为[H×W, D]（将2D像素位置转为1D序列，适配注意力输入格式）
        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))
        return retention_rel_pos


class GateLinearAttentionNoSilu(nn.Module):
    """
    门控线性注意力模块（无SiLU激活）：RALA的核心计算单元
    功能：结合RoPE位置编码，以线性复杂度实现注意力计算，支持2D图像特征输入
    输入：
        x: 图像特征 [B, C, H, W]（C=总特征维度，与embed_dim一致）
        sin: RoPE生成的正弦编码 [H×W, D]
        cos: RoPE生成的余弦编码 [H×W, D]
    输出：注意力加权后的特征 [B, C, H, W]（与输入维度一致）
    核心创新：1. 门控机制（ELU+1）抑制噪声；2. 线性复杂度（O(B*C*H*W)）；3. 局部位置增强（LEPE）
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim  # 总特征维度（如64）
        self.num_heads = num_heads  # 注意力头数（如4）
        self.head_dim = dim // num_heads  # 单头特征维度（如64//4=16）
        self.scale = self.head_dim ** (-0.5)  # 缩放因子（缓解梯度消失）

        # 1×1卷积：一次性生成Q（查询）、K（键）、V（值）、O（输出门控），减少参数数量
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)  # 输入dim，输出4*dim（QKV各dim，O dim）
        self.elu = nn.ELU()  # 门控激活函数：ELU+1确保输出非负（模拟概率门控）
        # 局部增强投影（LEPE）：5×5深度可分离卷积，捕捉局部空间依赖（弥补线性注意力全局不足）
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)  # 分组卷积=深度可分离，padding=2保持尺寸
        self.proj = nn.Conv2d(dim, dim, 1)  # 最终1×1卷积：融合注意力特征，恢复维度

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        前向传播流程：QKV生成→门控处理→RoPE编码→线性注意力计算→局部增强→输出
        '''
        B, C, H, W = x.shape  # 解析输入维度：[B, C, H, W]
        # 步骤1：生成QKV和O：[B, 4C, H, W] → QKV=[B, 3C, H, W]，O=[B, C, H, W]
        qkvo = self.qkvo(x)
        qkv = qkvo[:, :3 * self.dim, :, :]  # QKV特征（前3*dim通道）
        o = qkvo[:, 3 * self.dim:, :, :]  # 输出门控（后dim通道）

        # 步骤2：局部增强投影（LEPE）：对V特征做局部卷积，增强空间细节
        # 提取V特征（qkv的第3个dim通道：2*dim ~ 3*dim-1）
        lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])  # [B, C, H, W]

        # 步骤3：多头拆分与维度重排：将2D特征[B, 3C, H, W]转为多头格式[3, B, N, L, D]
        # m=3对应Q/K/V，n=num_heads，l=H×W（像素数），d=head_dim
        q, k, v = rearrange(
            qkv, 'b (m n d) h w -> m b n (h w) d',
            m=3, n=self.num_heads, d=self.head_dim
        )  # q/k/v: [B, N, L, D]（L=H×W）

        # 步骤4：门控处理：ELU+1确保Q/K非负，模拟“注意力门控”（抑制噪声特征）
        q = self.elu(q) + 1.0  # [B, N, L, D]，非负门控
        k = self.elu(k) + 1.0  # [B, N, L, D]，非负门控

        # 步骤5：高效注意力权重计算（线性复杂度）：避免传统QK^T的O(L²)计算
        q_mean = q.mean(dim=-2, keepdim=True)  # Q的序列维度均值：[B, N, 1, D]（全局Q统计）
        # 计算K的权重：Q均值 × K^T → 缩放 → Softmax → 转置，维度[B, N, L, 1]
        eff = self.scale * q_mean @ k.transpose(-1, -2)  # [B, N, 1, L]
        eff = torch.softmax(eff, dim=-1).transpose(-1, -2)  # [B, N, L, 1]
        # 加权K：K × 权重 × 序列长度（避免长度归一化偏差）
        k = k * eff * (H * W)  # [B, N, L, D]

        # 步骤6：RoPE编码：将位置信息融入Q和K（关键步骤，编码相对位置）
        q_rope = theta_shift(q, sin, cos)  # [B, N, L, D]（旋转后的Q）
        k_rope = theta_shift(k, sin, cos)  # [B, N, L, D]（旋转后的K）

        # 步骤7：线性注意力计算（O(L)复杂度）：替代传统QK^T的O(L²)矩阵乘法
        # 计算分母：Q × K均值^T + 小常数（避免除零），维度[B, N, L, 1]
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        # 计算KV乘积（全局上下文）：K^T × V → 维度[B, N, D, D]
        kv = (k_rope.transpose(-2, -1) * ((H * W) ** -0.5)) @ (v * ((H * W) ** -0.5))
        # 注意力输出：Q_rope × KV × 分母 → [B, N, L, D]
        res = q_rope @ kv * z

        # 步骤8：维度恢复与局部增强融合
        # 多头特征合并：[B, N, L, D] → [B, N*D, H, W] = [B, C, H, W]
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        # 融合LEPE局部特征：注意力输出 + 局部增强投影
        res = res + lepe

        # 步骤9：输出门控与最终投影：注意力特征 × 输出门控 → 1×1卷积融合
        return self.proj(res * o)

class RoPE3D(nn.Module):
    """
    3D 旋转位置编码：为 (T, H, W) 时空坐标生成 sin/cos 编码。
    与 2D RoPE 的区别：额外引入时间轴编码，使 token 感知帧间相对位置。
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        # 将 head_dim 拆为 3 个偶数段：temporal 分得余量（最重要的轴）
        d_hw = (head_dim // 6) * 2
        d_t = head_dim - 2 * d_hw
        self.d_t, self.d_h, self.d_w = d_t, d_hw, d_hw

        for name, d in [('angle_t', d_t), ('angle_h', d_hw), ('angle_w', d_hw)]:
            angle = 1.0 / (10000 ** torch.linspace(0, 1, d // 2))
            angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
            self.register_buffer(name, angle)

    def forward(self, slen):
        """
        slen: (T, H, W)
        Returns: (sin, cos) 各 [T*H*W, D]
        """
        T, H, W = slen
        idx_t = torch.arange(T).to(self.angle_t)
        idx_h = torch.arange(H).to(self.angle_h)
        idx_w = torch.arange(W).to(self.angle_w)

        sin_t = torch.sin(idx_t[:, None] * self.angle_t[None, :])[:, None, None, :].expand(-1, H, W, -1)
        cos_t = torch.cos(idx_t[:, None] * self.angle_t[None, :])[:, None, None, :].expand(-1, H, W, -1)

        sin_h = torch.sin(idx_h[:, None] * self.angle_h[None, :])[None, :, None, :].expand(T, -1, W, -1)
        cos_h = torch.cos(idx_h[:, None] * self.angle_h[None, :])[None, :, None, :].expand(T, -1, W, -1)

        sin_w = torch.sin(idx_w[:, None] * self.angle_w[None, :])[None, None, :, :].expand(T, H, -1, -1)
        cos_w = torch.cos(idx_w[:, None] * self.angle_w[None, :])[None, None, :, :].expand(T, H, -1, -1)

        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1).reshape(-1, self.d_t + self.d_h + self.d_w)
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1).reshape(-1, self.d_t + self.d_h + self.d_w)
        return sin, cos


class GateLinearAttention3D(nn.Module):
    """
    3D 门控线性注意力：跨帧时空注意力 + 3D RoPE。
    QKV 投影和 LEPE 仍按帧 Conv2d，注意力跨越 T*H*W。
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)

        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        """
        x: [B, C, T, H, W]
        sin, cos: [T*H*W, D]
        """
        B, C, T, H, W = x.shape
        L = T * H * W

        x_2d = rearrange(x, 'b c t h w -> (b t) c h w')
        qkvo = self.qkvo(x_2d)
        qkv = qkvo[:, :3 * self.dim, :, :]
        o = qkvo[:, 3 * self.dim:, :, :]

        lepe = self.lepe(qkv[:, 2 * self.dim:, :, :])

        qkv = rearrange(qkv, '(b t) c h w -> b t c h w', b=B, t=T)
        q, k, v = rearrange(
            qkv, 'b t (m n d) h w -> m b n (t h w) d',
            m=3, n=self.num_heads, d=self.head_dim
        )

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q_mean = q.mean(dim=-2, keepdim=True)
        eff = self.scale * q_mean @ k.transpose(-1, -2)
        eff = torch.softmax(eff, dim=-1).transpose(-1, -2)
        k = k * eff * L

        q_rope = theta_shift(q, sin, cos)
        k_rope = theta_shift(k, sin, cos)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (L ** -0.5)) @ (v * (L ** -0.5))
        res = q_rope @ kv * z

        res = rearrange(res, 'b n (t h w) d -> (b t) (n d) h w', t=T, h=H, w=W)
        res = res + lepe
        res = self.proj(res * o)
        return rearrange(res, '(b t) c h w -> b c t h w', b=B, t=T)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- 2D test ---
    x2d = torch.randn(1, 64, 32, 32).to(device)
    rope2d = RoPE(64, 4).to(device)
    attn2d = GateLinearAttentionNoSilu(64, 4).to(device)
    sin, cos = rope2d((32, 32))
    y2d = attn2d(x2d, sin, cos)
    print("2D 输入:", x2d.shape, "→ 输出:", y2d.shape)

    # --- 3D test ---
    x3d = torch.randn(1, 64, 4, 8, 8).to(device)
    rope3d = RoPE3D(64, 4).to(device)
    attn3d = GateLinearAttention3D(64, 4).to(device)
    sin3, cos3 = rope3d((4, 8, 8))
    y3d = attn3d(x3d, sin3, cos3)
    print("3D 输入:", x3d.shape, "→ 输出:", y3d.shape)