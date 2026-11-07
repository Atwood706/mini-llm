# model.py
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """
    预先计算每个位置、每个维度的 cos 和 sin
    返回 cos, sin，形状都是 (max_seq_len, head_dim)
    """
    # 每组的转速 θ_i = 1 / base^(2i/d)，一共 head_dim/2 组
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))   # (hd/2,)
    # 位置 m = 0, 1, ..., max_seq_len-1
    positions = torch.arange(max_seq_len).float()                                   # (T,)
    # 每个位置、每组的角度 mθ_i：外积
    angles = torch.outer(positions, inv_freq)                                       # (T, hd/2)
    # 前一半和后一半用同样的角度（因为第 j 维和第 j+hd/2 维是一组）
    angles = torch.cat([angles, angles], dim=-1)                                    # (T, hd)
    return angles.cos(), angles.sin()
    """
    cos 表（假设 max_seq_len=5，hd=4）：
             维0    维1    维2    维3
    位置0  [ 1.00  1.00  1.00  1.00 ]
    位置1  [ 0.54  1.00  0.54  1.00 ]
    位置2  [-0.42  1.00 -0.42  1.00 ]
    位置3  [-0.99  1.00 -0.99  1.00 ]
    位置4  [-0.65  1.00 -0.65  1.00 ]
    """


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维切成前后两半 [x1, x2]，返回 [-x2, x1]"""
    # TODO 1
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat([-x2, x1], dim = -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    对 x 做旋转位置编码
    x:        (..., T, hd)
    cos, sin: (T, hd)
    返回:     (..., T, hd)
    """
    # TODO 2
    return x * cos + rotate_half(x) * sin


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    单头因果自注意力
    q, k, v: (B, T, d)
    返回:    (B, T, d)
    """
    T, d = q.shape[-2], q.shape[-1]    # ✅ 不管前面有几维都能用

    # TODO 1：计算注意力分数 scores = Q K^T / sqrt(d)
    #         结果形状应为 (B, T, T)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)

    # TODO 2：构造因果掩码，把 "不能看的位置"（上三角，j > i）填成 -inf
    #         结果形状仍为 (B, T, T)
    mask = torch.tril(torch.ones(T, T, device = q.device))
    scores = scores.masked_fill(mask==0, float("-inf"))

    # TODO 3：对最后一维做 softmax，得到注意力权重
    #         形状 (B, T, T)，每一行加起来等于 1
    weights = F.softmax(scores, dim=-1)

    # TODO 4：用权重对 v 加权求和
    #         (B, T, T) @ (B, T, d) → (B, T, d)
    out = weights @ v

    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # 所有头的 W_q / W_k / W_v 各合成一个大矩阵
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # 输出投影 W_o
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)    # (max_seq_len, hd)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, hd = self.n_head, self.head_dim

        # TODO 1：计算 q、k、v，形状都是 (B, T, C)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # TODO 2：把 q、k、v 都变成 (B, H, T, hd)
        #         先 view 成 (B, T, H, hd)，再 transpose(1, 2)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)

        # 添加旋转
        q = apply_rope(q, self.cos[:T], self.sin[:T])
        k = apply_rope(k, self.cos[:T], self.sin[:T])

        # TODO 3：调用 causal_attention，得到 (B, H, T, hd)
         # out = causal_attention(q, k, v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # 原来是 causal_attention(q, k, v)

        # TODO 4：把多个头拼回去：(B, H, T, hd) → (B, T, H, hd) → (B, T, C)
        #         提示：transpose(1, 2)，然后 .contiguous().view(B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # TODO 5：输出投影
        out = self.o_proj(out)

        return out


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO：up_proj（C → 4C）、down_proj（4C → C），都 bias=False
        self.up_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.down_proj = nn.Linear(4 * d_model, d_model, bias=False)
        self.act = nn.GELU()


    def forward(self, x):          # x: (B, T, C)
        # TODO：升维 → GELU → 降维
        # 中间形状 (B, T, 4C)，输出 (B, T, C)
        x_up = self.up_proj(x)
        x_act = self.act(x_up)
        x_down = self.down_proj(x_act)
        return x_down

class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int):
        super().__init__()
        # TODO：4 个零件
        # ln1、attn（你的 MultiHeadAttention）、ln2、mlp（你的 MLP）
        self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, max_seq_len=max_seq_len)
        self.mlp = MLP(d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)


    def forward(self, x):          # x: (B, T, C)
        # TODO：两行，照着上面的公式写
        x_attn = x + self.attn(self.ln1(x))
        x_mlp = x_attn + self.mlp(self.ln2(x_attn))
        return x_mlp                   # (B, T, C)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, max_seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, n_head=n_head, max_seq_len=max_seq_len) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias = False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
             B, T, N = logits.shape
             loss = F.cross_entropy(logits.view(B*T, N), targets.view(B*T))

        return logits, loss

