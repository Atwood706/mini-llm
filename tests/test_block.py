# tests/test_block.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import Block

torch.manual_seed(0)
B, T, C, H = 2, 6, 32, 4
block = Block(d_model=C, n_head=H, max_seq_len=T)
x = torch.randn(B, T, C)

# 检查 1：形状
out = block(x)
print("1. 输出形状：", out.shape, "（应为 torch.Size([2, 6, 32])）")

# 检查 2：参数量 = 注意力 4C² + MLP 8C² + 两个 LN 各 2C
n_params = sum(p.numel() for p in block.parameters())
print("2. 参数量：", n_params, "（应为", 12 * C * C + 4 * C, "）")

# 检查 3：因果性 —— 改最后一个 token，前面位置的输出不变
x2 = x.clone()
x2[:, -1] += 10.0
out2 = block(x2)
print("3. 前面位置不受后面影响：", torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5))

# 检查 4：Pre-LN 主干道 —— 把 attn 和 mlp 的输出层权重清零，Block 应该原样输出 x
with torch.no_grad():
    block.attn.o_proj.weight.zero_()
    block.mlp.down_proj.weight.zero_()
print("4. 支路为 0 时输出 = 输入：", torch.allclose(block(x), x))