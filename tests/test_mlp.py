# tests/test_mlp.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import MLP

torch.manual_seed(0)
B, T, C = 2, 6, 32
mlp = MLP(d_model=C)
x = torch.randn(B, T, C)

# 检查 1：形状
out = mlp(x)
print("1. 输出形状：", out.shape, "（应为 torch.Size([2, 6, 32])）")

# 检查 2：参数量 = 2 个 C×4C 矩阵 = 8C²
n_params = sum(p.numel() for p in mlp.parameters())
print("2. 参数量：", n_params, "（应为", 8 * C * C, "）")

# 检查 3：逐位置独立 —— 只改位置 2，其他位置的输出完全不变
x2 = x.clone()
x2[:, 2] += 10.0
out2 = mlp(x2)
others = [0, 1, 3, 4, 5]
print("3. 其他位置不变：", torch.allclose(out[:, others], out2[:, others]))
print("   位置 2 改变：", not torch.allclose(out[:, 2], out2[:, 2]))