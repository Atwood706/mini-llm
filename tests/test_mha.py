# tests/test_mha.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import MultiHeadAttention

torch.manual_seed(0)
B, T, C, H = 2, 6, 32, 4
mha = MultiHeadAttention(d_model=C, n_head=H, max_seq_len=T)
x = torch.randn(B, T, C)

# 检查 1：形状
out = mha(x)
print("1. 输出形状：", out.shape, "（应为 torch.Size([2, 6, 32])）")

# 检查 2：参数量 = 4 个 C×C 矩阵
n_params = sum(p.numel() for p in mha.parameters())
print("2. 参数量：", n_params, "（应为", 4 * C * C, "）")

# 检查 3：因果性 —— 修改最后一个 token 的输入，前面位置的输出不变
x2 = x.clone()
x2[:, -1] += 10.0
out2 = mha(x2)
print("3. 前面位置不受后面影响：", torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5))

# 检查 4：反过来 —— 修改第一个 token，后面位置的输出应该改变
x3 = x.clone()
x3[:, 0] += 10.0
out3 = mha(x3)
print("4. 后面位置能看到前面：", not torch.allclose(out[:, 1:], out3[:, 1:], atol=1e-5))