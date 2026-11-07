# tests/test_attention.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 让 tests/ 里能 import 根目录的 model.py

import torch
import torch.nn.functional as F

from model import causal_attention

torch.manual_seed(0)
B, T, d = 2, 5, 8
q, k, v = torch.randn(B, T, d), torch.randn(B, T, d), torch.randn(B, T, d)

# 检查 1：形状
out = causal_attention(q, k, v)
print("1. 输出形状：", out.shape, "（应为 torch.Size([2, 5, 8])）")

# 检查 2：和 PyTorch 官方实现的结果对比
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("2. 和官方实现一致：", torch.allclose(out, ref, atol=1e-5))

# 检查 3：因果性 —— 修改最后一个 token，前面位置的输出不应该改变
k2, v2 = k.clone(), v.clone()
k2[:, -1] += 100.0
v2[:, -1] += 100.0
out2 = causal_attention(q, k2, v2)
print("3. 前面位置不受后面影响：", torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5))