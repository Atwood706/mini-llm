# tests/test_rope.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import apply_rope, precompute_rope, rotate_half

torch.manual_seed(0)
hd, T = 8, 20
cos, sin = precompute_rope(hd, T)

# 检查 1：rotate_half 的结果
x = torch.arange(8).float()
print("1. rotate_half：", rotate_half(x).tolist(), "（应为 [-4, -5, -6, -7, 0, 1, 2, 3]）")

# 检查 2：位置 0 不旋转
v = torch.randn(1, 1, T, hd)
out = apply_rope(v, cos, sin)
print("2. 位置 0 不变：", torch.allclose(out[..., 0, :], v[..., 0, :]))

# 检查 3：旋转不改变长度
print("3. 长度不变：", torch.allclose(out.norm(dim=-1), v.norm(dim=-1), atol=1e-5))

# 检查 4：点积只取决于相对距离
q = torch.randn(hd).expand(1, 1, T, hd)   # 同一个 q 放在每个位置
k = torch.randn(hd).expand(1, 1, T, hd)   # 同一个 k 放在每个位置
rq, rk = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
dot_a = (rq[..., 2, :] * rk[..., 5, :]).sum()     # 位置 2 和 5，距离 3
dot_b = (rq[..., 10, :] * rk[..., 13, :]).sum()   # 位置 10 和 13，距离 3
dot_c = (rq[..., 2, :] * rk[..., 9, :]).sum()     # 位置 2 和 9，距离 7
print("4a. 距离相同，点积相同：", torch.allclose(dot_a, dot_b, atol=1e-5))
print("4b. 距离不同，点积不同：", not torch.allclose(dot_a, dot_c, atol=1e-5))