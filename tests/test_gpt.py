# tests/test_gpt.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model import GPT

torch.manual_seed(0)
V, C, N, H, T, B = 100, 32, 2, 4, 8, 2
model = GPT(vocab_size=V, d_model=C, n_layer=N, n_head=H, max_seq_len=T)
idx = torch.randint(0, V, (B, T))          # 随机 token 编号，(B, T)

# 检查 1：形状
logits, _ = model(idx)
print("1. logits 形状：", logits.shape, "（应为 torch.Size([2, 8, 100])）")

# 检查 2：参数量 = 嵌入 VC + N 个 Block (12C² + 4C) + ln_f 2C + lm_head VC
n_params = sum(p.numel() for p in model.parameters())
expected = V * C + N * (12 * C * C + 4 * C) + 2 * C    # 共享后，嵌入和输出头只算一次
print("2. 参数量：", n_params, "（应为", expected, "）")

# 检查 3：因果性 —— 改最后一个 token，前面位置的 logits 不变
idx2 = idx.clone()
idx2[:, -1] = (idx2[:, -1] + 1) % V
logits2, _ = model(idx2)
print("3. 前面位置不受后面影响：", torch.allclose(logits[:, :-1], logits2[:, :-1], atol=1e-5))
print("   最后位置改变：", not torch.allclose(logits[:, -1], logits2[:, -1], atol=1e-5))

# 检查 4：初始 loss 应接近 ln(V)
import math
targets = torch.randint(0, V, (B, T))
_, loss = model(idx, targets)
print("4. 初始 loss：", round(loss.item(), 3), "（应接近 ln(V) =", round(math.log(V), 3), "）")