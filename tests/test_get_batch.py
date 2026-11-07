# tests/test_get_batch.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import get_config
from train import get_batch
from utils import get_device

cfg = get_config("small")
device = get_device()
B, T = cfg.batch_size, cfg.block_size

x, y = get_batch("train", cfg, device)

# 检查 1：形状和类型
print("1. 形状：", tuple(x.shape), tuple(y.shape), f"（应为 ({B}, {T})）")
print("   类型：", x.dtype, "（应为 torch.int64）")
print("   设备：", x.device)

# 检查 2：y 是 x 往左错开一位
print("2. 错位正确：", torch.equal(x[:, 1:], y[:, :-1]))

# 检查 3：编号都在词表范围内
print("3. 编号范围：", x.min().item(), "~", x.max().item(), f"（应在 0 ~ {cfg.vocab_size - 1}）")

# 检查 4：每次调用取到的数据不同（随机）
x2, _ = get_batch("train", cfg, device)
print("4. 两次取的不一样：", not torch.equal(x, x2))

# 检查 5：验证集也能取
xv, yv = get_batch("val", cfg, device)
print("5. 验证集形状：", tuple(xv.shape))