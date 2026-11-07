# check_model.py
"""第一次用真实数据跑模型：从 train.bin 取一个 batch，看初始 loss"""
import math

import numpy as np
import torch

from config import parse_config
from model import GPT
from utils import get_device, set_seed


def main() -> None:
    cfg = parse_config()
    set_seed(0)
    device = get_device()
    print(f"预设：{cfg.preset}，设备：{device}")

    # 1. 用 memmap 打开 train.bin：不会一次读进内存，按需读取
    data = np.memmap(cfg.data_dir / "train.bin", dtype=np.uint16, mode="r")
    print(f"train.bin 共 {len(data):,} 个 token")

    # 2. 随机取 B 个起点，每段取 T+1 个 token：前 T 个是输入，后 T 个是目标（错开一位）
    B, T = cfg.batch_size, cfg.block_size
    starts = np.random.randint(0, len(data) - T - 1, size=B)
    x = torch.stack([torch.from_numpy(data[s : s + T].astype(np.int64)) for s in starts])          # (B, T)
    y = torch.stack([torch.from_numpy(data[s + 1 : s + 1 + T].astype(np.int64)) for s in starts])  # (B, T)
    x, y = x.to(device), y.to(device)
    print(f"x: {tuple(x.shape)}，y: {tuple(y.shape)}")
    print("x[0] 前 8 个：", x[0, :8].tolist())
    print("y[0] 前 8 个：", y[0, :8].tolist(), "← 应该是 x 往左挪一位")

    # 3. 建模型
    model = GPT(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, max_seq_len=cfg.block_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量：{n_params / 1e6:.2f}M")

    # 4. 前向一次（不需要梯度）
    with torch.no_grad():
        logits, loss = model(x, y)
    print(f"logits: {tuple(logits.shape)}")
    print(f"初始 loss：{loss.item():.3f}（应接近 ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}）")


if __name__ == "__main__":
    main()