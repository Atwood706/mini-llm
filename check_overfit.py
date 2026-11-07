# check_overfit.py
"""单 batch 过拟合：同一个 batch 反复训练，loss 应该降到接近 0"""
import numpy as np
import torch

from config import parse_config
from model import GPT
from utils import get_device, set_seed


def main() -> None:
    cfg = parse_config()
    set_seed(0)
    device = get_device()

    # 1. 取一个固定的小 batch（和 check_model.py 一样，只是 B 改小，跑得更快）
    data = np.memmap(cfg.data_dir / "train.bin", dtype=np.uint16, mode="r")
    B, T = 4, 64
    starts = np.random.randint(0, len(data) - T - 1, size=B)
    x = torch.stack([torch.from_numpy(data[s : s + T].astype(np.int64)) for s in starts]).to(device)
    y = torch.stack([torch.from_numpy(data[s + 1 : s + 1 + T].astype(np.int64)) for s in starts]).to(device)

    # 2. 模型和优化器
    model = GPT(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, max_seq_len=cfg.block_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 3. 训练循环：每一步都用同一个 (x, y)
    for step in range(300):
        # TODO：训练一步的 4 件事
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % 30 == 0 or step == 299:
            print(f"step {step:3d}  loss {loss.item():.4f}")


if __name__ == "__main__":
    main()