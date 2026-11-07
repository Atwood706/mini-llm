# train.py
"""阶段 3：预训练"""
import numpy as np
import torch
import math
from config import parse_config
from model import GPT
from utils import get_device, set_seed
from contextlib import nullcontext
import time


def get_batch(split: str, cfg, device):
    """从 train.bin 或 val.bin 随机取一个 batch，返回 x, y，形状都是 (B, T)"""
    data = np.memmap(cfg.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    B, T = cfg.batch_size, cfg.block_size

    # TODO 1：随机取 B 个起点，范围要保证每段 T+1 个 token 不越界
    starts = np.random.randint(0, len(data) - T - 1, size=B)

    # TODO 2：切出 x 和 y，转成 int64，再拼成 (B, T)
    x = torch.stack([torch.from_numpy(data[s : s + T].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(data[s + 1 : s + T + 1].astype(np.int64)) for s in starts])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, cfg, device, ctx, eval_iters: int = 20) -> dict:
    """在 train 和 val 上各取 eval_iters 个 batch，返回平均 loss"""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            # TODO：取一个 split 的 batch，算 loss，存进 losses[i]
            x, y = get_batch(split, cfg, device)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(step: int, cfg) -> float:
    """warmup + cosine 学习率调度"""
    max_lr = cfg.lr
    min_lr = max_lr / 10                     # 最低降到最大值的 1/10
    warmup = max(cfg.max_iters // 20, 1)     # 前 5% 的步数用来 warmup

    # TODO 1：warmup 阶段，线性增加
    if step < warmup:
        return (max_lr * (step + 1)) / warmup

    # TODO 2：cosine 阶段
    ratio = (step - warmup) / (cfg.max_iters - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))    # 从 1 平滑降到 0
    return min_lr + coeff * (max_lr - min_lr)

def main() -> None:
    cfg = parse_config()
    set_seed(0)
    device = get_device()
    print(f"预设：{cfg.preset}，设备：{device}")

    model = GPT(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layer=cfg.n_layer,
        n_head=cfg.n_head, max_seq_len=cfg.block_size,
    ).to(device)
        # 混合精度：只在 CUDA 上用 bf16，其他设备保持 float32
    if device.type == "cuda":
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()       # 一个"什么都不做"的占位符


    print(f"参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    start_step = 0
    ckpt_path = cfg.out_dir / "ckpt.pt"
    if cfg.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        # TODO：把模型参数、优化器状态装回去，并算出从第几步继续
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step'] + 1
        print(f"从 checkpoint 恢复，继续 step {start_step}")

    t0 = time.time()

    for step in range(start_step, cfg.max_iters):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        # TODO：取一个新 batch，然后做训练的 4 

        optimizer.zero_grad()
        for micro_step in range(cfg.grad_accum_steps):
            x, y = get_batch("train", cfg, device)             # 取一个小 batch
            with ctx:
                _, loss = model(x, y)                          # 前向
            loss = loss / cfg.grad_accum_steps                 # 除以累积次数，保证累加后是平均值
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model, cfg, device, ctx)
            print(f"step {step:4d}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"lr {lr:.6f}  norm {norm:.2f}  已用 {time.time() - t0:.0f} 秒")


            # TODO：把训练状态打包成一个字典
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "model_args": dict(
                    vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layer=cfg.n_layer,
                    n_head=cfg.n_head, max_seq_len=cfg.block_size,
                ),
            }
            torch.save(checkpoint, cfg.out_dir / "ckpt.pt")

if __name__ == "__main__":
    main()