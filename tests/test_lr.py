# tests/test_lr.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config
from train import get_lr

cfg = get_config("small")          # lr=1e-3，max_iters=500，warmup=25
for step in [0, 12, 24, 25, 100, 262, 400, 499]:
    print(f"step {step:3d}  lr {get_lr(step, cfg):.6f}")

lrs = [get_lr(s, cfg) for s in range(cfg.max_iters)]
print("最大值：", max(lrs), "（应为 0.001）")
print("warmup 后单调下降：", all(lrs[i] >= lrs[i + 1] for i in range(25, cfg.max_iters - 1)))