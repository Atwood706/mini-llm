import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent

def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()

DATA_DIR = _env_path("MINILLM_DATA_DIR", PROJECT_ROOT / "workspace" / "data")
OUT_DIR = _env_path("MINILLM_OUT_DIR", PROJECT_ROOT / "workspace" / "out")

# ---------- 配置 ----------
@dataclass
class Config:
    preset: str                        # 预设名称："small"（Mac）或 "full"（3080）

    # 数据
    num_train_stories: Optional[int]   # 用多少条故事训练；None 表示全部
    vocab_size: int                    # 词表大小

    # 模型
    n_layer: int                       # Transformer 层数
    n_head: int                        # 注意力头数
    d_model: int                       # 隐藏层维度
    block_size: int                    # 上下文长度（一次最多看多少个 token）

    # 训练
    batch_size: int
    grad_accum_steps: int              # 梯度累积步数
    max_iters: int                     # 总训练步数
    lr: float                          # 学习率
    eval_interval: int                 # 每多少步评估一次

    # 运行选项
    resume: bool = False               # 新增：是否从 checkpoint 续训（有默认值，必须放在最后）

    # ---------- 由预设名称推导出的路径 ----------
    @property
    def data_dir(self) -> Path:
        return DATA_DIR / f"tinystories_{self.preset}"

    @property
    def tokenizer_path(self) -> Path:
        return self.data_dir / "tokenizer.json"

    @property
    def out_dir(self) -> Path:
        return OUT_DIR / self.preset

    def make_dirs(self) -> None:
        """创建数据和输出文件夹（已存在则跳过）"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)


# ---------- 预设 ----------
PRESETS = {
    # Mac：只用来验证流程能跑通
    "small": dict(
        num_train_stories=20_000, vocab_size=4096,
        n_layer=4, n_head=4, d_model=128, block_size=256,
        batch_size=16, grad_accum_steps=1, max_iters=500, lr=1e-3, eval_interval=100,
    ),
    # 3080：正式训练（数值是初始设定，之后训练时再调）
    "full": dict(
        num_train_stories=None, vocab_size=8192,
        n_layer=8, n_head=8, d_model=512, block_size=512,
        batch_size=32, grad_accum_steps=4, max_iters=10_000, lr=6e-4, eval_interval=500,
    ),
}


def get_config(preset: Optional[str] = None, resume: bool = False) -> Config:   # 新增参数 resume
    """
    根据预设名称返回配置。选择 preset 的优先级：
    1. 调用时直接传入的参数
    2. 环境变量 MINILLM_PRESET
    3. 自动判断：有 CUDA 就用 full，否则用 small
    """
    if preset is None:
        preset = os.environ.get("MINILLM_PRESET")
    if preset is None:
        preset = "full" if torch.cuda.is_available() else "small"

    if preset not in PRESETS:
        raise ValueError(f"未知的预设 {preset!r}，可选：{list(PRESETS)}")

    cfg = Config(preset=preset, resume=resume, **PRESETS[preset])   # 新增：传入 resume
    cfg.make_dirs()
    return cfg


def parse_config() -> Config:
    """给各个脚本用的入口，支持命令行参数：python train.py --preset small --resume"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default=None)
    parser.add_argument("--resume", action="store_true")    # 新增：写了 --resume 就是 True
    args = parser.parse_args()
    return get_config(args.preset, args.resume)