# prepare_data.py
"""阶段 1：下载 TinyStories 并保存为文本文件"""
from datasets import load_dataset

from config import parse_config

DATASET_NAME = "roneneldan/TinyStories"
EOT = "<|endoftext|>"   # 故事之间的分隔符（end of text）


def load_stories(split: str, n: int | None) -> list[str]:
    """读取 n 条故事；n 为 None 时读取全部"""
    if n is None:
        ds = load_dataset(DATASET_NAME, split=split)                            # 完整下载
    else:
        ds = load_dataset(DATASET_NAME, split=split, streaming=True).take(n)    # 流式读取前 n 条
    stories = [ex["text"].strip() for ex in ds]
    return [s for s in stories if s]   # 去掉空故事


def save_stories(stories: list[str], path) -> None:
    """把所有故事写进一个文本文件，故事之间用 EOT 隔开"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"\n{EOT}\n".join(stories))


def main() -> None:
    cfg = parse_config()
    n_train = cfg.num_train_stories
    n_val = None if n_train is None else max(n_train // 20, 500)   # 验证集取训练集的 1/20

    print(f"预设：{cfg.preset}，数据目录：{cfg.data_dir}")

    for split, n, filename in [("train", n_train, "train.txt"), ("validation", n_val, "val.txt")]:
        stories = load_stories(split, n)
        path = cfg.data_dir / filename
        save_stories(stories, path)
        size_mb = path.stat().st_size / 1e6
        print(f"{split}: {len(stories)} 条故事 → {path.name}（{size_mb:.1f} MB）")


if __name__ == "__main__":
    main()