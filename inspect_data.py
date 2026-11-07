# inspect_data.py
"""查看训练数据的基本统计"""
import random

from config import get_config

EOT = "<|endoftext|>"


def main() -> None:
    cfg = get_config()
    path = cfg.data_dir / "train.txt"

    text = path.read_text(encoding="utf-8")   # 整个文件读成一个大字符串

    # TODO 1：按 EOT 把 text 切成一条一条的故事，得到一个列表 stories
    stories = [s.strip() for s in text.split(EOT)]

    # TODO 2：算出每条故事的字符数，得到一个列表 lengths
    lengths = [len(story) for story in stories]
    print(f"故事条数：{len(stories)}")

    # TODO 3：打印平均字符数、最长字符数、最短字符数
    average_len = sum(lengths)/len(stories)
    max_len = max(lengths)
    min_len = min(lengths)
    print(f"平均长度{average_len:.0f}")
    print(f"最长的故事{max_len}")
    print(f"最短的故事{min_len}")

    print("\n----- 随机一条故事 -----")
    # TODO 4：随机挑一条故事打印出来
    print(random.choice(stories))


if __name__ == "__main__":
    main()