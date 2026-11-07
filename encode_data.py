# encode_data.py
"""阶段 1：用训练好的分词器，把故事编码成 uint16 的二进制文件"""
import numpy as np
from tokenizers import Tokenizer

from config import parse_config

EOT = "<|endoftext|>"
CHUNK = 10_000   # 每批编码多少条故事


def load_stories(path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [s.strip() for s in text.split(EOT) if s.strip()]


def encode_to_bin(tokenizer: Tokenizer, stories: list[str], out_path, eot_id: int) -> int:
    """把故事编码后写进二进制文件，返回写入的 token 总数"""
    total = 0
    with open(out_path, "wb") as f:                      # "wb"：以二进制模式写入
        for i in range(0, len(stories), CHUNK):
            batch = stories[i : i + CHUNK]
            encodings = tokenizer.encode_batch(batch)    # 批量编码，比逐条编码快很多
            ids = []
            for enc in encodings:
                ids.extend(enc.ids)                      # 这条故事的 ids
                ids.append(eot_id)                       # 末尾加上 EOT
            arr = np.array(ids, dtype=np.uint16)         # 转成 uint16 数组
            arr.tofile(f)                                # 追加写入文件
            total += len(arr)
    return total


def main() -> None:
    cfg = parse_config()
    tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))
    eot_id = tokenizer.token_to_id(EOT)
    assert tokenizer.get_vocab_size() <= 65536, "词表太大，uint16 装不下"

    for name in ["train", "val"]:
        stories = load_stories(cfg.data_dir / f"{name}.txt")
        out_path = cfg.data_dir / f"{name}.bin"
        n_tokens = encode_to_bin(tokenizer, stories, out_path, eot_id)
        size_mb = out_path.stat().st_size / 1e6
        print(f"{name}: {len(stories)} 条故事 → {n_tokens:,} 个 token → {out_path.name}（{size_mb:.1f} MB）")


if __name__ == "__main__":
    main()