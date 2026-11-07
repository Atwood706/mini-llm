# train_tokenizer.py
"""阶段 1：在训练集上训练 byte-level BPE 分词器"""
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from config import parse_config

EOT = "<|endoftext|>"


def load_stories(path) -> list[str]:
    """读取 train.txt，按 EOT 切回一条条故事（和你写的 inspect_data 一样）"""
    text = path.read_text(encoding="utf-8")
    return [s.strip() for s in text.split(EOT) if s.strip()]


def main() -> None:
    cfg = parse_config()
    stories = load_stories(cfg.data_dir / "train.txt")
    print(f"用 {len(stories)} 条故事训练分词器，目标词表大小 {cfg.vocab_size}")

    # 1. 模型：BPE
    tokenizer = Tokenizer(models.BPE())
    # 2. 预切分：先把文本转成字节，再按空格、标点粗切（BPE 只在每一小块内部合并）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # 3. 解码器：把字节还原成文字（例如 Ġ 还原成空格）
    tokenizer.decoder = decoders.ByteLevel()

    # 4. 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=[EOT],                                    # 特殊 token，不会被拆开，编号为 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),    # 初始词表：全部 256 个字节
        min_frequency=2,                                         # 只出现 1 次的组合不合并
    )
    tokenizer.train_from_iterator(stories, trainer=trainer)

    tokenizer.save(str(cfg.tokenizer_path))
    print(f"词表大小：{tokenizer.get_vocab_size()}，已保存到 {cfg.tokenizer_path}")


if __name__ == "__main__":
    main()