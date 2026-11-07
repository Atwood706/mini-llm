# check_tokenizer.py
"""检查分词器是否正常工作"""
from tokenizers import Tokenizer

from config import get_config

EOT = "<|endoftext|>"


def main() -> None:
    cfg = get_config()
    tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))

    sentence = "Once upon a time, there was a little dog named Max."

    # TODO 1：用 tokenizer.encode(sentence) 编码，得到 enc。
    #         打印 enc.tokens（切出来的 token）和 enc.ids（对应的编号）
    enc = tokenizer.encode(sentence)
    print(enc.tokens)
    print(enc.ids)

    # TODO 2：用 tokenizer.decode(enc.ids) 把编号还原成文字，
    #         打印出来，并打印它是否和 sentence 完全相同（用 ==）
    dec = tokenizer.decode(enc.ids)
    print(dec)
    print(f"是否相等{dec==sentence}")

    # TODO 3：打印 EOT 的编号。提示：tokenizer.token_to_id(EOT)
    print(f"EOT的编码： {tokenizer.token_to_id(EOT)}")

    # TODO 4：读取 val.txt，按 EOT 切成故事列表，
    #         对每条故事编码，统计"平均每条故事多少个 token"和"平均每个 token 多少个字符"
    path = cfg.data_dir / "val.txt"
    valid_text = path.read_text(encoding = "utf-8")
    valid_stories = [s.strip() for s in valid_text.split(EOT)]
    valid_stories_len = [len(tokenizer.encode(s).tokens) for s in valid_stories]
    print(f"平均每个故事 {sum(valid_stories_len)/len(valid_stories_len):.1f} 个token")

    valid_char = [len(s) for s in valid_stories]
    print(f"平均每个token {sum(valid_char)/sum(valid_stories_len):.1f} 个字符")
    


if __name__ == "__main__":
    main()