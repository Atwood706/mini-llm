# check_bin.py
"""检查编码后的二进制文件"""
import numpy as np
from tokenizers import Tokenizer

from config import get_config


def main() -> None:
    cfg = get_config()
    tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))

    # 用 memmap 打开 train.bin：不会把整个文件读进内存
    data = np.memmap(cfg.data_dir / "train.bin", dtype=np.uint16, mode="r")

    # TODO 1：打印 data 的长度（token 总数）和 data.dtype，
    #         和 encode_data.py 打印的 token 数对比，应该完全一致
    print(f"data长度： {len(data)}")
    print(f"data的数据类型： {data.dtype}")
    print(data)

    # TODO 2：统计 data 里有多少个 EOT（编号 0），应该正好等于故事条数 20000
    #         提示：(data == 0) 会得到一个布尔数组，对它求和 .sum() 就是 True 的个数
    EOT_cnt = [(single_data == 0) for single_data in data]
    print(f"故事的数量：{sum(EOT_cnt)}")

    # TODO 3：取出前 100 个 token（切片 data[:100]），
    #         用 tokenizer.decode(...) 解码成文字打印出来，应该是 train.txt 里第一个故事的开头
    #         提示：decode 需要传入 Python 列表，可以用 .tolist() 转换
    data_100 = data[:100]
    story_100 = tokenizer.decode(data_100.tolist())
    print(story_100)

if __name__ == "__main__":
    main()