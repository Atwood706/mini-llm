# sample.py
"""阶段 4：加载训练好的模型，生成故事"""
import torch
from tokenizers import Tokenizer
import torch.nn.functional as F
from config import parse_config
from model import GPT
from utils import get_device

EOT = "<|endoftext|>"

@torch.no_grad()
def generate(model, idx, max_new_tokens: int, max_seq_len: int, eot_id: int,
             temperature: float = 1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]                       # (1, V)

        # TODO：除以温度 → softmax 变成概率 → 按概率抽一个
        logits =  logits / temperature                                 # (1, V)
        probs = F.softmax(logits, dim = -1)              # (1, V)，每行加起来等于 1
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)

        idx = torch.cat((idx, next_id), dim=1)
        if next_id.item() == eot_id:
            break
    return idx

def main() -> None:
    cfg = parse_config()
    device = get_device()

    # 1. 加载 checkpoint，按保存时的结构参数建模型，再装入权重
    ckpt = torch.load(cfg.out_dir / "ckpt.pt", map_location=device)
    model = GPT(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"已加载 step {ckpt['step']} 的模型，设备：{device}")

    # 2. 加载分词器
    tokenizer = Tokenizer.from_file(str(cfg.tokenizer_path))
    eot_id = tokenizer.token_to_id(EOT)

    # 3. 把开头的文本编码成 (1, T) 的张量
    prompt = "Once upon a time"
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)   # (1, T)

    # 4. 生成，再解码回文字
    out = generate(model, idx, max_new_tokens=200,
                   max_seq_len=ckpt["model_args"]["max_seq_len"], eot_id=eot_id, temperature=0.8)
    print("-" * 50)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()