#!/usr/bin/env python3
"""深度外推量化评测: loss vs n_loops

加载最新 checkpoints_act checkpoint, 在固定的一批验证数据上扫描推理循环
深度 n_loops ∈ {1,2,4,8,12,16}, 记录:
  - 平均 CE loss (无 kv_cache, ACT 早停生效 → 只有不停机的位置消耗额外循环)
  - 实际执行的循环数 (act["loops"], 批级早停后的真实值)
  - 每个 batch 前向耗时

判读(对照 Ouro 论文 §5.3, 其模型训练 T_max=4, 外推 5-8 温和退化):
  loss 最低点出现在 ≤8 (训练深度) 为正常; >8 温和上翘 = 与论文一致的
  外推退化; >8 仍下降 = 外推性好, 推理时可加深换质量。

用法:
    python tests/test_depth_extrap.py [--ckpt checkpoints_act/step_0001000.pt] \
        [--batches 8] [--loops 1,2,4,8,12,16]
"""

import argparse
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_infer import find_latest_checkpoint, load_inference_model  # noqa: E402

SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_eval_batches(n_batches):
    """从训练数据流取固定的 n_batches 个 (x, y) 作为验证集
    (复用训练脚本的 SkyworkDataset, 与训练分布一致)。"""
    import importlib.util
    from torch.utils.data import DataLoader
    spec = importlib.util.spec_from_file_location(
        "train_mod",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "training", "0.1b_fine_skywork.py"),
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    dataset = train_mod.SkyworkDataset(
        seq_len=SEQ_LEN, rank=0, world_size=1, start_step=0,
        grad_accum=4, micro_batch=1,
        tokenizer_model_id="Langboat/mengzi-t5-base",
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
    batches = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        batches.append((x.to(DEVICE), y.to(DEVICE)))
    return batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="默认自动选 checkpoints_act 最新")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--loops", default="1,2,4,8,12,16")
    args = p.parse_args()

    ckpt = args.ckpt or find_latest_checkpoint(
        "/home/ranhao/projects/OpenMythos/checkpoints_act")
    print(f"[extrap] checkpoint: {ckpt}")
    model = load_inference_model(ckpt, DEVICE)
    vocab = model.embed.weight.shape[0]

    print(f"[extrap] 拉取 {args.batches} 个验证 batch ...")
    batches = get_eval_batches(args.batches)

    print(f"\n[extrap] {'n_loops':>7} | {'loss':>7} | {'实际循环':>8} | "
          f"{'ms/batch':>8}")
    results = []
    for n in [int(s) for s in args.loops.split(",")]:
        loss_tot, loops_max, dt_tot = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in batches:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, act = model(x, n_loops=n, return_act=True)
                    loss = torch.nn.functional.cross_entropy(
                        logits.float().view(-1, vocab), y.view(-1))
                torch.cuda.synchronize()
                dt_tot += time.perf_counter() - t0
                loss_tot += loss.item()
                loops_max = max(loops_max, act["loops"])
        nb = len(batches)
        results.append((n, loss_tot / nb, loops_max, dt_tot / nb * 1e3))
        print(f"[extrap] {n:>7} | {loss_tot/nb:7.4f} | {loops_max:>8.1f} | "
              f"{dt_tot/nb*1e3:>8.1f}", flush=True)

    best = min(results, key=lambda r: r[1])
    print(f"\n[extrap] loss 最低点: n_loops={best[0]} (loss {best[1]:.4f})")
    print("[extrap] 判读: 最低点 ≤8 (训练深度) 且 >8 温和上翘 = 与 Ouro 一致;")


if __name__ == "__main__":
    main()
