#!/usr/bin/env python3
"""
OpenMythos PPL（困惑度）评测脚本

从 SkyPile-150B 流式随机抽样 N 篇文本（默认 100），按训练时的 seq_len 切窗，
用 next-token 交叉熵计算困惑度。这是预训练模型最基础的能力指标，可用来
横向对比不同 checkpoint。

抽样方式（如实说明）：流式数据集无法对全库做均匀随机 —— 脚本流式读取前
--buffer_docs 篇（默认 500），对其做 reservoir 均匀抽样取 --n_docs 篇。
对预训练 PPL 评估来说，这个前缀窗口内的随机样本已经足够有代表性。

网络说明：
  - SkyPile 流式读取与训练脚本完全同款（MsDataset + use_streaming=True，
    训练进程正在用同一方式跑着，证明可用）。整个进程只调用一次
    MsDataset.load，避开训练脚本注释里记录的重复调用补丁递归 bug。
  - tokenizer 自动解析到 HF 本地缓存快照目录加载，不发出任何在线请求
    （hf-mirror 在当前网络下会无限挂起，见 tests/eval_cmmlu.py 注释）。

用法：
    # 默认：100 篇文本，seq_len=1024，n_loops=8
    python tests/eval_ppl.py

    # 快速验证：3 篇文本，只流式读前 10 篇做抽样
    python tests/eval_ppl.py --n_docs 3 --buffer_docs 10

    # 指定 checkpoint / 循环深度 / 窗口长
    python tests/eval_ppl.py --checkpoint checkpoints_act/step_0013200.pt \
        --n_loops 4 --seq_len 1024
"""

import os

# 与训练脚本一致的环境变量（MsDataset 流式读取的配套设置）。
# tokenizer 不走网络（本地快照），这些只影响 datasets/modelscope 的流式管线。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import math
import random
import argparse
import statistics
import torch
import torch.nn.functional as F
from loguru import logger

# 以 `python tests/eval_ppl.py` 运行时 sys.path[0] 是 tests/，补上项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mythos.tokenizer import MythosTokenizer
from tests.test_infer import find_latest_checkpoint, load_inference_model


def resolve_tokenizer_path(model_id: str) -> str:
    """
    hub id（如 Langboat/mengzi-t5-base）→ HF 本地缓存快照目录；
    已是本地路径或缓存不存在时原样返回。避免 from_pretrained 的在线元数据请求。
    """
    if os.path.exists(model_id):
        return model_id
    cache = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots"
    )
    if os.path.isdir(cache):
        snaps = sorted(os.listdir(cache))
        if snaps:
            return os.path.join(cache, snaps[-1])
    return model_id


def sample_texts(dataset_name: str, n_docs: int, buffer_docs: int, seed: int) -> list:
    """
    流式读取前 buffer_docs 篇，reservoir 均匀抽样 n_docs 篇文本。

    reservoir 抽样保证窗口内每篇被选中的概率都是 n_docs/buffer_docs。
    流中途断线时：若已抽满 n_docs 就用现有样本继续，否则抛出。
    """
    from modelscope.msdatasets import MsDataset

    logger.info(f"开始流式读取 {dataset_name}（前 {buffer_docs} 篇中抽 {n_docs} 篇）...")
    ds = MsDataset.load(dataset_name, split="train", use_streaming=True)

    rng = random.Random(seed)
    reservoir = []
    seen = 0
    try:
        for sample in ds:
            text = sample.get("text", sample.get("content", ""))
            if not text or not text.strip():
                continue
            seen += 1
            if len(reservoir) < n_docs:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < n_docs:
                    reservoir[j] = text
            if seen >= buffer_docs:
                break
    except Exception as exc:
        if len(reservoir) < n_docs:
            raise
        logger.warning(f"数据流中断（{type(exc).__name__}），已读 {seen} 篇，"
                       f"用现有 {len(reservoir)} 篇样本继续")
    logger.info(f"抽样完成：读了 {seen} 篇，抽出 {len(reservoir)} 篇")
    return reservoir


@torch.no_grad()
def doc_nll(model, ids: list, device: str, n_loops: int, seq_len: int,
            min_tail: int = 64):
    """
    计算一篇文本的 (nll_sum, n_tokens)。

    按 seq_len 不重叠切窗，每窗取 seq_len+1 个 token（前 seq_len 个做输入、
    后移一位做标签，与训练时的 (input, target) 构造一致）。不足 min_tail
    的尾巴窗口丢弃，避免极短窗的噪声主导 per-doc PPL。
    """
    nll_sum = 0.0
    n_tok = 0
    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start : start + seq_len + 1]
        if len(chunk) < max(min_tail, 2):
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        with torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=model.embed.weight.dtype,
        ):
            logits = model(x, n_loops=n_loops)  # (1, T, V)
        nll_sum += F.cross_entropy(
            logits[0].float(), y[0], reduction="sum"
        ).item()
        n_tok += y.shape[1]
    return nll_sum, n_tok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMythos SkyPile PPL 评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="指定 .pt 文件；不指定则从 --checkpoint_dir 取最新")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/home/ranhao/projects/OpenMythos/checkpoints_act",
                        help="checkpoints 目录（--checkpoint 未给时生效）")
    parser.add_argument("--encoder_model_id", type=str,
                        default="Langboat/mengzi-t5-base",
                        help="必须与训练时的分词器一致；自动解析为本地缓存快照")
    parser.add_argument("--dataset", type=str, default="swift/SkyPile-150B",
                        help="modelscope 流式数据集名")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--n_loops", type=int, default=8,
                        help="推理循环深度（与 test_infer.py 一致）")
    parser.add_argument("--n_docs", type=int, default=100,
                        help="抽样文本篇数")
    parser.add_argument("--buffer_docs", type=int, default=500,
                        help="流式读取的窗口大小，从中抽 n_docs 篇")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="PPL 切窗长度（与训练 seq_len 一致）")
    parser.add_argument("--seed", type=int, default=42,
                        help="reservoir 抽样随机种子")
    parser.add_argument("--output", type=str, default=None,
                        help="结果 JSON 保存路径；默认 runs/ppl_<ckpt>_loops<N>.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.error(f"找不到 checkpoint（dir={args.checkpoint_dir}）")
        return

    logger.info("=" * 50)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"数据集: {args.dataset} | n_docs: {args.n_docs} | "
                f"buffer_docs: {args.buffer_docs} | seq_len: {args.seq_len} | "
                f"n_loops: {args.n_loops} | device: {device}")
    logger.info("=" * 50)

    tokenizer_path = resolve_tokenizer_path(args.encoder_model_id)
    if tokenizer_path != args.encoder_model_id:
        logger.info(f"Tokenizer 使用本地快照: {tokenizer_path}")
    tokenizer = MythosTokenizer(model_id=tokenizer_path)
    model = load_inference_model(ckpt_path, device)

    texts = sample_texts(args.dataset, args.n_docs, args.buffer_docs, args.seed)

    total_nll = 0.0
    total_tokens = 0
    doc_ppls = []
    skipped = 0
    for i, text in enumerate(texts):
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            skipped += 1
            continue
        nll, n_tok = doc_nll(model, ids, device, args.n_loops, args.seq_len)
        if n_tok == 0:
            skipped += 1
            continue
        doc_ppls.append(math.exp(nll / n_tok))
        total_nll += nll
        total_tokens += n_tok
        if (i + 1) % 20 == 0 or i + 1 == len(texts):
            logger.info(f"[{i + 1}/{len(texts)}] 累计 {total_tokens:,} token, "
                        f"当前 overall PPL={math.exp(total_nll / total_tokens):.4f}")

    overall_ppl = math.exp(total_nll / total_tokens) if total_tokens else float("nan")
    mean_ppl = statistics.mean(doc_ppls) if doc_ppls else float("nan")
    median_ppl = statistics.median(doc_ppls) if doc_ppls else float("nan")

    print("\n" + "=" * 50)
    print(f"PPL 评测完成 | checkpoint: {os.path.basename(ckpt_path)} | "
          f"n_loops={args.n_loops} | seq_len={args.seq_len}")
    print(f"  overall PPL (token 加权): {overall_ppl:.4f}  "
          f"({len(doc_ppls)} 篇, {total_tokens:,} token)")
    print(f"  per-doc PPL: mean={mean_ppl:.4f}  median={median_ppl:.4f}")
    print(f"  skipped: {skipped}")
    print("=" * 50)

    out_path = args.output or os.path.join(
        "runs",
        f"ppl_{os.path.splitext(os.path.basename(ckpt_path))[0]}"
        f"_loops{args.n_loops}.json",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "dataset": args.dataset,
                "n_loops": args.n_loops,
                "seq_len": args.seq_len,
                "n_docs": len(doc_ppls),
                "buffer_docs": args.buffer_docs,
                "seed": args.seed,
                "overall_ppl": overall_ppl,
                "mean_doc_ppl": mean_ppl,
                "median_doc_ppl": median_ppl,
                "total_tokens": total_tokens,
                "doc_ppls": doc_ppls,
            },
            f, ensure_ascii=False, indent=2,
        )
    logger.success(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
