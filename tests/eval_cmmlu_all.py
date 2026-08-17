#!/usr/bin/env python3
"""
对 checkpoints_act/ 下全部 checkpoint 批量跑 CMMLU 评测并画 acc-step 曲线。

复用 tests/eval_cmmlu.py（以子进程方式调用，每个 checkpoint 一个进程，
模型显存随进程退出彻底释放，避免同进程反复加载大模型的碎片问题）。

已有结果的 checkpoint 会直接读 runs/ 下的 JSON 跳过不重跑（可断点续跑）；
全部评测完成后汇总画 micro/macro acc 随 step 变化的曲线。

用法：
    python tests/eval_cmmlu_all.py                      # 全部 ckpt, 5-shot, n_loops=4
    python tests/eval_cmmlu_all.py --n_loops 8
    python tests/eval_cmmlu_all.py --max_per_subject 50 # 快速粗评
    python tests/eval_cmmlu_all.py --plot_only          # 不评测, 只用已有 JSON 画图
"""

import os
import re
import sys
import glob
import json
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 checkpoints 目录下全部 .pt 批量 CMMLU 评测并画图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_act")
    parser.add_argument("--n_loops", type=int, default=4)
    parser.add_argument("--n_shots", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_per_subject", type=int, default=0,
                        help="每科最多评多少题（0 = 全部），用于快速粗评")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--plot_only", action="store_true",
                        help="跳过评测，只用已有结果 JSON 画图")
    return parser.parse_args()


def result_path(runs_dir: str, ckpt_path: str, n_shots: int,
                n_loops: int, temperature: float) -> str:
    """与 tests/eval_cmmlu.py 的默认输出命名保持一致。"""
    name = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join(
        runs_dir, f"cmmlu_{name}_{n_shots}shot_loops{n_loops}_t{temperature}.json"
    )


def step_of(path: str) -> int:
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    args = parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")),
                   key=step_of)
    if not ckpts:
        print(f"在 {args.checkpoint_dir}/ 下没找到 .pt 文件")
        return

    results = {}  # step -> dict
    for ckpt in ckpts:
        step = step_of(ckpt)
        out = result_path(args.runs_dir, ckpt, args.n_shots,
                          args.n_loops, args.temperature)
        if os.path.exists(out):
            print(f"[skip] {os.path.basename(ckpt)} 已有结果 → {out}")
        elif args.plot_only:
            print(f"[miss] {os.path.basename(ckpt)} 无结果, --plot_only 模式跳过")
            continue
        else:
            cmd = [
                sys.executable, os.path.join("tests", "eval_cmmlu.py"),
                "--checkpoint", ckpt,
                "--n_loops", str(args.n_loops),
                "--n_shots", str(args.n_shots),
                "--temperature", str(args.temperature),
                "--output", out,
            ]
            if args.max_per_subject > 0:
                cmd += ["--max_per_subject", str(args.max_per_subject)]
            print(f"[eval] {os.path.basename(ckpt)} ...")
            ret = subprocess.run(cmd).returncode
            if ret != 0 or not os.path.exists(out):
                print(f"[fail] {os.path.basename(ckpt)} 评测失败(returncode={ret})，跳过")
                continue
        with open(out, encoding="utf-8") as f:
            results[step] = json.load(f)

    if not results:
        print("没有任何可用结果，无法画图")
        return

    steps = sorted(results)
    micro = [results[s]["micro_acc"] for s in steps]
    macro = [results[s]["macro_acc"] for s in steps]

    print("\n" + "=" * 50)
    print(f"CMMLU 汇总 | {args.n_shots}-shot | n_loops={args.n_loops}")
    for s, mi, ma in zip(steps, micro, macro):
        print(f"  step {s:>6}: micro={mi:.4f}  macro={ma:.4f}")
    print("=" * 50)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(steps, micro, "o-", label="micro acc (per-question)", lw=2)
    ax.plot(steps, macro, "s--", label="macro acc (per-subject)", lw=1.5, alpha=0.8)
    ax.axhline(0.25, color="gray", ls=":", lw=1, label="random baseline (25%)")
    for x, y in zip(steps, micro):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy")
    ax.set_title(f"CMMLU acc vs training step "
                 f"({args.n_shots}-shot, n_loops={args.n_loops})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    suffix = f"_max{args.max_per_subject}" if args.max_per_subject > 0 else ""
    out_png = os.path.join(
        args.runs_dir,
        f"cmmlu_curve_{args.n_shots}shot_loops{args.n_loops}{suffix}.png",
    )
    fig.savefig(out_png, dpi=150)
    print(f"曲线已保存 → {out_png}")


if __name__ == "__main__":
    main()
