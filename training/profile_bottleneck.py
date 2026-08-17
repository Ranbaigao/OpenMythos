#!/usr/bin/env python3
"""
训练变慢瓶颈诊断脚本 — 配合 training/0.1b_fine_skywork.py 使用。

背景:训练吞吐从 ~16k tok/s 单调跌到 ~4.3k tok/s(step_seconds 16s → 60s)。
本脚本通过三组对照实验定位瓶颈,不改动训练脚本本身:

    python training/profile_bottleneck.py data   --n-batches 300
    python training/profile_bottleneck.py model  --ckpt checkpoints/step_0002000.pt
    python training/profile_bottleneck.py e2e    --steps 3 --grad-accum 16

实验 A (data):  纯数据管线测速(不占 GPU)。用与训练完全相同的
                SkyworkDataset + DataLoader(num_workers=4) 连续消费 N 个
                micro-batch,分段统计吞吐。若 samples/s × step_seconds < 256
                (grad_accum),说明 GPU 在等数据。

实验 B (model): 纯模型算力测试(合成数据,排除数据管线干扰)。
                1) 强制 n_loops=1..8 扫描 → 每个 recurrent 循环的成本;
                2) 随机初始化模型 + 自由 ACT → 模拟"训练初期"的实际循环数/耗时;
                3) 加载 checkpoint + 自由 ACT → 模拟"训练后期"的实际循环数/耗时。
                若 (3) 的循环数与耗时显著大于 (2),则确认 ACT halting 头
                随训练漂移(无 ponder cost 惩罚)是变慢根因。

实验 C (e2e):   端到端真实训练若干步,逐步分解 data_wait / fwd_bwd /
                optim 三段耗时占比 + 每步实际 ACT 循环数。

注意:实验期间正式训练仍在跑,GPU 有竞争,绝对值偏高,但组间相对比较有效。
"""

import argparse
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mythos import OpenMythos
from open_mythos.variants import mythos_0_1b
from open_mythos.tokenizer import MythosTokenizer

SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 与训练脚本一致(0.1b_fine_skywork.py:585,744)
TOKENIZER_ID = "Langboat/mengzi-t5-base"


# ---------------------------------------------------------------------------
# 工具: ACT 实际循环数统计
# ---------------------------------------------------------------------------
class LoopCounter:
    """
    通过 ACTHalting 的 forward hook 统计 recurrent block 的实际循环次数。

    RecurrentBlock 每个 loop iteration 恰好调用一次 self.act(h),
    因此一次模型前向中 hook 触发次数 == 实际执行的循环数。
    不修改 open_mythos/main.py 的任何代码。
    """

    def __init__(self, model: OpenMythos):
        self.count = 0
        self._handle = model.recurrent.act.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.count += 1

    def reset(self):
        self.count = 0

    def close(self):
        self._handle.remove()


def build_model(vocab_size: int) -> OpenMythos:
    """与训练脚本相同的 cfg 与种子(0.1b_fine_skywork.py:630-635)。"""
    torch.manual_seed(42)
    cfg = mythos_0_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = SEQ_LEN
    cfg.attn_type = "gqa"
    return OpenMythos(cfg)


def load_model_weights(model: OpenMythos, ckpt_path: str) -> None:
    """加载 checkpoint 的模型权重;兼容 torch.compile 的 `_orig_mod.` 前缀。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded {ckpt_path} (step {ckpt['step']}) "
          f"| missing={len(missing)} unexpected={len(unexpected)}")


# ---------------------------------------------------------------------------
# 实验 A: 纯数据管线吞吐
# ---------------------------------------------------------------------------
def exp_data(args):
    from torch.utils.data import DataLoader
    import importlib.util

    # 直接加载训练脚本模块以复用 SkyworkDataset(避免复制代码)
    spec = importlib.util.spec_from_file_location(
        "train_mod",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "0.1b_fine_skywork.py"),
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    dataset = train_mod.SkyworkDataset(
        seq_len=SEQ_LEN,
        rank=0,
        world_size=1,
        start_step=0,
        grad_accum=256,
        micro_batch=1,
        tokenizer_model_id=TOKENIZER_ID,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    print(f"[data] consuming {args.n_batches} micro-batches from the real stream ...")
    times = []
    t_start = time.perf_counter()
    it = iter(loader)
    for i in range(args.n_batches):
        t0 = time.perf_counter()
        next(it)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 50 == 0:
            print(f"  batch {i + 1:4d}/{args.n_batches}  "
                  f"avg wait {sum(times[-50:]) / 50 * 1e3:8.1f} ms/sample")
    total = time.perf_counter() - t_start

    n = args.n_batches
    seg = max(1, n // 3)
    head = sum(times[:seg]) / seg
    tail = sum(times[-seg:]) / seg
    samples_s = n / total
    tokens_s = samples_s * SEQ_LEN
    # 训练每步需要 grad_accum=256 个样本;若供给 < 需求,GPU 空转
    need_per_step = 256
    max_step_rate = samples_s / need_per_step

    print(f"\n[data] total {total:.1f}s | {samples_s:.2f} samples/s | {tokens_s:,.0f} tok/s (data side)")
    print(f"[data] wait per sample: first-3rd {head * 1e3:.1f} ms  vs  last-3rd {tail * 1e3:.1f} ms")
    print(f"[data] => 数据管线最多支撑 {max_step_rate:.2f} step/s "
          f"(即 1 step 最少 {need_per_step / samples_s:.1f}s)")
    print(f"[data] 对照:训练初期 16s/step(需要 16 samples/s),"
          f"当前 60s/step(需要 4.3 samples/s)")
    if tail > head * 1.5:
        print("[data] ⚠️ 数据管线自身在变慢(后段显著慢于前段)")
    else:
        print("[data] ✓ 数据管线速度基本稳定,无随时间退化趋势")


# ---------------------------------------------------------------------------
# 实验 B: 纯模型算力(合成数据)
# ---------------------------------------------------------------------------
def _bench_fwd_bwd(model, x, y, vocab_size, iters, n_loops=None, counter=None):
    """测 iters 次 fwd+bwd 的平均耗时,返回 (ms/iter, avg_loops)。"""
    model.train()
    loop_counts = []
    # warmup
    for _ in range(2):
        if counter:
            counter.reset()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, n_loops=n_loops)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        if counter:
            loop_counts.append(counter.count)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        if counter:
            counter.reset()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, n_loops=n_loops)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        if counter:
            loop_counts.append(counter.count)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    avg_loops = sum(loop_counts) / len(loop_counts) if loop_counts else float(n_loops or 0)
    return dt, avg_loops


def exp_model(args):
    encoding = MythosTokenizer(TOKENIZER_ID)
    vocab_size = encoding.vocab_size
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_model(vocab_size).to(DEVICE)
    counter = LoopCounter(model)

    g = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.randint(0, vocab_size, (1, SEQ_LEN), generator=g, device=DEVICE)
    y = torch.randint(0, vocab_size, (1, SEQ_LEN), generator=g, device=DEVICE)

    print(f"[model] vocab={vocab_size:,} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"[model] ⚠️ 正式训练仍在跑,GPU 有竞争,看相对值不看绝对值\n")

    # --- B1: 强制循环数扫描 → 每循环成本 ---
    print("[model] B1) 强制 n_loops 扫描(fwd+bwd, bf16 autocast):")
    sweep = {}
    for n in range(1, args.max_loops + 1):
        dt, _ = _bench_fwd_bwd(model, x, y, vocab_size, args.iters, n_loops=n)
        sweep[n] = dt
        print(f"  n_loops={n}: {dt:7.1f} ms")
    per_loop = (sweep[args.max_loops] - sweep[1]) / (args.max_loops - 1)
    print(f"  => 固定开销(prelude+coda+head) ≈ {sweep[1] - per_loop:.1f} ms, "
          f"每个 recurrent 循环 ≈ {per_loop:.1f} ms\n")

    # --- B2: 随机初始化 + 自由 ACT(模拟训练初期)---
    dt_init, loops_init = _bench_fwd_bwd(model, x, y, vocab_size, args.iters,
                                         n_loops=None, counter=counter)
    print(f"[model] B2) 随机初始化(模拟训练初期): "
          f"实际循环 {loops_init:.1f} 次, {dt_init:.1f} ms/fwd+bwd")

    # --- B3: 加载 checkpoint + 自由 ACT(模拟训练后期)---
    if args.ckpt and os.path.exists(args.ckpt):
        load_model_weights(model, args.ckpt)
        dt_ckpt, loops_ckpt = _bench_fwd_bwd(model, x, y, vocab_size, args.iters,
                                             n_loops=None, counter=counter)
        print(f"[model] B3) 加载 checkpoint(模拟训练后期): "
              f"实际循环 {loops_ckpt:.1f} 次, {dt_ckpt:.1f} ms/fwd+bwd")
        ratio = dt_ckpt / dt_init
        print(f"\n[model] => 后期/初期耗时比 = {ratio:.2f}x "
              f"(循环数 {loops_init:.1f} → {loops_ckpt:.1f})")
        if loops_ckpt > loops_init + 0.5:
            print("[model] ⚠️ 确认:ACT halting 随训练推迟,循环数增加,"
                  "是训练变慢的主要(或重要)因素")
        else:
            print("[model] ✓ ACT 循环数没有随训练增加,变慢另有原因(查数据管线)")
    else:
        print(f"[model] B3) 跳过:checkpoint 不存在 ({args.ckpt})")

    counter.close()


# ---------------------------------------------------------------------------
# 实验 C: 端到端分阶段耗时分解
# ---------------------------------------------------------------------------
def exp_e2e(args):
    import importlib.util
    from torch.utils.data import DataLoader

    spec = importlib.util.spec_from_file_location(
        "train_mod",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "0.1b_fine_skywork.py"),
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    encoding = MythosTokenizer(TOKENIZER_ID)
    vocab_size = encoding.vocab_size
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_model(vocab_size).to(DEVICE)
    if args.ckpt and os.path.exists(args.ckpt):
        load_model_weights(model, args.ckpt)
    counter = LoopCounter(model)

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]
    optim_adamw = torch.optim.AdamW(other_params, lr=3e-4, weight_decay=0.1,
                                    betas=(0.9, 0.95), fused=True)
    optim_muon = torch.optim.Muon(muon_params, lr=0.02, momentum=0.95)

    dataset = train_mod.SkyworkDataset(
        seq_len=SEQ_LEN, rank=0, world_size=1, start_step=0,
        grad_accum=args.grad_accum, micro_batch=1,
        tokenizer_model_id=TOKENIZER_ID,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    data_iter = iter(loader)

    print(f"[e2e] {args.steps} steps × grad_accum={args.grad_accum} "
          f"(正式训练 grad_accum=256,此处缩小以便快速完成)\n")
    model.train()
    for step in range(args.steps):
        t_data = t_fwdbwd = t_optim = 0.0
        loops_total = 0
        optim_muon.zero_grad()
        optim_adamw.zero_grad()
        for micro in range(args.grad_accum):
            t0 = time.perf_counter()
            try:
                bx, by = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                bx, by = next(data_iter)
            t_data += time.perf_counter() - t0

            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)

            counter.reset()
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(bx)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), by.view(-1)) / args.grad_accum
            loss.backward()
            torch.cuda.synchronize()
            t_fwdbwd += time.perf_counter() - t0
            loops_total += counter.count

        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim_muon.step()
        optim_adamw.step()
        torch.cuda.synchronize()
        t_optim = time.perf_counter() - t0

        total = t_data + t_fwdbwd + t_optim
        print(f"[e2e] step {step}: total {total:5.1f}s | "
              f"data {t_data:5.1f}s ({t_data / total:5.1%}) | "
              f"fwd+bwd {t_fwdbwd:5.1f}s ({t_fwdbwd / total:5.1%}) | "
              f"optim {t_optim:4.1f}s ({t_optim / total:5.1%}) | "
              f"avg ACT loops {loops_total / args.grad_accum:.1f}")

    counter.close()
    print("\n[e2e] 判读: 占比最大的分段即为瓶颈; "
          "data 占比高→数据管线; fwd+bwd 占比高且 ACT loops 偏高→模型/ACT")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["data", "model", "e2e", "all"])
    p.add_argument("--n-batches", type=int, default=300, help="实验A消费的micro-batch数")
    p.add_argument("--iters", type=int, default=5, help="实验B每档重复次数")
    p.add_argument("--max-loops", type=int, default=8, help="实验B扫描的最大循环数")
    p.add_argument("--steps", type=int, default=3, help="实验C的步数")
    p.add_argument("--grad-accum", type=int, default=16, help="实验C的梯度累积(正式=256)")
    p.add_argument("--ckpt", default="checkpoints/step_0002000.pt")
    args = p.parse_args()

    if args.mode in ("data", "all"):
        exp_data(args)
    if args.mode in ("model", "all"):
        exp_model(args)
    if args.mode in ("e2e", "all"):
        exp_e2e(args)


if __name__ == "__main__":
    main()
