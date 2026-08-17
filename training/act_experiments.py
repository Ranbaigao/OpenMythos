#!/usr/bin/env python3
"""
ACT 熵正则 / ponder cost 的原理验证实验(不改动 open_mythos/main.py 与训练脚本)。

通过 monkeypatch 把 RecurrentBlock.forward 替换为逐行复刻的插桩版本,
额外返回: 每步停机权重 W、停机 logit、逐位置停机步数 halt_step、
质量 m=Σw、ponder(期望循环数)、熵 H。

    # 实验1: 逐位置停机分布 + 质量守恒检验(fresh-init vs checkpoint)
    python training/act_experiments.py inspect --ckpt checkpoints/step_0002000.pt

    # 实验2: 梯度符号验证 — 任务损失是否把早期停机概率往下压(漂移力),
    #         ponder 梯度是否反向(抬早停机)
    python training/act_experiments.py grad --ckpt checkpoints/step_0002000.pt

    # 实验3: rescue 对照训练 — 从"已坍塌"的 checkpoint 出发,验证各正则能否
    #         把循环数拉回低位且损失不受损
    python training/act_experiments.py rescue --ckpt checkpoints/step_0002000.pt \
        --tau 1e-3 --beta 0 --tag ponder1e-3 --steps 120 --grad-accum 4

    # 实验4: 文档边界相关性 — 现训一个模型(正式配方), 追踪晚停机位置是否
    #         聚集在文章边界附近(判读: 聚集 → dataset 加 EOS 有望助早停)
    python training/act_experiments.py boundary --ckpt none \
        --train-steps 1500 --eval-every 250 --batches 16
"""

import argparse
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mythos import OpenMythos
from open_mythos.main import loop_index_embedding
from open_mythos.variants import mythos_0_1b
from open_mythos.tokenizer import MythosTokenizer

SEQ_LEN = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER_ID = "Langboat/mengzi-t5-base"


# ---------------------------------------------------------------------------
# 插桩版 RecurrentBlock.forward — 逐行复刻 main.py:848-914, 附带内部量
# ---------------------------------------------------------------------------
def instrumented_recurrent_forward(rb, h, e, freqs_cis, mask, n_loops=None, kv_cache=None):
    cfg = rb.cfg
    n_loops = n_loops or cfg.max_loop_iters
    B, T, D = h.shape
    # 实验开关: _force_loops=True 禁止批次提前退出(逐循环 loss 分析用),
    #           _store_loops=True 记录每轮循环后的隐状态 h
    force = getattr(rb, "_force_loops", False)
    store = getattr(rb, "_store_loops", False)

    halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
    cumulative_p = torch.zeros(B, T, device=h.device)
    h_out = torch.zeros_like(h)
    halt_step = torch.full((B, T), n_loops, device=h.device, dtype=torch.long)

    weights, logits = [], []
    h_list = []
    ran = 0
    for t in range(n_loops):
        ran = t + 1
        h_loop = loop_index_embedding(h, t, rb.loop_dim)
        combined = rb.norm(h_loop + e)
        trans_out = rb.block(combined, freqs_cis, mask, kv_cache, f"recurrent_loop_{t}")
        trans_out = trans_out + rb.lora(trans_out, t)
        h = rb.injection(h, e, trans_out)
        if store:
            h_list.append(h)

        logit = rb.act.halt(h).squeeze(-1)          # (B,T)  — 原实现 sigmoid 前取值
        p = torch.sigmoid(logit)
        still_running = ~halted

        remainder = (1.0 - cumulative_p).clamp(min=0)
        weight = torch.where(cumulative_p + p >= cfg.act_threshold, remainder, p)
        weight = weight * still_running.float()

        newly_halted = still_running & (cumulative_p + p >= cfg.act_threshold)
        halt_step = torch.where(newly_halted, torch.full_like(halt_step, t + 1), halt_step)

        h_out = h_out + weight.unsqueeze(-1) * h
        cumulative_p = cumulative_p + p * still_running.float()
        halted = halted | (cumulative_p >= cfg.act_threshold)

        weights.append(weight)
        logits.append(logit)
        if halted.all() and kv_cache is None and not force:
            break

    W = torch.stack(weights, 0)                       # (ran, B, T)
    mass = W.sum(0)                                   # (B,T)
    # 与 main.py 一致的修正: 预算耗尽未停机的位置, 剩余质量用最后一轮的 h 结算
    h_out = h_out + (1.0 - mass).clamp(min=0).unsqueeze(-1) * h
    idx = torch.arange(1, ran + 1, device=h.device, dtype=h.dtype).view(-1, 1, 1)
    # 期望循环数: 未停机的剩余质量视为跑满 n_loops
    ponder = (idx * W).sum(0) + n_loops * (1.0 - mass)
    # 分布熵: 把"未停机剩余质量"当作第 ran+1 个结局
    leftover = (1.0 - mass).clamp_min(1e-9)
    probs = torch.cat([W, leftover.unsqueeze(0)], 0).clamp_min(1e-9)
    entropy = -(probs * probs.log()).sum(0)

    rb._act_stats = {
        "ran": ran, "weights": W, "logits": torch.stack(logits, 0),
        "logits_list": logits,  # 未 stack 的原始张量列表, 在 loss 的计算图路径上
        "halt_step": halt_step, "mass": mass, "ponder": ponder, "entropy": entropy,
        "h_list": h_list,       # 仅 _store_loops=True 时非空
    }
    # main.py 的 RecurrentBlock.forward 现在返回 (h_out, act) 二元组, 保持一致
    return h_out, {"ponder": ponder, "loops": float(ran)}


def patch_model(model):
    rb = model.recurrent
    rb.forward = lambda h, e, freqs_cis, mask=None, n_loops=None, kv_cache=None: \
        instrumented_recurrent_forward(rb, h, e, freqs_cis, mask, n_loops, kv_cache)


def build_model(vocab_size, ckpt=None, reset_halt_head=False):
    torch.manual_seed(42)
    cfg = mythos_0_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = SEQ_LEN
    cfg.attn_type = "gqa"
    model = OpenMythos(cfg)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"  loaded {ckpt}")
    if reset_halt_head:
        # 打破饱和陷阱: 重置 ACT 停机头 → sigmoid≈0.5, 梯度信号恢复
        nn_init = torch.nn.init.normal_
        nn_init(model.recurrent.act.halt.weight, std=0.02)
        torch.nn.init.zeros_(model.recurrent.act.halt.bias)
        print("  halt head reset (weight~N(0,0.02), bias=0)")
    patch_model(model)
    return model.to(DEVICE)


def get_data_iter(grad_accum=4, num_workers=2):
    from torch.utils.data import DataLoader
    spec = importlib.util.spec_from_file_location(
        "train_mod",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "0.1b_fine_skywork.py"),
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    dataset = train_mod.SkyworkDataset(
        seq_len=SEQ_LEN, rank=0, world_size=1, start_step=0,
        grad_accum=grad_accum, micro_batch=1, tokenizer_model_id=TOKENIZER_ID,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, pin_memory=True)
    return iter(loader)


def fwd_task_loss(model, x, y, vocab_size, n_loops=None):
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x, n_loops=n_loops)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1))
    return loss


# ---------------------------------------------------------------------------
# 实验4: 文档边界相关性 — 晚停机位置是否聚集在文章边界附近?
# 判读 → 若在边界附近聚集: dataset 加 EOS 有望帮停机头识别"不可约难题", 助早停
#        若均匀散布: 边界不是晚停机的主因, 加 EOS 无济于事
# ---------------------------------------------------------------------------
def get_boundary_iter():
    """进程内复刻 SkyworkDataset 的拼接逻辑(0 worker), 并行记录每个 token 的
    (在文档内的位置 pos_in_doc, 文档总长 doc_len)。"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    from modelscope.msdatasets import MsDataset
    encoding = MythosTokenizer(TOKENIZER_ID)
    ds = MsDataset.load("swift/SkyPile-150B", split="train", use_streaming=True)
    buf, meta = [], []
    for sample in ds:
        text = sample.get("text", sample.get("content", ""))
        ids = encoding.encode(text)
        n = len(ids)
        if n == 0:
            continue
        buf.extend(ids)
        meta.extend([(i, n) for i in range(n)])
        while len(buf) >= SEQ_LEN + 1:
            chunk, mchunk = buf[: SEQ_LEN + 1], meta[: SEQ_LEN + 1]
            buf, meta = buf[SEQ_LEN + 1 :], meta[SEQ_LEN + 1 :]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            pos = torch.tensor([m[0] for m in mchunk[:-1]], dtype=torch.long)
            rem = torch.tensor([m[1] - 1 - m[0] for m in mchunk[:-1]], dtype=torch.long)
            yield x, y, pos, rem


def boundary_eval(model, data_iter, vocab_size, n_batches):
    """跑 n_batches 个带边界标记的前向, 返回 (halt_step, pos_in_doc, rem, rans)。"""
    was_training = model.training
    model.eval()
    hs_all, pos_all, rem_all, rans = [], [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _y, pos, rem = next(data_iter)
            x = x.unsqueeze(0).to(DEVICE)
            fwd_task_loss(model, x, x, vocab_size)
            st = model.recurrent._act_stats
            hs_all.append(st["halt_step"].flatten().cpu())
            pos_all.append(pos)
            rem_all.append(rem)
            rans.append(st["ran"])
    if was_training:
        model.train()
    return torch.cat(hs_all).float(), torch.cat(pos_all).float(), \
        torch.cat(rem_all).float(), rans


def boundary_summary(hs, pos, rem):
    """富集倍数摘要: P(边界|晚停机) / P(边界)。≈1 = 无关联, >>1 = 晚停机聚集边界。"""
    late = hs >= 6
    out = {"halt_mean": hs.mean().item(), "halt_p99": hs.quantile(0.99).item(),
           "late_frac": late.float().mean().item()}
    for name, m in [("rem0", rem == 0), ("rem<=3", rem <= 3), ("pos<=3", pos <= 3)]:
        base = m.float().mean()
        in_late = m[late].float().mean() if late.any() else torch.tensor(0.0)
        out[f"enrich_{name}"] = (in_late / base.clamp_min(1e-9)).item()
        out[f"base_{name}"] = base.item()
    return out


def boundary_report(hs, pos, rem, rans):
    n = hs.numel()
    print(f"\n[boundary] {n} 个位置 | batch 循环数: {rans}")
    print(f"[boundary] 停机步数: mean={hs.mean():.2f} p90={hs.quantile(0.9):.0f} "
          f"p99={hs.quantile(0.99):.0f} max={hs.max():.0f}")
    print(f"[boundary] 文档长度估计: pos_in_doc p50={pos.median():.0f} "
          f"p90={pos.quantile(0.9):.0f} (边界密度 ~ 1/平均文档长)")

    print("\n[boundary] 距文档开头 pos_in_doc 分桶 → 平均停机步数:")
    for lo, hi in [(0, 0), (1, 3), (4, 15), (16, 63), (64, 255), (256, 1 << 30)]:
        m = (pos >= lo) & (pos <= hi)
        if m.any():
            print(f"  pos {lo:>3}..{hi if hi < 1 << 30 else '∞':>3}: "
                  f"halt mean={hs[m].mean():.2f} | 占比 {m.float().mean():.1%}")

    print("\n[boundary] 距文档结尾 rem 分桶 → 平均停机步数 (rem=0 即预测下一篇开头):")
    for lo, hi in [(0, 0), (1, 3), (4, 15), (16, 1 << 30)]:
        m = (rem >= lo) & (rem <= hi)
        if m.any():
            print(f"  rem {lo:>3}..{hi if hi < 1 << 30 else '∞':>3}: "
                  f"halt mean={hs[m].mean():.2f} | 占比 {m.float().mean():.1%}")

    late = hs >= 6
    print(f"\n[boundary] 晚停机位置 (halt≥6) 占比 {late.float().mean():.1%}:")
    for name, m in [("rem==0 (文档最后一个 token)", rem == 0),
                    ("rem<=3 (靠近结尾)", rem <= 3),
                    ("pos<=3 (靠近开头)", pos <= 3),
                    ("pos<=15", pos <= 15),
                    ("边界附近 (rem<=3 或 pos<=3)", (rem <= 3) | (pos <= 3))]:
        base = m.float().mean()
        in_late = m[late].float().mean() if late.any() else torch.tensor(0.0)
        print(f"  {name}: 全局占比 {base:.2%} | 晚停机位置中占比 {in_late:.2%} "
              f"| 富集倍数 {(in_late / base.clamp_min(1e-9)):.1f}x")


def improvement_eval(model, data_iter, vocab_size, n_batches):
    """逐位置 × 逐循环 loss 曲线 (强制跑满 max_loop_iters, 不提前退出)。
    回答: 边界位置是否"loss 高且多循环不改善"(= 应当早停的地面真值)。"""
    rb = model.recurrent
    rb._force_loops = True
    rb._store_loops = True
    was_training = model.training
    model.eval()
    losses, pos_all, rem_all = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y, pos, rem = next(data_iter)
            x, y = x.unsqueeze(0).to(DEVICE), y.unsqueeze(0).to(DEVICE)
            fwd_task_loss(model, x, y, vocab_size)
            h_list = model.recurrent._act_stats["h_list"]
            per_loop = []
            for h_t in h_list:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits_t = model.head(model.norm(h_t))
                l_t = torch.nn.functional.cross_entropy(
                    logits_t.float().view(-1, vocab_size), y.view(-1),
                    reduction="none")
                per_loop.append(l_t.float().cpu())
            losses.append(torch.stack(per_loop, 0))     # (n_loops, T)
            pos_all.append(pos)
            rem_all.append(rem)
    rb._force_loops = False
    rb._store_loops = False
    if was_training:
        model.train()
    L = torch.cat(losses, 1)                            # (n_loops, N)
    return L, torch.cat(pos_all).float(), torch.cat(rem_all).float()


def improvement_report(L, pos, rem):
    n_loops = L.shape[0]
    steps_shown = sorted({0, 1, 2, 4, n_loops - 1})
    print(f"\n[improve] {L.shape[1]} 个位置 × {n_loops} 次强制循环, 逐位置 loss:")
    hdr = "  位置桶            | 占比   | " + " | ".join(
        f"loop{t + 1:>2}" for t in steps_shown) + " | 改善(1→末)"
    print(hdr)
    buckets = [("rem==0 (跨界预测)", rem == 0), ("rem 1..3", (rem >= 1) & (rem <= 3)),
               ("rem 4..15", (rem >= 4) & (rem <= 15)), ("rem ≥16 (文档内部)", rem >= 16),
               ("pos<=3 (文档开头)", pos <= 3), ("pos 4..63", (pos >= 4) & (pos <= 63))]
    for name, m in buckets:
        if not m.any():
            continue
        row = " | ".join(f"{L[t][m].mean():.3f}" for t in steps_shown)
        gain = (L[0][m] - L[-1][m]).mean()
        print(f"  {name:<17} | {m.float().mean():>5.2%} | {row} | {gain:+.3f}")
    print("[improve] 判读: 边界位置若 loss 高但改善≈0 → 多循环对它们是浪费,"
          " 理想停机头应在这些位置早退; EOS 能把'不可约难'标记出来帮它学会这点")


def exp_boundary(args):
    encoding = MythosTokenizer(TOKENIZER_ID)
    eos_id = getattr(encoding.tokenizer, "eos_token_id", None)
    print(f"[boundary] tokenizer eos_token_id = {eos_id} "
          f"(在词表内: {eos_id is not None and eos_id < encoding.vocab_size})")
    ckpt = None if args.ckpt.lower() in ("none", "") else args.ckpt
    model = build_model(encoding.vocab_size, ckpt)
    eval_iter = get_boundary_iter()

    if args.train_steps > 0:
        # 用与正式训练相同的配方现训一个模型, 追踪"晚停机↔边界"相关性随训练量
        # 的演化 (step 0 = 随机停机头基线, 富集应 ≈1)
        model.train()
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        other_params = [p for p in model.parameters() if p.ndim != 2]
        # lr 对齐正式 run 逃逸点 (step 50-60, warmup ~5% 峰值处):
        # 满血 lr 会把停机头直接打进饱和区 (前两轮实验 pond 被钉死在 1.0)
        optim_adamw = torch.optim.AdamW(other_params, lr=1.5e-5, weight_decay=0.1,
                                        betas=(0.9, 0.95), fused=True)
        optim_muon = torch.optim.Muon(muon_params, lr=1e-3, momentum=0.95)
        train_iter = get_data_iter(args.grad_accum)
        os.makedirs("runs", exist_ok=True)
        traj = open("runs/act_boundary_trajectory.jsonl", "a")
        print(f"[boundary] 先训练 {args.train_steps} 步 (配方同正式训练: "
              f"τ=1e-2, B=3.0, grad_accum={args.grad_accum}), "
              f"每 {args.eval_every} 步评估一次边界相关性")
        for step in range(args.train_steps + 1):
            if step % args.eval_every == 0:
                hs, pos, rem, rans = boundary_eval(model, eval_iter,
                                                   encoding.vocab_size, 4)
                s = boundary_summary(hs, pos, rem)
                line = (f"  step {step:5d} | halt mean {s['halt_mean']:.2f} "
                        f"p99 {s['halt_p99']:.0f} | 晚停机 {s['late_frac']:.1%} | "
                        f"富集 rem0 {s['enrich_rem0']:.1f}x "
                        f"rem<=3 {s['enrich_rem<=3']:.1f}x "
                        f"pos<=3 {s['enrich_pos<=3']:.1f}x | batch loops {rans}")
                print(line, flush=True)
                traj.write(json.dumps({"step": step, **{k: round(v, 4) for k, v in s.items()},
                                       "batch_loops": rans}) + "\n")
                traj.flush()
            if step == args.train_steps:
                break
            optim_muon.zero_grad()
            optim_adamw.zero_grad()
            for _ in range(args.grad_accum):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = get_data_iter(args.grad_accum)
                    x, y = next(train_iter)
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = fwd_task_loss(model, x, y, encoding.vocab_size)
                st = model.recurrent._act_stats
                pen = torch.relu(st["ponder"].float() - 3.0).mean()
                ((loss + 1e-2 * pen) / args.grad_accum).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_muon.step()
            optim_adamw.step()
        # 训练结束立即落盘, 后续分析阶段即使被杀也保留模型供离线分析
        torch.save({"model": model.state_dict()}, "runs/act_boundary_model.pt")
        traj.close()
        print("[boundary] 现训模型已保存 → runs/act_boundary_model.pt", flush=True)

    hs, pos, rem, rans = boundary_eval(model, eval_iter, encoding.vocab_size,
                                       args.batches)
    boundary_report(hs, pos, rem, rans)

    # 机制验证: 逐位置×逐循环 loss 改善 vs 边界距离 (Ouro Stage-II 伪标签的真值)
    L, pos2, rem2 = improvement_eval(model, eval_iter, encoding.vocab_size,
                                     args.batches)
    improvement_report(L, pos2, rem2)


# ---------------------------------------------------------------------------
# 实验1: 逐位置停机分布 + 质量守恒
# ---------------------------------------------------------------------------
def exp_inspect(args):
    encoding = MythosTokenizer(TOKENIZER_ID)
    vocab_size = encoding.vocab_size
    data_iter = get_data_iter()

    for tag, ckpt in [("fresh-init", None), ("checkpoint", args.ckpt)]:
        if ckpt and not os.path.exists(ckpt):
            continue
        model = build_model(vocab_size, ckpt)
        model.eval()
        all_halt, all_mass, rans, ponders = [], [], [], []
        with torch.no_grad():
            for _ in range(args.batches):
                x, y = next(data_iter)
                x = x.to(DEVICE)
                fwd_task_loss(model, x, x, vocab_size)  # y 不用, 只要前向
                st = model.recurrent._act_stats
                all_halt.append(st["halt_step"].flatten().cpu())
                all_mass.append(st["mass"].flatten().cpu())
                rans.append(st["ran"])
                ponders.append(st["ponder"].mean().item())
        hs = torch.cat(all_halt).float()
        ms = torch.cat(all_mass)
        print(f"\n[inspect:{tag}] batch 循环数(逐批): {rans}")
        print(f"[inspect:{tag}] 逐位置停机步数: mean={hs.mean():.2f} "
              f"median={hs.median():.0f} p90={hs.quantile(0.9):.0f} "
              f"p99={hs.quantile(0.99):.0f} max={hs.max():.0f}")
        print(f"[inspect:{tag}] 停机步数直方图(1..8): "
              f"{torch.bincount(hs.long(), minlength=9)[1:].tolist()}")
        print(f"[inspect:{tag}] 质量 m=Σw: mean={ms.mean():.4f} min={ms.min():.4f} "
              f"| m<0.99 的位置占比 {(ms < 0.99).float().mean():.2%} (质量泄漏)")
        print(f"[inspect:{tag}] ponder(期望循环数) mean={sum(ponders)/len(ponders):.2f}")
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 实验2: 梯度符号验证
# ---------------------------------------------------------------------------
def exp_grad(args):
    encoding = MythosTokenizer(TOKENIZER_ID)
    vocab_size = encoding.vocab_size
    model = build_model(vocab_size, args.ckpt)
    model.train()
    data_iter = get_data_iter()

    # 对每步停机 logit ℓ_t 收集: ∂L_task/∂ℓ_t 与 ∂ponder/∂ℓ_t
    # 注意: 不用 retain_graph(峰值显存翻倍, 与正式训练并存时会触发 WSL 显存换页
    # 导致上百倍减速); 改为两次独立前向, 各自 backward 后即释放计算图
    g_task = torch.zeros(8)
    g_ponder = torch.zeros(8)
    n_batches = 0
    for i in range(args.batches):
        x, y = next(data_iter)
        x, y = x.to(DEVICE), y.to(DEVICE)

        model.zero_grad(set_to_none=True)
        loss = fwd_task_loss(model, x, y, vocab_size)
        st = model.recurrent._act_stats
        gt = torch.stack(torch.autograd.grad(loss, st["logits_list"]), 0).mean(dim=(1, 2))
        ran = st["ran"]

        model.zero_grad(set_to_none=True)
        loss2 = fwd_task_loss(model, x, y, vocab_size)
        st2 = model.recurrent._act_stats
        gp = torch.stack(
            torch.autograd.grad(st2["ponder"].float().mean(), st2["logits_list"]), 0
        ).mean(dim=(1, 2))

        g_task[:ran] += gt.float().cpu()
        g_ponder[:ran] += gp.float().cpu()
        n_batches += 1
        print(f"[grad] batch {i + 1}/{args.batches} done (ran={ran})", flush=True)

    g_task /= n_batches
    g_ponder /= n_batches
    print(f"\n[grad] 对每步停机 logit ℓ_t 的平均梯度 (checkpoint, {n_batches} batches):")
    print(f"  t          : {[t + 1 for t in range(8)]}")
    print(f"  ∂L_task/∂ℓ : {[round(v, 5) for v in g_task.tolist()]}")
    print(f"  ∂ponder/∂ℓ : {[round(v, 5) for v in g_ponder.tolist()]}")
    print("[grad] 判读: ℓ 增大=p(停机)增大=更早退出。")
    print("  ∂L_task/∂ℓ > 0 → 梯度下降压低 ℓ → 更晚退出 = 漂移力指向更深(验证 H1)")
    print("  ∂ponder/∂ℓ < 0 → 加入 τ·ponder 后抬高 ℓ → 更早退出 = 反向矫正(验证 H2)")


# ---------------------------------------------------------------------------
# 实验3: rescue 对照训练
# ---------------------------------------------------------------------------
def exp_rescue(args):
    encoding = MythosTokenizer(TOKENIZER_ID)
    vocab_size = encoding.vocab_size
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpt = None if args.ckpt.lower() in ("none", "") else args.ckpt
    model = build_model(vocab_size, ckpt, reset_halt_head=args.reset_halt_head)
    model.train()

    muon_params = [p for p in model.parameters() if p.ndim == 2]
    other_params = [p for p in model.parameters() if p.ndim != 2]
    optim_adamw = torch.optim.AdamW(other_params, lr=args.adam_lr, weight_decay=0.1,
                                    betas=(0.9, 0.95), fused=True)
    optim_muon = torch.optim.Muon(muon_params, lr=args.muon_lr, momentum=0.95)

    data_iter = get_data_iter(args.grad_accum)
    out_path = f"runs/act_rescue_{args.tag}.jsonl"
    os.makedirs("runs", exist_ok=True)
    fout = open(out_path, "a")

    n_loops = args.max_loops or None
    print(f"[rescue:{args.tag}] τ={args.tau} β={args.beta} steps={args.steps} "
          f"grad_accum={args.grad_accum} max_loops={n_loops or 'cfg'} → {out_path}")
    for step in range(args.steps):
        t0 = time.perf_counter()
        optim_muon.zero_grad()
        optim_adamw.zero_grad()
        task_tot = ponder_tot = 0.0
        ran_max = 0
        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = get_data_iter(args.grad_accum)
                x, y = next(data_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = fwd_task_loss(model, x, y, vocab_size, n_loops=n_loops)
            st = model.recurrent._act_stats
            ponder_pos = st["ponder"].float()
            ponder = ponder_pos.mean()
            ent = st["entropy"].float().mean()
            # 逐位置预算式惩罚(与正式脚本一致): 低于 budget 时零梯度,
            # 避免把循环数压塌到 1
            pen = torch.relu(ponder_pos - args.budget).mean() \
                if args.budget > 0 else ponder
            total = (loss + args.tau * pen - args.beta * ent) / args.grad_accum
            total.backward()
            task_tot += loss.item() / args.grad_accum
            ponder_tot += ponder.item() / args.grad_accum
            ran_max = max(ran_max, st["ran"])

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim_muon.step()
        optim_adamw.step()
        dt = time.perf_counter() - t0

        rec = {"step": step, "task_loss": round(task_tot, 4),
               "ponder": round(ponder_tot, 3), "batch_loops": ran_max,
               "sec": round(dt, 2)}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:4d} | task_loss {task_tot:.4f} | "
                  f"ponder {ponder_tot:.2f} | batch_loops {ran_max} | {dt:.1f}s")
    fout.close()
    print(f"[rescue:{args.tag}] done → {out_path}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["inspect", "grad", "rescue", "boundary"])
    p.add_argument("--ckpt", default="checkpoints/step_0002000.pt")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.0, help="ponder cost 系数")
    p.add_argument("--budget", type=float, default=0.0,
                   help=">0 时改用预算式惩罚 τ·ReLU(ponder-budget)")
    p.add_argument("--beta", type=float, default=0.0, help="熵正则系数")
    p.add_argument("--tag", default="run")
    p.add_argument("--reset-halt-head", action="store_true",
                   help="加载 checkpoint 后重置 ACT 停机头, 打破饱和陷阱")
    p.add_argument("--max-loops", type=int, default=0,
                   help="rescue 模式: 强制 T_max (0 = 用 cfg 默认), 用于砍预算 A/B")
    p.add_argument("--adam-lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--train-steps", type=int, default=0,
                   help="boundary 模式: 先用正式配方现训 N 步再做相关分析")
    p.add_argument("--eval-every", type=int, default=250,
                   help="boundary 模式: 现训期间每隔多少步评估一次边界相关性")
    args = p.parse_args()

    {"inspect": exp_inspect, "grad": exp_grad, "rescue": exp_rescue,
     "boundary": exp_boundary}[args.mode](args)


if __name__ == "__main__":
    main()
