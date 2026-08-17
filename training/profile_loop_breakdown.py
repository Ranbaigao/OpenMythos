#!/usr/bin/env python3
"""逐循环开销解剖 — 回答"写 Triton 值不值":

每个 recurrent 循环 ~22ms 里, attention / MoE-FFN / 其他逐元素算子 /
kernel 启动开销(wall − kernel) 各占多少。

    python training/profile_loop_breakdown.py
"""

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
DEVICE = "cuda"
TOKENIZER_ID = "Langboat/mengzi-t5-base"
N_LOOPS = 8


class ModTimer:
    """CUDA-event 计时器, 包住一个子模块的 forward (每次调用后 sync,
    串行化换来干净的归属; 只用于相对比例)。"""

    def __init__(self, mod):
        self.total_ms = 0.0
        self.calls = 0
        self._orig = mod.forward
        mod.forward = self._wrap

    def _wrap(self, *a, **k):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = self._orig(*a, **k)
        e.record()
        torch.cuda.synchronize()
        self.total_ms += s.elapsed_time(e)
        self.calls += 1
        return out


def main():
    torch.manual_seed(42)
    encoding = MythosTokenizer(TOKENIZER_ID)
    cfg = mythos_0_1b()
    cfg.vocab_size = encoding.vocab_size
    cfg.max_seq_len = SEQ_LEN
    cfg.attn_type = "gqa"
    model = OpenMythos(cfg).to(DEVICE)
    model.train()
    # 强制永不停机: 每次前向都跑满 N_LOOPS, 计时口径一致
    torch.nn.init.constant_(model.recurrent.act.halt.bias, -20.0)

    vocab = encoding.vocab_size
    x = torch.randint(0, vocab, (1, SEQ_LEN), device=DEVICE)
    y = torch.randint(0, vocab, (1, SEQ_LEN), device=DEVICE)

    def fwd():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, n_loops=N_LOOPS)
            return torch.nn.functional.cross_entropy(
                logits.view(-1, vocab), y.view(-1))

    # warmup
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        fwd().backward()
    torch.cuda.synchronize()

    # ---- 1) 模块级 fwd 分解 -------------------------------------------------
    attn_t = ModTimer(model.recurrent.block.attn)
    ffn_t = ModTimer(model.recurrent.block.ffn)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = fwd()
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1e3
    loops = attn_t.calls
    attn_ms, ffn_ms = attn_t.total_ms, ffn_t.total_ms
    other_ms = fwd_ms - attn_ms - ffn_ms
    print(f"[fwd] 一次前向 {fwd_ms:.1f} ms / {loops} 循环 = {fwd_ms/loops:.2f} ms/循环")
    print(f"  attention : {attn_ms/loops:.2f} ms/循环 ({attn_ms/fwd_ms:.0%})")
    print(f"  MoE-FFN   : {ffn_ms/loops:.2f} ms/循环 ({ffn_ms/fwd_ms:.0%})")
    print(f"  其他(norm/injection/lora/halt/embedding+sync损耗): "
          f"{other_ms/loops:.2f} ms/循环 ({other_ms/fwd_ms:.0%})")

    # ---- 2) fwd+bwd 全步 profiler: kernel 总时长 vs wall -------------------
    model.zero_grad(set_to_none=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fwd().backward()
        torch.cuda.synchronize()
    evts = prof.key_averages()
    kernel_us = sum(e.self_device_time_total for e in evts)
    cpu_us = sum(e.self_cpu_time_total for e in evts)
    n_kernels = sum(e.count for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA)
    print(f"\n[fwd+bwd] CUDA kernel 净耗时 {kernel_us/1e3:.1f} ms | "
          f"CPU 算子耗时 {cpu_us/1e3:.1f} ms | kernel 启动次数 ~{n_kernels}")
    print(f"  => 每循环 kernel 启动 ~{n_kernels/N_LOOPS:.0f} 次; "
          f"CPU 耗时/CUDA 耗时 = {cpu_us/max(kernel_us,1):.2f} "
          f"(>1.5 说明启动/调度开销大, CUDA Graph 收益 > 重写 kernel)")
    print("\n[fwd+bwd] Top-12 CUDA kernel (按净耗时):")
    print(prof.key_averages().table(sort_by="self_device_time_total",
                                    row_limit=12, max_name_column_width=60))


if __name__ == "__main__":
    main()
