#!/usr/bin/env python3
"""
OpenMythos pretraining on FineWeb-Edu with FSDP + AdamW.

Single GPU:
    python training/3b_fine_web_edu.py

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/3b_fine_web_edu.py
"""

import os
import os
# ==============================================================================
# 🚀 必须放在所有 import 之前！否则 datasets 会在读取到之前就初始化为官方域名
# ==============================================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_ENDPOINT"] = "https://hf-mirror.com" # 增加这条保险

# 3. 提高 Datasets 远程流式读取的超时阈值（从默认10秒延长到120秒，防止网络抖动崩溃）
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. 核心黑科技：直接拦截并替换 huggingface_hub 内部的常量
import huggingface_hub
if hasattr(huggingface_hub, "constants"):
    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False # 禁用可能导致分片下载失败的 hf_transfer
    huggingface_hub.constants.HUGGINGFACE_CO_URL = "https://hf-mirror.com"
    huggingface_hub.constants.HF_HUB_HTTP_ENDPOINT = "https://hf-mirror.com"

import itertools

import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from contextlib import nullcontext

from datasets import load_dataset
from modelscope.msdatasets import MsDataset

from torch.utils.tensorboard import SummaryWriter

from open_mythos.main_fast_moe import OpenMythos
from open_mythos.main_fast_moe import TransformerBlock, RecurrentBlock, Expert, GroupedSwiGLUExperts
from open_mythos.variants import mythos_3b,mythos_0_1b
from open_mythos.tokenizer import MythosTokenizer
from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl,  # 新增：用于指定非重入式引擎
)



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True






# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SkyworkDataset(IterableDataset):
    """
    针对 BAAI/CCI3.0 的流式数据加载器，支持多卡 Sharding 与自动断点续训 (Auto-Resume)。
    已修复多进程 FileLock 死锁与 Tokenizer Fork 死锁。
    """

    def __init__(self, seq_len: int, rank: int, world_size: int, 
                 start_step: int = 0, grad_accum: int = 1, micro_batch: int = 1,
                 avg_tokens_per_doc: int = 1500,
                 max_docs_per_worker: int = None, skip_docs_per_worker: int = 0,
                 tokenizer_model_id: str = None):
        
        # =================================================================
        # 🚀 防御手段一：主进程预热 (解决 Worker 1 失踪/卡死问题)
        # =================================================================
        # 在多进程 fork 之前，由主进程先下载好 datasets 的 builder script 并写入缓存。
        # 这样随后的 Worker 进程就不会再争抢 FileLock 导致死锁。
        from datasets import load_dataset
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        load_dataset("Skywork/SkyPile-150B", split="train", streaming=True)
        # =================================================================
        
        # ⚠️ 注意：移除了 __init__ 中对 encoding 的接收，改为在内部实例化
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.start_step = start_step
        self.grad_accum = grad_accum
        self.micro_batch = micro_batch
        self.avg_tokens_per_doc = avg_tokens_per_doc
        self.max_docs_per_worker = max_docs_per_worker
        self.skip_docs_per_worker = skip_docs_per_worker
        # Must match the tokenizer used to size the model's Embedding layer
        # (main()'s `encoding`) — instantiating with the wrong model_id here
        # silently produces a different vocab, whose token IDs can exceed
        # the embedding's row count and crash CUDA with an out-of-bounds gather.
        self.tokenizer_model_id = tokenizer_model_id

    def __iter__(self):
        import os
        import huggingface_hub
        from datasets import load_dataset
        
        # 注入 Worker 进程的环境变量
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["CURL_CA_BUNDLE"] = ""      
        os.environ["REQUESTS_CA_BUNDLE"] = "" 
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
        os.environ["FSSPEC_HTTP_CLIENT_KWARGS"] = '{"timeout": 30}' # 强行防假死
        
        if hasattr(huggingface_hub, "constants"):
            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = False
            huggingface_hub.constants.HUGGINGFACE_CO_URL = "https://hf-mirror.com"
            huggingface_hub.constants.HF_HUB_HTTP_ENDPOINT = "https://hf-mirror.com"
        
        # =================================================================
        # 🚀 防御手段二：Worker 独立分词 (解决 Rust 多线程 Fork 死锁)
        # =================================================================
        from open_mythos.tokenizer import MythosTokenizer
        if self.tokenizer_model_id:
            self.encoding = MythosTokenizer(self.tokenizer_model_id)
        else:
            self.encoding = MythosTokenizer()

        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        # 此时调用，瞬间走本地缓存，不会再死锁！
        # =================================================================
        # 🚀 彻底抛弃不稳定的 HF 镜像，使用阿里云官方魔搭节点
        # =================================================================
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(
            "swift/SkyPile-150B",
            split="train",
            use_streaming=True,      # 魔搭的流式加载参数是 use_streaming
        )

        ds = ds.shard(num_shards=total_shards, index=shard_index)
        
        # 自动断点恢复逻辑...
        if self.start_step > 0:
            total_samples_for_rank = self.start_step * self.grad_accum * self.micro_batch
            samples_to_skip = total_samples_for_rank // num_workers
            if worker_id < total_samples_for_rank % num_workers:
                samples_to_skip += 1
            tokens_to_skip = samples_to_skip * self.seq_len
            auto_skip_docs = tokens_to_skip // self.avg_tokens_per_doc

            if auto_skip_docs > 0:
                ds = ds.skip(auto_skip_docs)
                if worker_id == 0:
                    logger.info(f"[Rank {self.rank} Worker 0] 自动断点恢复: 跳过了 {auto_skip_docs:,} 篇文章")

        if self.skip_docs_per_worker > 0:
            ds = ds.skip(self.skip_docs_per_worker)
        if self.max_docs_per_worker is not None:
            ds = ds.take(self.max_docs_per_worker)

        buf = []
        for sample in ds:
            text_content = sample.get("text", sample.get("content", ""))
            buf.extend(self.encoding.encode(text_content))
            
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )
                
class FineWebEduDataset(IterableDataset):
    """
    Streaming FineWeb-Edu loader yielding fixed-length (input, target) pairs.

    FineWeb-Edu is trillions of tokens, so `streaming=True` pulls shards on
    demand instead of materializing to disk. Sharding is two-dimensional —
    `world_size` ranks × `num_workers` DataLoader workers per rank — and each
    `(rank, worker_id)` deterministically owns one shard of the global stream.
    That gives disjoint coverage without any cross-process coordination.

    Streaming datasets are not seekable, so a resumed run re-enters its shard
    from the beginning. Acceptable at pretraining scale: the chance of
    re-playing the same tokens before the run ends is negligible versus the
    cost of a true resumable loader.
    """

    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        """
        Args:
            encoding   -- tokenizer exposing `.encode(str) -> list[int]`
            seq_len    -- context length; every yielded pair has this many tokens
            subset     -- FineWeb-Edu config name (e.g. "sample-10BT", "default")
            rank       -- global rank of this process within the distributed job
            world_size -- total number of distributed processes
        """
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        """
        Yield `(input_ids, target_ids)` tensors of length `seq_len` forever.

        Inputs and targets are shifted by one for next-token prediction —
        `target[i] == input[i + 1]`. Documents are concatenated into a rolling
        buffer and sliced into fixed-length chunks, packing short docs together
        and splitting long ones. This keeps every step at the same shape,
        which under FSDP avoids recompute from variable-length inputs and
        removes the need for a pad-aware attention mask.
        """
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        # ds = load_dataset(
        #     # "HuggingFaceFW/fineweb-edu",
        #     "Skywork/SkyPile-150B",
        #     # name=self.subset,
        #     split="train",
        #     streaming=True,
        # ).shard(num_shards=total_shards, index=shard_index)

        
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load(
            "BAAI/OpenSeek-Pretrain-100B",
            split="train",
            use_streaming=True,      # 魔搭的流式加载参数是 use_streaming
            trust_remote_code=True,  # MNBVC 是脚本型数据集，加载需执行仓库自带脚本
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup → half-cosine decay to `min_lr`.

    Standard language-model pretraining schedule. The warmup phase prevents
    Adam's second-moment estimate from collapsing to a huge LR in the first
    few steps when gradients are noisy. The cosine tail lets the model make
    small, increasingly conservative updates near the end of training rather
    than crashing to `min_lr` at a fixed step.

    Behavior by region:
        step < warmup                 → linear ramp 0 → max_lr
        warmup ≤ step < total         → cosine decay max_lr → min_lr
        step ≥ total                  → clamped at min_lr (safety for
                                        off-by-one step counters at the end
                                        of training)

    Args:
        step    -- current global optimizer step (0-indexed)
        warmup  -- number of warmup steps before cosine decay begins
        total   -- step at which the cosine reaches `min_lr`
        max_lr  -- peak learning rate reached at the end of warmup
        min_lr  -- floor learning rate at and after `total` steps

    Returns:
        Scalar learning rate for this step.
    """
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def fmt_eta(seconds: float) -> str:
    """
    Format a duration in seconds as a compact human-readable ETA.

    Picks the largest unit that holds a non-zero value and keeps the next
    unit for sub-day precision. Avoids printing huge second counts that are
    useless at a glance, and avoids the opposite trap of showing only `H:MM`
    which loses information once a run spans multiple days.

    Examples:
        45       → "45s"
        754      → "12m34s"
        5025     → "1h23m"
        192345   → "2d5h"
    """
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """
    Return checkpoint paths in `ckpt_dir` sorted oldest → newest.

    Relies on the zero-padded `step_{0000000}.pt` filename convention so
    lexicographic sort matches chronological order. Changing the filename
    format elsewhere without updating the pad width would silently break
    both `keep_last` pruning and resume-latest on startup, since both pick
    the last element of this list.

    Args:
        ckpt_dir -- directory to scan; missing directory returns []

    Returns:
        Sorted list of absolute paths to matching checkpoint files.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model,
    optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
) -> None:
    """
    Gather full model + optimizer state, write atomically, prune old files.

    Under FSDP both states are collected inside a single FULL_STATE_DICT
    context so the optim-state tensors bind to fully-unsharded parameters;
    mixing contexts between model and optimizer has caused silent divergence
    on resume in past torch versions. The temp-file + os.replace write means
    a kill mid-save leaves the previous checkpoint intact instead of a
    truncated .pt file. Non-master ranks participate in the FSDP gather
    (otherwise the collective would hang) but exit before touching disk.

    Args:
        model       -- FSDP-wrapped (ddp=True) or raw (ddp=False) model
        optimizer   -- the optimizer whose state should round-trip with the model
        step        -- global step number; encoded zero-padded into the filename
        cfg         -- model config object; saved so downstream eval can
                       reconstruct the model without re-importing the variant
        vocab_size  -- tokenizer vocab size at train time; saved for sanity-check
                       on load against a (possibly updated) tokenizer
        ckpt_dir    -- directory to write into; created if missing
        ddp         -- True if FSDP path; False for single-GPU / CPU
        master      -- whether this rank writes to disk (rank 0 only)
        keep_last   -- number of most-recent checkpoints to retain; older ones
                       are unlinked after a successful write

    Returns:
        None. Writes to disk as a side effect on master rank.
    """
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")

    logger.success(f"Checkpoint saved → {final_path}")

def load_checkpoint(model, optimizer, path: str, ddp: bool) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # ---------- 自动迁移旧 routed_experts → grouped_experts ----------
    state_dict = ckpt["model"]
    prefix = "recurrent.block.ffn.routed_experts."
    if any(k.startswith(prefix) for k in state_dict):
        logger.warning("检测到旧格式 MoE 权重，正在迁移为 grouped_experts ...")
        # 收集所有专家权重
        gate_list, up_list, down_list = [], [], []
        for i in itertools.count():
            g_key = f"{prefix}{i}.gate.weight"
            if g_key not in state_dict:
                break
            gate_list.append(state_dict.pop(g_key))
            up_list.append(state_dict.pop(f"{prefix}{i}.up.weight"))
            down_list.append(state_dict.pop(f"{prefix}{i}.down.weight"))

        # 堆叠成 grouped 格式
        new_prefix = "recurrent.block.ffn.grouped_experts."
        state_dict[f"{new_prefix}gate_weight"] = torch.stack(gate_list, dim=0)   # (E, expert_dim, dim)
        state_dict[f"{new_prefix}up_weight"]   = torch.stack(up_list, dim=0)
        state_dict[f"{new_prefix}down_weight"] = torch.stack(down_list, dim=0)   # (E, dim, expert_dim)

    # ---------- 加载模型 ----------
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(state_dict)
            # 优化器状态：旧格式已无法直接映射，重新初始化
            logger.warning("跳过旧优化器状态，从当前优化器初值开始训练")
    else:
        model.load_state_dict(state_dict)
        # 单卡同样忽略优化器状态，保持 optimizer 初始状态
        logger.warning("跳过旧优化器状态，从当前优化器初值开始训练")

    return int(ckpt["step"])
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    End-to-end pretraining entry point.

    Order matters: distributed init must run before any CUDA allocation, the
    tokenizer must exist before the model is built (vocab_size flows into
    cfg), and FSDP must wrap the model before the optimizer is constructed
    (FSDP re-flattens parameters, so an optimizer built on the unwrapped
    model would track stale param objects). Resume then loads state into the
    already-constructed optimizer in-place.

    Lifecycle:
        1. Initialize torch.distributed (NCCL) if launched under torchrun.
        2. Build tokenizer → derive vocab_size.
        3. Construct OpenMythos with the 3B variant config.
        4. Wrap in FSDP with FULL_SHARD + bf16/fp16 mixed precision (multi-GPU)
           or move to device + autocast (single-GPU).
        5. Build fused AdamW on (possibly sharded) parameters.
        6. Resume from the latest checkpoint in `ckpt_dir` if one exists.
        7. Stream FineWeb-Edu through grad-accumulation microbatches with
           cosine LR schedule, per-step logging, and periodic checkpoints.
        8. Write a final checkpoint if the last save wasn't aligned to
           `ckpt_every`, then barrier + tear down the process group.

    All hyperparameters are literal constants in this function by design —
    pretraining runs are long-lived and each run pins exact settings; a
    CLI/config layer is deliberately avoided to keep the file self-auditable.
    """
    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        logger.info(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer("deepseek-ai/DeepSeek-V2-Lite-Chat")
    vocab_size = encoding.vocab_size

    if master:
        logger.info(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    seq_len = 1024
    micro_batch = 1
    target_tokens = 30_000_000_000
    # grad_accum = max(1, 256 // (world_size * micro_batch))
    grad_accum = 256
    
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = target_tokens // grad_accum // seq_len // 20
    lr = 3e-4
    wd = 0.1
    log_every = 1
    ckpt_every = 500
    ckpt_dir = "checkpoints"
    dataset_subset = "sample-10BT"  # → sample-100BT or "default" for full run
    attention_type = 'gqa'
    
    if master:
        logger.info(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # cfg = mythos_3b()
    cfg = mythos_0_1b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len
    cfg.attn_type = attention_type
    
    
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = OpenMythos(cfg)

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
            cast_forward_inputs=True,  # <--- 新增这一行：强制所有子模块转换输入类型

        )
        wrap_policy = ModuleWrapPolicy({TransformerBlock, RecurrentBlock})
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
            use_orig_params=True,  # 👈 必须加上这行保命！
        )
        # ======== 【修复核心：非重入式引擎 + 取消嵌套】 ========
        # 1. 强制使用现代的 NO_REENTRANT 后端，这是官方强烈推荐与 FSDP 配合的模式
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        
        # ======== 【新增：开启激活重计算】 ========
        # 只对前后向结构稳定的 TransformerBlock 做重计算；跳过 recurrent
        # loop 里的 MoE / ACT 路径，避免 checkpoint 重算图和原始前向不一致。
        # check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
        # apply_activation_checkpointing(
        #     model,
        #     checkpoint_wrapper_fn=non_reentrant_wrapper,
        #     check_fn=check_fn,
        # )
        # =========================================
        
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    # FSDP handles its own mixed precision; only need autocast for single-GPU
    amp_ctx = nullcontext() if ddp else amp_ctx  # type: ignore[possibly-undefined]
    # amp_ctx = (
    #     torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    #     if "cuda" in device
    #     else nullcontext()
    # )
    
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")

    
    
    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    # Streaming datasets are not resumable by position, so re-iterating from
    # the beginning is accepted — at pretraining scale the loss of dataset
    # position is negligible vs. the cost of discarded training steps.
    start_step = 0
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step = load_checkpoint(model, optimizer, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")

    
    
    
    # moe = model.recurrent.block.ffn

    # grouped_experts = GroupedSwiGLUExperts(
    #     n_experts=moe.n_experts,
    #     dim=moe.router.in_features,
    #     expert_dim=moe.routed_experts[0].gate.out_features,
    #     backend="grouped_mm",
    # ).to(
    #     device=moe.router.weight.device,
    #     dtype=torch.bfloat16,
    # )

    # grouped_experts.load_from_module_list(moe.routed_experts)

    # moe.grouped_experts = grouped_experts

    # # 前提是你已经把 MoEFFN.forward 改成调用 self.grouped_experts
    # del moe.routed_experts
    
    model.recurrent.block.ffn.grouped_experts.to(torch.bfloat16)
    
    
    
    model = torch.compile(model)
    
    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = SkyworkDataset(
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        start_step=start_step,
        grad_accum=grad_accum,
        micro_batch=micro_batch,
        avg_tokens_per_doc=1500,
        max_docs_per_worker=None,
        tokenizer_model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    )
    # dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    # TensorBoard writer only on master rank — non-master ranks skip
    # logging entirely (both for the writer API and any scalar ops), so
    # the variable must be defined on every rank even though it is unused.
    writer = None
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)
        # TensorBoard logs land under `runs/` so they sit alongside the
        # checkpoint tree (already gitignored). Each run gets its own
        # event file; reusing the same directory on resume would mix two
        # streams into one curve in the UI.
        tb_dir = os.path.join("runs", os.path.basename(os.path.abspath(ckpt_dir)))
        writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
        logger.info(f"TensorBoard logs -> {tb_dir}")

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step
    # Anchor for the per-window step counter. When `start_step` is not a
    # multiple of `log_every` (resumed mid-run), the first window has fewer
    # than `log_every` steps; dividing by `log_every` would inflate throughput
    # and shrink the ETA. Track the real step delta instead.
    last_log_step = start_step

    while step < total_steps:
        # print('111')
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum):
            # print('222')
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            # print('333')
            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()
            # print('444')
            # if master:
            #     logger.info(
            #         f"step {step:6d}/{total_steps} | micro {micro_step + 1:3d}/{grad_accum} "
            #         f"| loss {loss.item():.4f} | accum {loss_accum:.4f} "
            #         f"| x {x.dtype} y {y.dtype} | sync={'no_sync' if ddp and micro_step != grad_accum - 1 else 'sync'}"
            #     )
        # FSDP shards parameters, so `nn.utils.clip_grad_norm_` would clip
        # against each rank's local norm and miss the cross-shard gather.
        # FSDP.clip_grad_norm_ computes the true global norm and returns it.
        # print('555')
        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1
        # print('666')
        if master and step % log_every == 0:
            dt = time.perf_counter() - t0
            # Use the real step delta over this window, not `log_every` —
            # otherwise a resumed run whose `start_step` is not a multiple of
            # `log_every` would report its first window as a full window's
            # worth of work in less wall time, doubling the throughput and
            # halving the ETA.
            steps_in_window = step - last_log_step
            tokens_in_window = steps_in_window * global_batch_tok
            tok_per_sec = tokens_in_window / dt
            tokens_seen = step * global_batch_tok
            # Average wall time per step over this window × remaining steps.
            # Extrapolating from a recent window (not since start) keeps the
            # ETA responsive if the rate shifts after warmup or a recovery
            # from a stalled batch.
            step_dt = dt / steps_in_window
            eta_secs = (total_steps - step) * step_dt
            logger.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec / 1e3:.2f}k tok/s "
                f"| {tokens_seen / 1e9:.1f}B tokens seen "
                f"| {step_dt:.2f}s/step "
                f"| eta {fmt_eta(eta_secs)}"
            )
            # TensorBoard scalars. Same window-aggregated metrics as the
            # log line above, but separate fields so each one gets its own
            # curve and the y-axis can scale per-metric (loss vs. gnorm
            # otherwise squashes into a flat line near the bottom).
            writer.add_scalar("train/loss", loss_accum, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            writer.add_scalar("train/lr", cur_lr, step)
            writer.add_scalar("train/tokens_per_sec", tok_per_sec, step)
            writer.add_scalar("train/step_seconds", step_dt, step)
            t0 = time.perf_counter()
            last_log_step = step

        if step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master
            )
    # print('777')
    # Final checkpoint — total_steps may not be divisible by ckpt_every, so
    # without this the tail of the run is lost if the schedule doesn't align.
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)

    if ddp:
        # Barrier so no rank exits while another is still finishing its
        # checkpoint gather — avoids NCCL "process group destroyed" noise.
        dist.barrier()
        dist.destroy_process_group()

    if master:
        logger.success("Training complete.")

    # Flush any pending scalar events before exit — if the script is
    # killed by signal rather than exiting the loop cleanly, the unflushed
    # tail of the event file is lost. Closing also forces the final write.
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
