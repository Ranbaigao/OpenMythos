#!/usr/bin/env python3
"""
OpenMythos pretraining on SkyPile with budget-ponder ACT regularization.

Variant of 0.1b_fine_skywork.py that trains FROM SCRATCH with a budget-style
ponder penalty, L += tau * ReLU(ponder - B), which clamps the ACT loop count
near B instead of letting it drift to max_loop_iters (depth collapse — the
cause of the 16k → 4.3k tok/s slowdown in the original run; full analysis in
docs/act_depth_collapse_analysis.md). Requires the ponder-returning
RecurrentBlock in open_mythos/main.py (return_act=True).

Differences from 0.1b_fine_skywork.py:
    1. Budget-ponder penalty from step 0 (PONDER_TAU / PONDER_BUDGET below).
    2. Logs train/ponder and train/act_loops to TensorBoard for drift monitoring.
    3. Fresh ckpt_dir (checkpoints_act) so the collapsed checkpoints of the
       previous run are not auto-resumed.

Single GPU:
    python training/0.1b_fine_skywork_act.py

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") training/0.1b_fine_skywork_act.py
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



import math
import queue
import threading
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

from open_mythos import OpenMythos
from open_mythos.main import TransformerBlock, RecurrentBlock, Expert
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
                 tokenizer_model_id: str = None,
                 resume_docs: dict = None):
        
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
        # 逐 worker 精确文档计数(来自 checkpoint 的 data_docs_seen),
        # {worker_id: 已消费文档数}。优先于 start_step 的估算路径。
        # 注意: 按 rank-0 的 worker_id 索引; 多 rank 场景需按 (rank, worker) 扩展。
        self.resume_docs = resume_docs
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

        # =================================================================
        # 🚀 彻底抛弃不稳定的 HF 镜像，使用阿里云官方魔搭节点
        # =================================================================
        from modelscope.msdatasets import MsDataset

        # -----------------------------------------------------------------
        # 断点位置: 优先逐 worker 实时文档计数(resume_docs 与主循环共享
        # 同一个 dict —— 启动时来自 checkpoint, 之后被主循环就地更新,
        # 因此 worker 死亡重建时拿到的是最新位置而不是冻结的起点);
        # 旧 checkpoint 没有计数器时, 回退到 步数×平均文档长 的估算。
        # docs_seen 是 worker 在分片内的绝对文档位置(含初始 skip),
        # 之后每消费一篇 +1, 随样本上报给主进程记账。
        # -----------------------------------------------------------------
        saved = self.resume_docs.get(worker_id) if self.resume_docs is not None else None
        if saved is not None:
            docs_seen = int(saved)
            if worker_id == 0:
                logger.info(f"[Rank {self.rank} Worker 0] 精确断点恢复: "
                            f"跳过 {docs_seen:,} 篇(实时计数)")
        elif self.start_step > 0:
            total_samples_for_rank = self.start_step * self.grad_accum * self.micro_batch
            samples_to_skip = total_samples_for_rank // num_workers
            if worker_id < total_samples_for_rank % num_workers:
                samples_to_skip += 1
            tokens_to_skip = samples_to_skip * self.seq_len
            auto_skip_docs = tokens_to_skip // self.avg_tokens_per_doc
            docs_seen = auto_skip_docs
            if worker_id == 0:
                logger.info(f"[Rank {self.rank} Worker 0] 估算断点恢复(无计数器): "
                            f"跳过 {auto_skip_docs:,} 篇 "
                            f"(= {tokens_to_skip:,} token ÷ 平均文档长 "
                            f"{self.avg_tokens_per_doc}, 近似值, 可能回放部分数据)")
        else:
            docs_seen = 0
        if self.skip_docs_per_worker > 0:
            docs_seen += self.skip_docs_per_worker
        initial_docs_seen = docs_seen

        # MsDataset.load 每个进程只能调用一次: 它会为 datasets.streaming
        # 打补丁, 同进程第二次调用重复打补丁 → wrapper 自我递归, 必现
        # RecursionError (2026-07-25 线上事故)。因此 load+shard 只做一次,
        # 原地重试时仅在惰性管线上重建 skip/take/iter (纯本地操作)。
        base = MsDataset.load(
            "swift/SkyPile-150B",
            split="train",
            use_streaming=True,      # 魔搭的流式加载参数是 use_streaming
        )
        base = base.shard(num_shards=total_shards, index=shard_index)

        def open_stream():
            """在惰性管线上重建 skip→take→iter (无网络请求)。重试时调用。"""
            s = base
            if docs_seen > 0:
                s = s.skip(docs_seen)
            if self.max_docs_per_worker is not None:
                remaining = self.max_docs_per_worker - (docs_seen - initial_docs_seen)
                if remaining <= 0:
                    return iter([])
                s = s.take(remaining)
            return iter(s)

        # -----------------------------------------------------------------
        # 消费循环 + 原地重试: 网络瞬断(如 WSL DNS 抖动 → httpx session
        # 被关闭)时, worker 自己重建流并从精确的 docs_seen 续上, 不牵连
        # 其他 worker, 也不需要 Prefetcher 全员重建。重试耗尽才抛给上层。
        # -----------------------------------------------------------------
        buf = []
        max_retries = 5
        attempt = 0
        while True:
            try:
                for sample in open_stream():
                    text_content = sample.get("text", sample.get("content", ""))
                    buf.extend(self.encoding.encode(text_content))
                    docs_seen += 1
                    while len(buf) >= self.seq_len + 1:
                        chunk = buf[: self.seq_len + 1]
                        buf = buf[self.seq_len + 1 :]
                        yield (
                            torch.tensor(chunk[:-1], dtype=torch.long),
                            torch.tensor(chunk[1:], dtype=torch.long),
                            worker_id,
                            docs_seen,
                        )
                return  # 流正常耗尽
            except Exception as exc:
                if attempt >= max_retries:
                    raise  # 交给 Prefetcher 的重建逻辑兜底
                attempt += 1
                wait = min(5 * attempt, 30)
                logger.warning(
                    f"[Rank {self.rank} Worker {worker_id}] 数据流中断 "
                    f"({type(exc).__name__}), {wait}s 后原地重试 "
                    f"(第 {attempt}/{max_retries} 次, 从第 {docs_seen:,} 篇续上)")
                time.sleep(wait)
                # 死亡的全局 HTTP session 不重置则重建流也会立即失败
                # (已验证: close_session() 后 get_session() 可进程内复活)
                try:
                    from huggingface_hub.utils import _http as _hf_http
                    _hf_http.close_session()
                except Exception:
                    pass
                
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
# Resilient prefetcher: deep buffer + auto-rebuild between DataLoader and loop
# ---------------------------------------------------------------------------


class Prefetcher:
    """
    Background-thread prefetch queue that decouples the training loop from the
    bursty remote stream.

    Why this exists (measured on the previous run):
      - The ModelScope stream is latency-bound, not bandwidth-bound: steady
        supply ~22 samples/s but delivered in bursts (shard transitions,
        reconnects), while the default DataLoader buffer is only
        num_workers × prefetch_factor = 8 samples — any network hiccup
        longer than a second reaches the GPU directly (observed: at a
        constant 8 ACT loops, step time varied 15.4s → 49.0s).
      - The streaming HTTP layer can die mid-run (httpx "Cannot send a
        request, as the client has been closed"), which crashed the previous
        run at step 272. Here any stream exception only kills the current
        DataLoader; the daemon thread backs off and rebuilds workers from
        scratch instead of taking the run down.

    The queue depth absorbs ~1 minute of total supply outage at the
    consumption rate of ~17 samples/s (256 samples / 15s step).
    """

    def __init__(self, dataset, micro_batch: int, num_workers: int,
                 capacity: int = 1024, retry_backoff: float = 10.0):
        self._dataset = dataset
        self._micro_batch = micro_batch
        self._num_workers = num_workers
        self._retry_backoff = retry_backoff
        self._q: queue.Queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="data-prefetcher")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                loader = DataLoader(
                    self._dataset,
                    batch_size=self._micro_batch,
                    num_workers=self._num_workers,
                    pin_memory=True,
                )
                for batch in loader:
                    if self._stop.is_set():
                        return
                    self._q.put(batch)  # blocks when full → backpressure
                # Stream exhausted (finite dataset) → rebuild and re-enter,
                # matching the previous StopIteration-restart behavior.
                logger.warning("[prefetch] stream exhausted; restarting from shard begin")
            except Exception as exc:
                if self._stop.is_set():
                    return
                logger.warning(
                    f"[prefetch] stream died ({type(exc).__name__}: {exc}); "
                    f"rebuilding workers in {self._retry_backoff:.0f}s"
                )
                self._stop.wait(self._retry_backoff)

    def next(self):
        """Block until the next micro-batch is available."""
        return self._q.get()

    def queue_size(self) -> int:
        return self._q.qsize()

    def close(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# System monitor: one run carries all the evidence needed to attribute a
# slowdown — no rerun experiments required
# ---------------------------------------------------------------------------


class SystemMonitor:
    """
    Background sampler writing hardware/system health scalars to TensorBoard.

    Covers every plausible "starts fast, degrades over time" cause outside the
    training step itself:

      gpu/temp_c, gpu/sm_clock_mhz, gpu/power_w, gpu/util_pct,
      gpu/throttle_bitmask   -- thermal throttling: clocks sag as the card
                                heats up over hours (common on desktop GPUs)
      cuda/mem_reserved_gb, cuda/mem_allocated_gb, cuda/mem_fragmentation
                             -- allocator fragmentation growth; on WSL a full
                                VRAM also triggers paging to system RAM, which
                                is catastrophic for step time
      sys/rss_gb, sys/cpu_pct-- dataloader worker memory leaks, CPU contention
      data/prefetch_queue    -- leading indicator of data starvation: the
                                queue drains before data_wait rises

    Attribution cheat sheet (step_seconds ≈ data_wait + fwd_bwd + optim):
      train/data_wait_seconds ↑    → stream / network
      train/compute_per_loop ↑     → GPU-side: check gpu/temp + sm_clock
                                     (thermal), cuda/mem (paging/fragmentation)
      train/optim_seconds ↑        → optimizer / grad-clip path
      spikes every ckpt_every      → checkpoint saving (train/ckpt_save_seconds)
      data/prefetch_queue → 0      → data supply falling behind demand
      sys/rss_gb 单调上涨           → memory leak (likely dataloader workers)
    """

    def __init__(self, writer, prefetcher, get_step, interval: float = 10.0):
        self._writer = writer
        self._prefetcher = prefetcher
        self._get_step = get_step
        self._interval = interval
        self._stop = threading.Event()

        import pynvml
        import psutil

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._proc = psutil.Process()
        self._psutil = psutil

        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="system-monitor")
        self._thread.start()

    def _tree_rss_gb(self) -> float:
        """RSS of the main process + all children (dataloader workers)."""
        total = self._proc.memory_info().rss
        for child in self._proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except self._psutil.NoSuchProcess:
                pass
        return total / 1e9

    def _sample(self) -> None:
        w = self._writer
        step = self._get_step()
        nv = self._nvml
        try:
            w.add_scalar("gpu/temp_c", nv.nvmlDeviceGetTemperature(self._gpu, 0), step)
            w.add_scalar("gpu/sm_clock_mhz",
                         nv.nvmlDeviceGetClockInfo(self._gpu, nv.NVML_CLOCK_SM), step)
            w.add_scalar("gpu/power_w",
                         nv.nvmlDeviceGetPowerUsage(self._gpu) / 1000.0, step)
            w.add_scalar("gpu/util_pct",
                         nv.nvmlDeviceGetUtilizationRates(self._gpu).gpu, step)
            w.add_scalar("gpu/vram_used_mb",
                         nv.nvmlDeviceGetMemoryInfo(self._gpu).used / 1e6, step)
            try:
                w.add_scalar("gpu/throttle_bitmask",
                             nv.nvmlDeviceGetCurrentClocksThrottleReasons(self._gpu),
                             step)
            except Exception:
                pass
        except Exception:
            pass  # NVML hiccup must never threaten the run

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            w.add_scalar("cuda/mem_allocated_gb", alloc, step)
            w.add_scalar("cuda/mem_reserved_gb", reserved, step)
            if reserved > 0:
                w.add_scalar("cuda/mem_fragmentation", 1.0 - alloc / reserved, step)

        try:
            w.add_scalar("sys/rss_gb", self._tree_rss_gb(), step)
            w.add_scalar("sys/cpu_pct", self._psutil.cpu_percent(), step)
        except Exception:
            pass
        w.add_scalar("data/prefetch_queue", self._prefetcher.queue_size(), step)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._sample()
            except Exception:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass


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
    optim_adamw,
    optim_muon,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    ddp: bool,
    master: bool,
    keep_last: int = 3,
    data_docs_seen: dict = None,
    data_meta: dict = None,
) -> None:
    """
    Gather full model + both optimizer states, write atomically, prune old files.

    Under FSDP all three states are collected inside a single FULL_STATE_DICT
    context so the optim-state tensors bind to fully-unsharded parameters;
    mixing contexts between model and optimizer has caused silent divergence
    on resume in past torch versions. Because the model uses *two* optimizers
    (AdamW for non-2D params, Muon for 2D matrices) both states must be saved
    separately — Muon's Newton-Schulz momentum buffer is not interchangeable
    with AdamW's exp-avgs, and on resume both are required for the schedule
    to be bit-exact identical. The temp-file + os.replace write means a kill
    mid-save leaves the previous checkpoint intact instead of a truncated
    .pt file. Non-master ranks participate in the FSDP gather (otherwise
    the collective would hang) but exit before touching disk.

    Args:
        model         -- FSDP-wrapped (ddp=True) or raw (ddp=False) model
        optim_adamw   -- AdamW over embeddings / norms / scalar params
        optim_muon    -- Muon over 2D weight matrices
        step          -- global step number; encoded zero-padded into filename
        cfg           -- model config object; saved so downstream eval can
                         reconstruct the model without re-importing the variant
        vocab_size    -- tokenizer vocab size at train time; saved for sanity-check
                         on load against a (possibly updated) tokenizer
        ckpt_dir      -- directory to write into; created if missing
        ddp           -- True if FSDP path; False for single-GPU / CPU
        master        -- whether this rank writes to disk (rank 0 only)
        keep_last     -- number of most-recent checkpoints to retain; older ones
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
            optim_adamw_state = FSDP.optim_state_dict(model, optim_adamw)
            optim_muon_state = FSDP.optim_state_dict(model, optim_muon)
    else:
        model_state = model.state_dict()
        optim_adamw_state = optim_adamw.state_dict()
        optim_muon_state = optim_muon.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optim_adamw": optim_adamw_state,
            "optim_muon": optim_muon_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
            # 数据断点: 逐 worker "已训练到" 的文档位置 + 有效性元信息。
            # 恢复时若 meta 与当前配置不一致, 计数作废并回退步数估算。
            "data_docs_seen": data_docs_seen,
            "data_meta": data_meta,
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


def load_checkpoint(
    model,
    optim_adamw,
    optim_muon,
    path: str,
    ddp: bool,
) -> int:
    """
    Restore model + both optimizers from disk, returning the step to resume at.

    Every rank reads the file (`rank0_only=False` on load) so FSDP has access
    to the full state on each rank — the complement to the `rank0_only=True`
    save path. Must mirror save's single-context pattern; splitting the model
    and optimizer loads across two `state_dict_type` blocks has historically
    produced optimizer state bound to the wrong shard shapes.

    Both optimizers are restored under the same FULL_STATE_DICT context as
    the model — important because FSDP reshards parameters lazily and the
    optimizer state must be re-keyed against the current shard layout. Muon
    state is restored alongside AdamW; their buffers (exp-avgs vs Newton-
    Schulz momentum) are different in shape and dtype, so the keys in the
    saved dict must match (`optim_adamw` vs `optim_muon`).

    `weights_only=False` is required because the checkpoint contains the
    pickled `cfg` dataclass — flip to `weights_only=True` only if you
    separate config out.

    Args:
        model        -- same FSDP-wrapped or raw model used during save
        optim_adamw  -- freshly constructed AdamW to be filled in-place
        optim_muon   -- freshly constructed Muon to be filled in-place
        path         -- absolute path to a `step_{N:07d}.pt` file
        ddp          -- whether the model is FSDP-wrapped; must match save

    Returns:
        (step, data_docs_seen, data_meta):
            step           -- the step number the checkpoint was taken at;
            data_docs_seen -- 逐 worker 已训练文档计数 dict, 旧 checkpoint
                              没有该字段时为 None (调用方回退到步数估算);
            data_meta      -- 计数有效性元信息 (workers/grad_accum/数据集),
                              没有时为 None
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Tolerate checkpoints saved through torch.compile (keys carry an
    # "_orig_mod." prefix) so older runs remain resumable after compile
    # was disabled in this script.
    if any(k.startswith("_orig_mod.") for k in ckpt["model"]):
        ckpt["model"] = {
            k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()
        }

    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_adamw.load_state_dict(
                FSDP.optim_state_dict_to_load(
                    model=model,
                    optim=optim_adamw,
                    optim_state_dict=ckpt["optim_adamw"],
                )
            )
            optim_muon.load_state_dict(
                FSDP.optim_state_dict_to_load(
                    model=model,
                    optim=optim_muon,
                    optim_state_dict=ckpt["optim_muon"],
                )
            )
    else:
        model.load_state_dict(ckpt["model"])
        optim_adamw.load_state_dict(ckpt["optim_adamw"])
        optim_muon.load_state_dict(ckpt["optim_muon"])

    return (int(ckpt["step"]),
            ckpt.get("data_docs_seen"),
            ckpt.get("data_meta"))


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
    encoding = MythosTokenizer("Langboat/mengzi-t5-base")
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
    warmup_steps = 1000
    lr = 3e-4
    # Muon's recommended peak LR for 0.1B-scale transformers — large enough
    # that the Newton-Schulz step on 2D matrices roughly matches AdamW's
    # per-token progress, small enough that warmup is required. The
    # schedule (warmup → cosine) is otherwise shaped identically to AdamW.
    muon_lr = 0.02
    wd = 0.1
    log_every = 1
    ckpt_every = 200
    ckpt_dir = "checkpoints_act"  # fresh dir: previous run's collapsed ckpts live in "checkpoints"
    # Per-position budget-ponder ACT regularization
    # (see docs/act_depth_collapse_analysis.md §7):
    # penalty = tau * mean( ReLU(ponder_per_position - budget) ).
    # Each late-halting position is penalized individually, so the ~1%
    # halting tail can no longer hide behind the batch mean and pin every
    # batch at max_loop_iters. Below budget the gradient is exactly zero,
    # so healthy positions are never crushed toward 1 loop (the failure
    # mode of a flat ponder penalty). Budget 3 sits at the measured
    # per-position median halt step (~3.4 mean / p50=3 at step 500).
    ponder_tau = 1e-2
    ponder_budget = 3.0
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
    # Fixed seed before model construction: the ACT halting head in
    # RecurrentBlock is randomly initialized, and the early-exit depth
    # (4~8 of max_loop_iters) is decided by that init. Without a seed each
    # run lands on a different effective loop depth, showing up as stable
    # but run-to-run-different step times (observed ~2x spread).
    torch.manual_seed(42)
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
    

    muon_params = [p for p in model.parameters() if p.ndim == 2]

    other_params = [p for p in model.parameters() if p.ndim != 2]

    optim_adamw  = torch.optim.AdamW(
        other_params, lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )
    optim_muon = torch.optim.Muon(muon_params, lr=muon_lr, momentum=0.95)

    # ------------------------------------------------------------------
    # Resume from latest checkpoint (if any)
    # ------------------------------------------------------------------
    # 数据断点: checkpoint 里的 data_docs_seen 提供逐 worker 精确文档位置;
    # 旧 checkpoint 没有该字段时回退到 start_step × 平均文档长的估算
    # (dataset 内两条路径, 见 SkyworkDataset.__iter__)。
    num_workers = 8
    start_step = 0
    resume_docs = None
    data_meta = {"num_workers": num_workers, "grad_accum": grad_accum,
                 "micro_batch": micro_batch, "dataset": "swift/SkyPile-150B"}
    existing_ckpts = _list_ckpts(ckpt_dir)
    if existing_ckpts:
        latest = existing_ckpts[-1]
        if master:
            logger.info(f"Resuming from checkpoint: {latest}")
        start_step, saved_docs, saved_meta = load_checkpoint(
            model, optim_adamw, optim_muon, latest, ddp)
        if master:
            logger.success(f"Resumed at step {start_step}")
        if saved_docs and saved_meta == data_meta:
            resume_docs = {int(k): int(v) for k, v in saved_docs.items()}
            if master:
                logger.info(f"数据断点: 使用 checkpoint 逐 worker 文档计数 "
                            f"(共 {sum(resume_docs.values()):,} 篇)")
        else:
            if saved_docs and master:
                logger.warning("数据配置(workers/grad_accum/数据集)与 checkpoint "
                               "不一致, 文档计数失效, 回退到步数估算")
            elif master:
                logger.warning("checkpoint 无文档计数(旧版), "
                               "回退到步数×平均文档长的估算断点")

    # torch.compile deliberately disabled: measured on this 0.1B model,
    # compiled fwd+bwd is 1.25–1.31× SLOWER than eager (the data-dependent
    # ACT break forces ~8 graph breaks per forward, and kernels at
    # dim=512/seq=1024/batch=1 are too small for inductor gains to cover
    # the overhead). Revisit if the model gets bigger or the ACT loop
    # becomes static.
    # model = torch.compile(model)
    
    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    # 与 dataset 共享同一个 dict 对象: 主循环就地更新 "已训练到" 的位置,
    # Prefetcher 重建 DataLoader 时数据集随 pickle 带上最新计数 → worker
    # 死亡重建也能精确续上, 而不是回退到进程启动时冻结的 start_step 估算
    # (2026-07-25 线上事故)。同时它就是 checkpoint 落盘的内容。
    docs_seen_latest = resume_docs if resume_docs is not None else {}
    dataset = SkyworkDataset(
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        start_step=start_step,
        grad_accum=grad_accum,
        micro_batch=micro_batch,
        # 实测平均文档长(boundary 实验: 文档末尾 token 占比 0.14-0.17%
        # → 均值 ~600-700); 取略偏高的值使估算偏向"回放"而非"跳过未读数据"
        avg_tokens_per_doc=700,
        max_docs_per_worker=None,
        tokenizer_model_id="Langboat/mengzi-t5-base",
        resume_docs=docs_seen_latest,
    )
    # dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    # 8 workers: the stream is per-request-latency-bound (HTTP range reads),
    # so more parallel connections genuinely raise supply. The Prefetcher
    # adds a deep buffer + auto-rebuild on stream death (see its docstring).
    prefetcher = Prefetcher(dataset, micro_batch, num_workers, capacity=1024)

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
    t0 = time.perf_counter()
    step = start_step
    # Anchor for the per-window step counter. When `start_step` is not a
    # multiple of `log_every` (resumed mid-run), the first window has fewer
    # than `log_every` steps; dividing by `log_every` would inflate throughput
    # and shrink the ETA. Track the real step delta instead.
    last_log_step = start_step

    # Hardware/system health sampler (master only). Runs in a daemon thread
    # and writes gpu/*, cuda/*, sys/*, data/* scalars every 10s — see the
    # SystemMonitor docstring for the attribution cheat sheet.
    monitor = None
    if master:
        monitor = SystemMonitor(writer, prefetcher, lambda: step, interval=10.0)

    while step < total_steps:
        # print('111')
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        cur_muon_lr = get_lr(step, warmup_steps, total_steps, muon_lr, muon_lr * 0.1)
        for g in optim_adamw.param_groups:
            g["lr"] = cur_lr
        for g in optim_muon.param_groups:
            g["lr"] = cur_muon_lr

        optim_muon.zero_grad()
        optim_adamw.zero_grad()
        loss_accum = 0.0
        ponder_accum = 0.0
        loops_max = 0.0
        data_wait = 0.0
        fwd_bwd = 0.0

        for micro_step in range(grad_accum):
            # print('222')
            t_data = time.perf_counter()
            x, y, wid, dseen = prefetcher.next()
            data_wait += time.perf_counter() - t_data

            # 记录"最后被训练样本"的逐 worker 文档位置(而非预取队列头的
            # 位置), 随 checkpoint 落盘 → 手动 kill/崩溃后能从精确断点续训,
            # 既不回放也不丢失。batch 内样本同 worker, 取最后一个(最新)。
            docs_seen_latest[int(wid.reshape(-1)[-1])] = int(dseen.reshape(-1)[-1])

            x = x.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)
            y = y.to(device if not ddp else f"cuda:{local_rank}", non_blocking=True)

            # print('333')
            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )

            t_fb = time.perf_counter()
            with sync, amp_ctx:
                logits, act = model(x, return_act=True)
                task_loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                # Per-position budget penalty (tail penalty): each position
                # whose expected loop count exceeds ponder_budget is penalized
                # individually; positions under budget get exactly zero
                # gradient. The previous mean version let ~1% late-halting
                # positions hide behind the batch average — the mean stayed
                # clamped at 2.1 while the per-batch MAX halt step drifted to
                # 8, pinning every batch at max_loop_iters (batch exits only
                # when ALL 1024 positions halt). With the per-position hinge,
                # penalty == 0  <=>  every position <= B  <=>  the batch
                # actually exits at B loops.
                ponder_all = act["ponder"].float()
                ponder = ponder_all.mean()
                ponder_pen = torch.relu(ponder_all - ponder_budget).mean()
                loss = (task_loss + ponder_tau * ponder_pen) / grad_accum
            loss.backward()
            loss_accum += task_loss.item() / grad_accum
            ponder_accum += ponder.item() / grad_accum
            loops_max = max(loops_max, act["loops"])
            fwd_bwd += time.perf_counter() - t_fb
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
        t_optim = time.perf_counter()
        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Muon sees only 2D matrices; AdamW sees the rest (embeddings,
        # norms, scalars). Combined `grad_norm` clips both together — log a
        # separate Muon-only norm so a divergence between the two optimizers'
        # gradient scales is visible in TensorBoard (a Muon norm 10× AdamW
        # norm is a strong signal that the orthogonalization is destabilizing).
        muon_grad_norm = sum(
            p.grad.detach().float().norm() ** 2
            for p in optim_muon.param_groups[0]["params"]
            if p.grad is not None
        ).sqrt()
        optim_muon.step()
        optim_adamw.step()
        if not ddp and torch.cuda.is_available():
            torch.cuda.synchronize()  # make the optim segment timing honest
        optim_time = time.perf_counter() - t_optim
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
                f"| gnorm {float(grad_norm):.2f} "
                f"| gnorm_muon {float(muon_grad_norm):.2f} "
                f"| lr_a {cur_lr:.2e} | lr_m {cur_muon_lr:.2e} "
                f"| pond {ponder_accum:.2f} | loops {loops_max:.0f} "
                f"| data {data_wait:.1f}s | fb {fwd_bwd:.1f}s | opt {optim_time:.1f}s "
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
            writer.add_scalar("train/grad_norm_muon", float(muon_grad_norm), step)
            # AdamW and Muon follow independent LR schedules with different
            # peaks and floors. Splitting them into two scalars avoids one
            # curve being a flat line near the bottom of the other in the
            # shared panel; the ratio is a useful divergence diagnostic
            # (it should track the ratio of peaks = 0.02 / 3e-4 ≈ 67).
            writer.add_scalar("train/lr_adamw", cur_lr, step)
            writer.add_scalar("train/lr_muon", cur_muon_lr, step)
            writer.add_scalar(
                "train/lr_ratio_muon_to_adamw",
                cur_muon_lr / cur_lr if cur_lr > 0 else 0.0,
                step,
            )
            writer.add_scalar("train/tokens_per_sec", tok_per_sec, step)
            writer.add_scalar("train/step_seconds", step_dt, step)
            # ACT health metrics. ponder (mean expected loop count) should sit
            # near ponder_budget once the penalty is active; a steady upward
            # drift is the early warning of depth collapse, and act_loops (max
            # loops any micro-batch actually executed) is what maps directly
            # to step_seconds.
            writer.add_scalar("train/ponder", ponder_accum, step)
            writer.add_scalar("train/act_loops", loops_max, step)
            # Wall-clock time this step spent waiting on the data stream.
            # This is the direct evidence for data-vs-compute bottleneck
            # attribution: at fixed act_loops, step_seconds ≈ compute +
            # data_wait, so this curve should stay near zero once the
            # prefetch buffer is deep enough.
            writer.add_scalar("train/data_wait_seconds", data_wait, step)
            # Step-time decomposition: step_seconds ≈ data_wait + fwd_bwd +
            # optim (+ checkpoint saves every ckpt_every steps). This is the
            # master key for attributing any future slowdown in ONE run:
            #   data_wait ↑        → stream/network
            #   fwd_bwd ↑ with
            #     flat act_loops   → GPU-side (cross-check gpu/temp_c,
            #                        gpu/sm_clock_mhz, cuda/mem_fragmentation)
            #   optim ↑            → optimizer / grad-clip path
            # compute_per_loop normalizes fwd_bwd by the ACT loop count so
            # ACT depth changes don't masquerade as hardware slowdowns.
            writer.add_scalar("train/fwd_bwd_seconds", fwd_bwd, step)
            writer.add_scalar("train/optim_seconds", optim_time, step)
            writer.add_scalar(
                "train/compute_per_loop",
                fwd_bwd / grad_accum / max(loops_max, 1.0),
                step,
            )
            t0 = time.perf_counter()
            last_log_step = step

        if step % ckpt_every == 0:
            t_ckpt = time.perf_counter()
            save_checkpoint(
                model,
                optim_adamw,
                optim_muon,
                step,
                cfg,
                vocab_size,
                ckpt_dir,
                ddp,
                master,
                data_docs_seen=docs_seen_latest,
                data_meta=data_meta,
            )
            # Checkpoint saves are periodic step-time spikes; keep the curve
            # so they don't get misread as a data or GPU regression.
            if master:
                writer.add_scalar(
                    "train/ckpt_save_seconds", time.perf_counter() - t_ckpt, step
                )
    # print('777')
    prefetcher.close()
    if monitor is not None:
        monitor.close()
    # Final checkpoint — total_steps may not be divisible by ckpt_every, so
    # without this the tail of the run is lost if the schedule doesn't align.
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(
            model,
            optim_adamw,
            optim_muon,
            step,
            cfg,
            vocab_size,
            ckpt_dir,
            ddp,
            master,
            data_docs_seen=docs_seen_latest,
            data_meta=data_meta,
        )

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
