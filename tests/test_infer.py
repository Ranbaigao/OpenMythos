#!/usr/bin/env python3
"""
OpenMythos 单机推理脚本
用于加载 FSDP 保存的 checkpoint 并进行文本生成

用法示例：
    # 使用默认参数
    python tests/test_infer.py

    # 自定义 prompt 和生成参数
    python tests/test_infer.py \\
        --prompt "人工智能的未来发展前景" \\
        --max_new_tokens 256 \\
        --temperature 0.7 \\
        --top_k 50 \\
        --repetition_penalty 1.3 \\
        --n_loops 12

    # 禁用重复惩罚（设为 1.0）
    python tests/test_infer.py --repetition_penalty 1.0
"""

import os
import glob
import argparse
import torch
from typing import Optional
from loguru import logger

# 引入模型和分词器
from open_mythos.main import OpenMythos
from open_mythos.tokenizer import MythosTokenizer


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    返回 checkpoints 目录下最新的 .pt 文件（按修改时间排序）
    """
    pt_files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not pt_files:
        return None
    return max(pt_files, key=os.path.getmtime)


def load_inference_model(ckpt_path: str, device: str = "cuda"):
    """
    从完整的 FSDP checkpoint 中提取配置并加载模型
    """
    logger.info(f"正在加载 Checkpoint: {ckpt_path}")

    # 1. 加载 checkpoint 文件（先加载到 CPU 内存）
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 2. 提取超参数 Config 和 词表大小
    cfg = ckpt["cfg"]
    # 兼容处理：确保配置了训练时的 vocab_size
    if hasattr(cfg, "vocab_size") is False:
        cfg.vocab_size = ckpt.get("vocab_size", 199998)

    logger.info(
        f"成功读取配置 - 参数维度: {cfg.dim}, 注意力类型: {cfg.attn_type}, 预设循环深度: {cfg.max_loop_iters}"
    )

    # 3. 初始化模型骨架
    model = OpenMythos(cfg)

    # 4. 清理 FSDP 前缀 (FSDP 收集的 state_dict 可能会带有多余的前缀)
    raw_state_dict = ckpt["model"]
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        # 移除 _fsdp_wrapped_module. 这种由 FSDP wrapper 自动加上的前缀
        clean_key = k.replace("_fsdp_wrapped_module.", "")
        # 移除 torch.compile 包装后 state_dict 带上的 _orig_mod. 前缀
        clean_key = clean_key.replace("_orig_mod.", "")

        # 可选：如果为了节省显存之前删除了 head.weight，这里跳过它，模型会自动复用 embed.weight
        clean_state_dict[clean_key] = v

    # 5. 加载权重 (strict=False 可以容忍由于 head.weight 共享带来的缺失或多余)
    load_result = model.load_state_dict(clean_state_dict, strict=False)
    # strict=False 下前缀不匹配会静默变成随机初始化，必须显式检查
    unexpected_missing = [k for k in load_result.missing_keys if k != "head.weight"]
    if unexpected_missing or load_result.unexpected_keys:
        logger.warning(
            f"权重可能未正确加载！missing: {unexpected_missing[:5]} "
            f"(共 {len(unexpected_missing)} 个), unexpected: {load_result.unexpected_keys[:5]} "
            f"(共 {len(load_result.unexpected_keys)} 个)"
        )

    # 6. 将模型转换到对应设备和半精度 bfloat16（与训练时对齐）
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_supported else torch.float16

    # 注意：不能用 model.to(dtype=...) 一把梭 —— 它会把 complex64 的
    # freqs_cis RoPE buffer 强转为实数（丢弃虚部），导致 RoPE 完全失效。
    # 只转换浮点参数/浮点 buffer，保留复数 buffer。
    model = model.to(device=device)
    model._apply(lambda t: t.to(dtype) if t.is_floating_point() else t)
    model.eval()  # 切换到推理模式（关闭 Dropout 等）

    logger.success(f"模型加载完毕！当前设备: {device}, 精度: {dtype}")
    return model


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="OpenMythos 文本生成推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 路径与分词器
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/home/ranhao/projects/OpenMythos/checkpoints_act",
        help="checkpoints 目录路径，自动读取最新的 .pt",
    )
    parser.add_argument(
        "--encoder_model_id",
        type=str,
        default="Langboat/mengzi-t5-base",
        help="必须与训练时使用的分词器一致，否则 token id 会超出模型 Embedding 范围",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备",
    )

    # Prompt 与生成长度
    parser.add_argument(
        "--prompt",
        type=str,
        default="山西警方扫黑除恶行动集中收网",
        help="想让模型续写的文本",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="生成的最大 token 数量",
    )

    # 采样参数
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="温度（越高越随机，越低越确定；1.0 = 标准分布，0 = 贪心）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="top-k 采样（0 = 禁用）",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="重复词惩罚系数（1.0 = 关闭，>1.0 抑制已出现过的 token，<1.0 鼓励重复）",
    )
    parser.add_argument(
        "--no_repetition_penalty",
        action="store_true",
        help="便捷开关：禁用重复惩罚（等价于 --repetition_penalty 1.0）",
    )

    # 循环深度
    parser.add_argument(
        "--n_loops",
        type=int,
        default=8,
        help="推理时的循环深度（可以大于训练时的 max_loop_iters 以实现深度外推）",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ================= 配置区（来自命令行） =================
    CHECKPOINT_DIR = args.checkpoint_dir
    ENCODER_MODEL_ID = args.encoder_model_id
    PROMPT_TEXT = args.prompt

    # 推理超参数
    MAX_NEW_TOKENS = args.max_new_tokens
    TEMPERATURE = args.temperature
    TOP_K = args.top_k
    # --no_repetition_penalty 开关会强制设为 1.0（关闭惩罚）
    REPETITION_PENALTY = 2.0 if args.no_repetition_penalty else args.repetition_penalty
    INFERENCE_LOOPS = args.n_loops
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    # ==========================================

    # 打印关键推理参数，方便确认
    logger.info("=" * 50)
    logger.info("推理参数:")
    logger.info(f"  Prompt: {PROMPT_TEXT}")
    logger.info(f"  max_new_tokens: {MAX_NEW_TOKENS}")
    logger.info(f"  temperature: {TEMPERATURE}")
    logger.info(f"  top_k: {TOP_K}")
    logger.info(
        f"  repetition_penalty: {REPETITION_PENALTY} "
        f"({'禁用' if REPETITION_PENALTY == 1.0 else '抑制重复' if REPETITION_PENALTY > 1.0 else '鼓励重复'})"
    )
    logger.info(f"  n_loops: {INFERENCE_LOOPS}")
    logger.info(f"  device: {device}")
    logger.info("=" * 50)

    # 1. 实例化分词器（与训练时使用的模型一致）
    tokenizer = MythosTokenizer(model_id=ENCODER_MODEL_ID)

    # 2. 加载模型
    CHECKPOINT_PATH = find_latest_checkpoint(CHECKPOINT_DIR)
    if not CHECKPOINT_PATH or not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"在目录中找不到任何 .pt checkpoint: {CHECKPOINT_DIR}")
        return

    logger.info(f"自动选中最新 Checkpoint: {CHECKPOINT_PATH}")
    model = load_inference_model(CHECKPOINT_PATH, device)

    # 3. 文本预处理 (Encode)
    logger.info(f"输入 Prompt: \n{PROMPT_TEXT}\n" + "-" * 40)

    # 将文本编码为 ID 列表，并转换为 shape 为 (1, T) 的 tensor
    input_ids = tokenizer.encode(PROMPT_TEXT)

    # 分词器与训练时不一致时，token id 会超出 Embedding 行数，
    # 在 CUDA 上表现为 device-side assert（indexSelect 越界），提前给出可读报错
    model_vocab = model.embed.weight.shape[0]
    max_id = max(input_ids) if input_ids else -1
    if max_id >= model_vocab:
        logger.error(
            f"分词器与模型不匹配：编码得到最大 token id {max_id}，"
            f"但模型 Embedding 只有 {model_vocab} 行。"
            f"请用 --encoder_model_id 指定训练时使用的分词器（当前: {ENCODER_MODEL_ID}）。"
        )
        return

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 4. 生成文本
    logger.info("开始生成...")
    with torch.no_grad():  # 推理时一定要禁用梯度图
        with torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=model.embed.weight.dtype,
        ):
            output_tensor = model.generate(
                input_ids=input_tensor,
                max_new_tokens=MAX_NEW_TOKENS,
                n_loops=INFERENCE_LOOPS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
            )

    # 5. 解码输出
    # generate 返回的 shape 是 (1, T + max_new_tokens)
    generated_ids = output_tensor[0].tolist()

    # 如果你的 Tokenizer 有 decode 方法
    if hasattr(tokenizer, "decode"):
        result_text = tokenizer.decode(generated_ids)
    else:
        # fallback: 有些 tokenizer 叫 decode_batch 或其他名字，这里尽量做常见兼容
        logger.warning("未找到默认的 decode 方法，请根据 MythosTokenizer 实际接口修改。")
        result_text = str(generated_ids)

    print("\n" + "=" * 20 + " 生成结果 " + "=" * 20)
    print(result_text)
    print("=" * 50)


if __name__ == "__main__":
    main()
