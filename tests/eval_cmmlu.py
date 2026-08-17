#!/usr/bin/env python3
"""
OpenMythos CMMLU 评测脚本（loglikelihood 选择题评测）

评测方式：对每个四选一题目，把 "问题 + 选项 + 答案：" 作为 prompt，分别把
A/B/C/D 四个字母作为续写，比较模型给的 log-likelihood，默认取 argmax 作为
预测（--temperature 0）。--temperature > 0 时改为按 softmax(score/T) 在四个
选项上采样（固定随机种子，可复现）。对 0.1B 量级的 base 模型，这比生成式
评测更稳 —— 不存在答案字母采样不出来、正则解析失败的问题（0.1B 模型的
自由生成大概率无法形成可解析的答案）。

数据来自 modelscope/cmmlu（67 个科目，test + dev 两个 split，csv 共约 1MB）。
注意：本脚本**全程离线** —— 数据直接读本地 CSV（data/cmmlu/）或 modelscope
下载缓存里的 zip，不调用 MsDataset/hub 在线接口。原因：当前网络环境下
modelscope/hf-mirror 的在线元数据请求会无限挂起，而 cmmlu 总共只有约 1MB，
一次下载后全部科目都在本地，离线读最稳也最快。模型加载复用
tests/test_infer.py 的 find_latest_checkpoint / load_inference_model。

用法：
    # 全部 67 个科目，5-shot（约 1.15 万题）
    python tests/eval_cmmlu.py

    # 快速小范围验证：2 个科目、每科 5 题、2-shot
    python tests/eval_cmmlu.py --subjects anatomy,logical --max_per_subject 5 --n_shots 2

    # 指定 checkpoint 与循环深度
    python tests/eval_cmmlu.py --checkpoint checkpoints_act/step_0008200.pt --n_loops 8

    # zero-shot
    python tests/eval_cmmlu.py --n_shots 0
"""

import os
import csv
import io
import sys
import json
import zipfile
import argparse
import torch
from loguru import logger

# 评测全程离线：数据是本地 CSV/zip，checkpoint 是本地文件，tokenizer 已缓存。
# 强制 HF 离线模式，否则 from_pretrained 每次启动都会先做一次在线元数据检查
# —— 直连 huggingface.co 在当前网络下的失败路径不稳定（见过挂起数小时，也见过
# 缓存的 httpx session 被关闭后直接抛 RuntimeError）。新机器上 tokenizer 未缓存
# 时，显式用 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 覆盖即可。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 以 `python tests/eval_cmmlu.py` 运行时 sys.path[0] 是 tests/，补上项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_mythos.tokenizer import MythosTokenizer
from tests.test_infer import find_latest_checkpoint, load_inference_model

# modelscope/cmmlu 数据集脚本中的全部 67 个科目（task_list，v1.0.1）
SUBJECTS = [
    "agronomy", "anatomy", "ancient_chinese", "arts", "astronomy",
    "business_ethics", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "clinical_knowledge",
    "college_actuarial_science", "college_education", "college_engineering_hydrology",
    "college_law", "college_mathematics", "college_medical_statistics",
    "college_medicine", "computer_science", "computer_security",
    "conceptual_physics", "construction_project_management", "economics",
    "education", "electrical_engineering", "elementary_chinese",
    "elementary_commonsense", "elementary_information_and_technology",
    "elementary_mathematics", "ethnology", "food_science", "genetics",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_geography", "high_school_mathematics", "high_school_physics",
    "high_school_politics", "human_sexuality", "international_law",
    "journalism", "jurisprudence", "legal_and_moral_basis", "logical",
    "machine_learning", "management", "marketing", "marxist_theory",
    "modern_chinese", "nutrition", "philosophy", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_study", "sociology", "sports_science",
    "traditional_chinese_medicine", "virology", "world_history", "world_religions",
]

CHOICES = ["A", "B", "C", "D"]


def format_question(row: dict, with_answer: bool) -> str:
    """CMMLU 官方风格的题目格式；with_answer=True 用于 few-shot 示例。"""
    s = f"问题：{row['Question']}\n"
    for c in CHOICES:
        s += f"{c}. {row[c]}\n"
    s += "答案："
    if with_answer:
        s += f"{row['Answer']}\n\n"
    return s


def build_prompt(subject: str, dev_rows: list, row: dict, n_shots: int) -> str:
    header = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
    shots = "".join(format_question(r, with_answer=True) for r in dev_rows[:n_shots])
    return header + shots + format_question(row, with_answer=False)


@torch.no_grad()
def score_question(model, tokenizer, prompt: str, device: str, n_loops: int,
                   max_seq_len: int):
    """
    给一道题打分：返回 A/B/C/D 四个续写的 log-likelihood 列表；超长返回 None。

    四个候选共享同一 prompt，只在结尾差 1~2 个 token，因此拼成 batch=4
    一次 forward 完成。右填充不影响结果 —— 因果注意力下真实 token 永远
    看不到后面的 pad，RoPE 又是按位置索引的。

    续写 token 用 prefix-delta 法计算（encode(prompt+letter) 去掉
    encode(prompt) 的前缀），避免手猜字母在前导空白下的分词结果；
    边界发生 merge 时退化为单独编码字母。
    """
    prompt_ids = tokenizer.encode(prompt)
    conts = []
    for letter in CHOICES:
        full = tokenizer.encode(prompt + letter)
        if full[: len(prompt_ids)] == prompt_ids:
            conts.append(full[len(prompt_ids):])
        else:
            conts.append(tokenizer.encode(letter))

    max_cont = max(len(c) for c in conts)
    plen = len(prompt_ids)
    T = plen + max_cont
    if T > max_seq_len:
        return None

    batch = torch.zeros(4, T, dtype=torch.long, device=device)
    target = torch.zeros(4, max_cont, dtype=torch.long, device=device)
    valid = torch.zeros(4, max_cont, dtype=torch.bool, device=device)
    for i, cont in enumerate(conts):
        seq = prompt_ids + cont
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        target[i, : len(cont)] = torch.tensor(cont, dtype=torch.long)
        valid[i, : len(cont)] = True

    with torch.autocast(
        device_type="cuda" if "cuda" in device else "cpu",
        dtype=model.embed.weight.dtype,
    ):
        logits = model(batch, n_loops=n_loops)  # (4, T, V)

    # 只需要续写位置的 logits：位置 plen-1 预测第 1 个续写 token，依此类推。
    # 先切片再做 float32 log_softmax，避免 (4, T, 200k) 的 fp32 显存峰值。
    window = logits[:, plen - 1 : plen - 1 + max_cont, :].float()
    logp = torch.log_softmax(window, dim=-1)
    picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    scores = (picked * valid).sum(dim=-1)  # (4,)
    return scores.cpu().tolist()


_CMMLU_COLUMNS = ("Question", "A", "B", "C", "D", "Answer")


def _find_cmmlu_zip() -> "str | None":
    """在 modelscope 下载缓存里找 cmmlu 数据 zip（含 test/ 与 dev/ 两套 CSV）。"""
    dl_dir = os.path.expanduser("~/.cache/modelscope/hub/datasets/downloads")
    if not os.path.isdir(dl_dir):
        return None
    for fname in os.listdir(dl_dir):
        path = os.path.join(dl_dir, fname)
        if not os.path.isfile(path) or not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
        if "test/anatomy.csv" in names and "dev/anatomy.csv" in names:
            return path
    return None


def load_cmmlu(subject: str, split: str, data_dir: str = "data/cmmlu") -> list:
    """
    读取 cmmlu 的 {split}/{subject}.csv，返回 [{Question, A, B, C, D, Answer}]。

    完全离线：优先读 data_dir 下已解压的 CSV；没有则在 modelscope 下载
    缓存里找到数据 zip 直接读（整个数据集只有约 1MB）。不走 MsDataset/hub
    在线接口 —— 当前网络下它们会无限挂起，且训练脚本注释里也记录过
    同进程多次 MsDataset.load 的补丁递归 bug。
    """
    name = f"{split}/{subject}.csv"
    path = os.path.join(data_dir, name)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        zip_path = _find_cmmlu_zip()
        if zip_path is None:
            raise FileNotFoundError(
                f"找不到 {path}，modelscope 缓存里也没有 cmmlu zip；"
                f"请先下载 modelscope/cmmlu 或解压到 {data_dir}/"
            )
        with zipfile.ZipFile(zip_path) as z:
            with z.open(name) as f:
                rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    # 数据集脚本用 pd.read_csv(header=0, index_col=0)，首列是无名索引列，丢弃
    return [{k: r.get(k, "") for k in _CMMLU_COLUMNS} for r in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenMythos CMMLU loglikelihood 评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="指定 .pt 文件；不指定则从 --checkpoint_dir 取最新")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/home/ranhao/projects/OpenMythos/checkpoints_act",
                        help="checkpoints 目录（--checkpoint 未给时生效）")
    parser.add_argument("--encoder_model_id", type=str,
                        default="Langboat/mengzi-t5-base",
                        help="必须与训练时的分词器一致")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--n_loops", type=int, default=8,
                        help="推理循环深度（与 test_infer.py 一致）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="答案选择温度：0 = 贪心 argmax（确定、可复现）；"
                             ">0 = 按 softmax(score/T) 在四个选项上采样")
    parser.add_argument("--n_shots", type=int, default=5,
                        help="few-shot 示例数（取自 dev split，每科共 5 条）")
    parser.add_argument("--subjects", type=str, default="all",
                        help="逗号分隔的科目名，或 all")
    parser.add_argument("--max_per_subject", type=int, default=0,
                        help="每科最多评多少题（0 = 全部），用于快速验证")
    parser.add_argument("--data_dir", type=str, default="data/cmmlu",
                        help="cmmlu 解压目录（含 test/ dev/ 子目录）；"
                             "文件不存在时自动回退到 modelscope 缓存里的 zip")
    parser.add_argument("--output", type=str, default=None,
                        help="结果 JSON 保存路径；默认 runs/cmmlu_<ckpt>_<shots>shot.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    if args.subjects.strip().lower() == "all":
        subjects = SUBJECTS
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
        unknown = [s for s in subjects if s not in SUBJECTS]
        if unknown:
            logger.error(f"未知科目: {unknown}")
            return

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.error(f"找不到 checkpoint（dir={args.checkpoint_dir}）")
        return

    logger.info("=" * 50)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"科目数: {len(subjects)} | n_shots: {args.n_shots} | "
                f"n_loops: {args.n_loops} | temperature: {args.temperature} | "
                f"device: {device}")
    logger.info("=" * 50)

    tokenizer = MythosTokenizer(model_id=args.encoder_model_id)
    model = load_inference_model(ckpt_path, device)
    max_seq_len = model.cfg.max_seq_len

    if args.temperature > 0:
        # 采样评测固定种子，保证同一 checkpoint 多次评测结果可复现
        torch.manual_seed(42)

    total_correct = total_seen = total_skipped = 0
    subject_stats = {}

    for si, subject in enumerate(subjects):
        try:
            test_rows = load_cmmlu(subject, "test", args.data_dir)
            dev_rows = load_cmmlu(subject, "dev", args.data_dir) if args.n_shots > 0 else []
        except Exception as exc:
            logger.warning(f"[{subject}] 数据加载失败，跳过: {type(exc).__name__}: {exc}")
            continue
        dev_rows = dev_rows[: args.n_shots]
        if args.max_per_subject > 0:
            test_rows = test_rows[: args.max_per_subject]

        correct = seen = skipped = 0
        for row in test_rows:
            answer = str(row.get("Answer", "")).strip().upper()
            if answer not in CHOICES:
                skipped += 1
                continue
            # 超长时逐个减 shot，直到能塞进 max_seq_len
            n_shots = len(dev_rows)
            while True:
                prompt = build_prompt(subject, dev_rows, row, n_shots)
                if (len(tokenizer.encode(prompt)) + 2 <= max_seq_len
                        or n_shots == 0):
                    break
                n_shots -= 1
            scores = score_question(model, tokenizer, prompt, device,
                                    args.n_loops, max_seq_len)
            if scores is None:
                skipped += 1
                continue
            if args.temperature > 0:
                # 温度采样：p ∝ exp(score / T)，在四个选项上抽样
                probs = torch.softmax(
                    torch.tensor(scores) / args.temperature, dim=-1
                )
                pred = CHOICES[torch.multinomial(probs, num_samples=1).item()]
            else:
                pred = CHOICES[max(range(4), key=lambda k: scores[k])]
            correct += pred == answer
            seen += 1
            if seen % 200 == 0:
                logger.info(f"[{si + 1}/{len(subjects)} {subject}] "
                            f"{seen} 题, 当前 acc={correct / seen:.4f}")

        acc = correct / seen if seen else 0.0
        subject_stats[subject] = {"correct": correct, "total": seen,
                                  "skipped": skipped, "acc": acc}
        total_correct += correct
        total_seen += seen
        total_skipped += skipped
        logger.info(f"[{si + 1}/{len(subjects)}] {subject}: "
                    f"acc={acc:.4f} ({correct}/{seen}, skipped={skipped})")

    micro = total_correct / total_seen if total_seen else 0.0
    macro = (sum(s["acc"] for s in subject_stats.values()) / len(subject_stats)
             if subject_stats else 0.0)

    print("\n" + "=" * 50)
    print(f"CMMLU 评测完成 | checkpoint: {os.path.basename(ckpt_path)} | "
          f"{args.n_shots}-shot | n_loops={args.n_loops} | "
          f"temperature={args.temperature}")
    print(f"  micro acc (按题数):   {micro:.4f}  ({total_correct}/{total_seen})")
    print(f"  macro acc (按科目):   {macro:.4f}  ({len(subject_stats)} 科)")
    print(f"  skipped: {total_skipped}")
    print("=" * 50)

    out_path = args.output or os.path.join(
        "runs",
        f"cmmlu_{os.path.splitext(os.path.basename(ckpt_path))[0]}"
        f"_{args.n_shots}shot_loops{args.n_loops}_t{args.temperature}.json",
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "n_shots": args.n_shots,
                "n_loops": args.n_loops,
                "temperature": args.temperature,
                "micro_acc": micro,
                "macro_acc": macro,
                "total_correct": total_correct,
                "total_seen": total_seen,
                "subjects": subject_stats,
            },
            f, ensure_ascii=False, indent=2,
        )
    logger.success(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
