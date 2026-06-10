"""超参数并行搜索脚本（v2）。

支持多组实验同时跑，自动分配 GPU 组。

用法：
  # 贝叶斯优化，并行 2 组
  python hyperparam_search.py --n_trials 30 --gpus 1,2 --parallel 2

  # 网格搜索，并行 2 组
  python hyperparam_search.py --mode grid --grid_params w_evidential,w_neg_reg --gpus 1,2 --parallel 2

  # 查看已有结果
  python hyperparam_search.py --report_only
"""

import argparse
import hashlib
import json
import math
import random
import re
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from pathlib import Path

import optuna

# ─── 搜索空间定义 ───────────────────────────────────────────────

SEARCH_SPACE = {
    "w_evidential":          {"type": "log_uniform", "low": 1e-4, "high": 1e-1},
    "w_neg_reg":             {"type": "log_uniform", "low": 1e-4, "high": 1e-1},
    "w_orth":                {"type": "log_uniform", "low": 1e-3, "high": 1.0},
    "w_uncertainty_reg":     {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
    "w_mil":                 {"type": "log_uniform", "low": 1e-4, "high": 1e-1},
    "log_sigma_min":         {"type": "uniform",     "low": -3.0, "high": -0.5},
    "log_sigma_max":         {"type": "uniform",     "low": 2.0,  "high": 6.0},
    "fusion_temperature":    {"type": "uniform",     "low": 0.5,  "high": 3.0},
}

GRID_VALUES = {
    "w_evidential":          [1e-3, 1e-2, 5e-2],
    "w_neg_reg":             [1e-3, 1e-2, 5e-2],
    "w_orth":                [1e-2, 0.1, 0.5],
    "w_uncertainty_reg":     [1e-4, 1e-3, 1e-2],
    "w_mil":                 [1e-3, 1e-2, 5e-2],
}

FIXED_PARAMS = {
    "epochs": 3,
    "log_sigma_min": -1.5,
    "log_sigma_max": 4.0,
    "fusion_temperature": 1.5,
    "anneal_warmup_epochs": 0,
}

RESULTS_DIR = Path("search_results")
RESULTS_FILE = RESULTS_DIR / "results.jsonl"
BEST_FILE = RESULTS_DIR / "best.json"

_write_lock = threading.Lock()
_id_counter = 0
_id_lock = threading.Lock()


def _alloc_trial_id() -> int:
    """线程安全分配递增 trial_id。"""
    global _id_counter
    with _id_lock:
        tid = _id_counter
        _id_counter += 1
    return tid


def params_fingerprint(params: dict) -> str:
    """参数组合的稳定指纹（用于去重）。"""
    # 只取搜索空间内的参数，忽略浮点精度差异
    search_keys = sorted(k for k in params if k in SEARCH_SPACE)
    raw = json.dumps({k: params[k] for k in search_keys}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def fill_fixed_params(params: dict) -> dict:
    """补齐固定参数，不覆盖已经采样的搜索参数。"""
    full_params = dict(params)
    for k, v in FIXED_PARAMS.items():
        if k not in full_params:
            full_params[k] = v
    return full_params


def find_free_port() -> int:
    """获取本机当前可用端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def is_successful_result(result: dict) -> bool:
    """判断一次试验是否成功产出可用指标。"""
    if result.get("status") == "failed":
        return False
    return bool(result.get("all_epochs")) and result.get("best_t2v_r1", 0) > 0


def partition_gpus(gpus: str, n_groups: int) -> list[str]:
    """将 GPU 列表均分成 n_groups 组。

    Args:
        gpus: 逗号分隔的 GPU ID，如 "1,2,3,4"。
        n_groups: 分组数。

    Returns:
        GPU 组列表，如 ["1,2", "3,4"]。
    """
    gpu_list = gpus.split(",")
    n_groups = max(1, min(n_groups, len(gpu_list)))
    group_size = len(gpu_list) // n_groups
    remainder = len(gpu_list) % n_groups

    groups = []
    start = 0
    for i in range(n_groups):
        size = group_size + (1 if i < remainder else 0)
        groups.append(",".join(gpu_list[start:start + size]))
        start += size
    return groups


def sample_params(trial: optuna.Trial) -> dict:
    """从 Optuna 搜索空间采样参数。"""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "log_uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif spec["type"] == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
    return params


def sample_random(rng: random.Random) -> dict:
    """随机采样一组参数。"""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "log_uniform":
            params[name] = math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
        elif spec["type"] == "uniform":
            params[name] = rng.uniform(spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            params[name] = rng.choice(spec["choices"])
    return params


def grid_combinations(selected: list[str] | None = None) -> list[dict]:
    """生成网格搜索的参数组合。"""
    if selected is None:
        selected = list(GRID_VALUES.keys())
    keys = [k for k in selected if k in GRID_VALUES]
    vals = [GRID_VALUES[k] for k in keys]
    combos = []
    for combo in product(*vals):
        params = dict(zip(keys, combo))
        for k, v in FIXED_PARAMS.items():
            if k not in params:
                params[k] = v
        combos.append(params)
    return combos


def run_trial(params: dict, trial_id: int, gpus: str, epochs: int = 3) -> dict:
    """运行一次训练，返回结果。"""
    trial_dir = RESULTS_DIR / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    script_path = trial_dir / "train.sh"
    log_path = trial_dir / "train.log"

    w_evidential = params.get("w_evidential", 1e-2)
    w_neg_reg = params.get("w_neg_reg", 1e-2)
    w_orth = params.get("w_orth", 0.1)
    w_uncertainty_reg = params.get("w_uncertainty_reg", 1e-3)
    w_mil = params.get("w_mil", 1e-2)
    log_sigma_min = params.get("log_sigma_min", -1.5)
    log_sigma_max = params.get("log_sigma_max", 4.0)
    fusion_temperature = params.get("fusion_temperature", 1.5)
    anneal_warmup_epochs = params.get("anneal_warmup_epochs", 0)

    run_id = f"search_{trial_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = f"ckpts/ckpt_msrvtt_{run_id}"
    n_gpus = len(gpus.split(","))
    port = find_free_port()

    # main_task_retrieval.py 会先将 --batch_size 除以 accum_steps。
    # 固定命令行 batch_size=256，按 GPU 数调整 accum，使每卡 micro-batch 不超过 64。
    micro_batch_per_gpu = 64
    batch_size = 256
    accum_steps = max(1, batch_size // (micro_batch_per_gpu * n_gpus))

    script_content = f"""#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=/data2/hxj/data/MSRVTT
RUN_ID="{run_id}"
OUTPUT_DIR="{output_dir}"

CUDA_VISIBLE_DEVICES="{gpus}" \\
    torchrun --nproc_per_node={n_gpus} --master_addr=127.0.0.9 --master_port={port} \\
    main_task_retrieval.py \\
    --do_train --num_thread_reader=8 --epochs={epochs} \\
    --batch_size={batch_size} --gradient_accumulation_steps={accum_steps} --n_display=50 \\
    --train_csv "${{DATA_PATH}}/csv/MSRVTT_train.9k.csv" \\
    --val_csv "${{DATA_PATH}}/csv/MSRVTT_JSFUSION_test.csv" \\
    --data_path "${{DATA_PATH}}/annotation/MSRVTT_v2.json" \\
    --features_path "${{DATA_PATH}}/videos/compressed_videos/msrvtt_224_12fps/" \\
    --output_dir "${{OUTPUT_DIR}}" \\
    --lr 1e-4 --max_words 32 --max_frames 8 --batch_size_val 16 \\
    --datatype msrvtt --expand_msrvtt_sentences \\
    --feature_framerate 1 --coef_lr 1e-3 \\
    --freeze_layer_num 0 --slice_framepos 3 \\
    --loose_type --linear_patch 2d --sim_header seqTransf \\
    --strategy 2 \\
    --pretrained_clip_name ViT-B/16 \\
    --extra_video_cls_num 2 \\
    --extra_text_cls_num 2 \\
    --n_video_embeddings 7 \\
    --n_text_embeddings 7 \\
    --mamba_lr_ratio 0.1 \\
    --uncertainty_text_head text \\
    --log_sigma_min {log_sigma_min} \\
    --log_sigma_max {log_sigma_max} \\
    --w_mil {w_mil} \\
    --w_evidential {w_evidential} \\
    --w_neg_reg {w_neg_reg} \\
    --w_orth {w_orth} \\
    --w_uncertainty_reg {w_uncertainty_reg} \\
    --gate_log_interval 100 \\
    --log_moe_weights \\
    --fusion_mode prob_mos \\
    --w_query_sim 0.5 \\
    --fusion_temperature {fusion_temperature} \\
    --rope_mode 2d \\
    --use_ada_norm \\
    --anneal_warmup_epochs {anneal_warmup_epochs} \\
    --experiment_desc "search_trial_{trial_id}"
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    param_summary = _format_params(params)
    print(f"[Trial {trial_id}] 启动 | GPUs={gpus} | batch={batch_size}×{accum_steps} | {param_summary}")

    start_time = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            ["bash", str(script_path)],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )
    elapsed = time.time() - start_time

    result = parse_log(log_path)
    result["trial_id"] = trial_id
    result["params"] = params
    result["return_code"] = proc.returncode
    result["elapsed_seconds"] = elapsed
    result["output_dir"] = output_dir
    result["gpus"] = gpus
    result["batch_size"] = batch_size
    result["accum_steps"] = accum_steps
    result["status"] = "completed" if proc.returncode == 0 and result["all_epochs"] else "failed"

    # 线程安全写入
    with _write_lock:
        with open(RESULTS_DIR / f"trial_{trial_id}" / "result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    r1 = result.get("best_t2v_r1", 0)
    print(f"[Trial {trial_id}] 完成 | R@1={r1:.1f} | {elapsed/3600:.1f}h | {param_summary}")
    return result


def _format_params(params: dict) -> str:
    """格式化参数摘要。"""
    return " ".join(
        f"{k}={v:.1e}" if isinstance(v, float) and abs(v) < 0.01 else f"{k}={v}"
        for k, v in params.items() if k not in FIXED_PARAMS
    )


def parse_log(log_path: Path) -> dict:
    """从日志提取最佳指标。

    策略：按行顺序匹配 T2V / V2T 成对出现，不依赖 "Retrieval Evaluation" 分块。
    """
    result = {"best_t2v_r1": 0.0, "best_v2t_r1": 0.0, "best_epoch": 0, "all_epochs": []}
    if not log_path.exists():
        return result

    t2v_pat = re.compile(r"T2V \| R@1:\s+([\d.]+)\s+R@5:\s+([\d.]+)\s+R@10:\s+([\d.]+)")
    v2t_pat = re.compile(r"V2T \| R@1:\s+([\d.]+)\s+R@5:\s+([\d.]+)\s+R@10:\s+([\d.]+)")

    # 按行扫描，收集 T2V/V2T 成对指标
    pending_t2v = None
    epoch_count = 0

    with open(log_path, "r") as f:
        for line in f:
            t2v_match = t2v_pat.search(line)
            if t2v_match:
                pending_t2v = {
                    "t2v_r1": float(t2v_match.group(1)),
                    "t2v_r5": float(t2v_match.group(2)),
                    "t2v_r10": float(t2v_match.group(3)),
                }
                continue

            v2t_match = v2t_pat.search(line)
            if v2t_match and pending_t2v is not None:
                epoch_count += 1
                result["all_epochs"].append({
                    "epoch": epoch_count,
                    **pending_t2v,
                    "v2t_r1": float(v2t_match.group(1)),
                    "v2t_r5": float(v2t_match.group(2)),
                    "v2t_r10": float(v2t_match.group(3)),
                })
                pending_t2v = None

    if result["all_epochs"]:
        best = max(result["all_epochs"], key=lambda x: x["t2v_r1"])
        result["best_t2v_r1"] = best["t2v_r1"]
        result["best_v2t_r1"] = best["v2t_r1"]
        result["best_epoch"] = best["epoch"]
    return result


def load_existing_results() -> list[dict]:
    """加载已有结果。"""
    results = []
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().strip().split("\n"):
            if line:
                results.append(json.loads(line))
    return results


def load_existing_fingerprints() -> set[str]:
    """已有成功结果的参数指纹集合（用于去重）。"""
    return {
        params_fingerprint(r["params"])
        for r in load_existing_results()
        if "params" in r and is_successful_result(r)
    }


def save_best(results: list[dict]):
    """保存并打印最优结果。"""
    successful_results = [r for r in results if is_successful_result(r)]
    if not successful_results:
        print("暂无成功完成且包含评估指标的试验。")
        return
    best = max(successful_results, key=lambda x: x.get("best_t2v_r1", 0))
    with open(BEST_FILE, "w") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"最优配置 | Trial {best['trial_id']} | T2V R@1={best['best_t2v_r1']:.1f} | V2T R@1={best['best_v2t_r1']:.1f}")
    print(f"参数: {json.dumps(best['params'], indent=2)}")
    print(f"{'='*70}")


def print_report():
    """打印搜索报告。"""
    results = load_existing_results()
    if not results:
        print("暂无搜索结果。")
        return

    successful_results = [r for r in results if is_successful_result(r)]
    failed_count = len(results) - len(successful_results)
    successful_results.sort(key=lambda x: x.get("best_t2v_r1", 0), reverse=True)

    print(f"\n{'='*95}")
    print(f"超参数搜索报告 | 成功 {len(successful_results)} 次 | 失败/无指标 {failed_count} 次")
    print(f"{'='*95}")
    print(f"{'Rank':>4} {'Trial':>6} {'T2V R@1':>8} {'V2T R@1':>8} {'Ep':>3} {'Time':>6} {'Batch':>10} | 参数")
    print("-" * 95)

    for rank, r in enumerate(successful_results, 1):
        t = r.get("trial_id", "?")
        t2v = r.get("best_t2v_r1", 0)
        v2t = r.get("best_v2t_r1", 0)
        ep = r.get("best_epoch", 0)
        hours = r.get("elapsed_seconds", 0) / 3600
        bs = r.get("batch_size", "?")
        acc = r.get("accum_steps", "?")
        params = r.get("params", {})
        param_str = _format_params(params)
        print(f"{rank:>4} {t:>6} {t2v:>8.1f} {v2t:>8.1f} {ep:>3} {hours:>5.1f}h {bs}×{acc:>8} | {param_str}")

    save_best(successful_results)


def run_parallel_trials(trial_specs: list[tuple[dict, int]], gpu_groups: list[str],
                        epochs: int) -> list[dict]:
    """并行执行多组试验。"""
    results = []
    n_workers = len(gpu_groups)
    gpu_semaphore = threading.Semaphore(n_workers)
    gpu_pool = list(gpu_groups)
    gpu_lock = threading.Lock()

    def run_one(spec):
        params, trial_id = spec
        gpu_semaphore.acquire()
        try:
            with gpu_lock:
                gpus = gpu_pool.pop(0)
            try:
                return run_trial(params, trial_id, gpus, epochs)
            finally:
                with gpu_lock:
                    gpu_pool.append(gpus)
        finally:
            gpu_semaphore.release()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one, spec): spec for spec in trial_specs}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                spec = futures[future]
                print(f"[Trial {spec[1]}] 异常: {e}")

    return results


def build_bayesian_specs(study: optuna.Study, batch_size: int, existing_fps: set[str]) -> list[tuple[dict, int, optuna.Trial]]:
    """构造一批未重复的 Optuna 试验。"""
    specs = []
    attempts = 0
    max_attempts = max(batch_size * 10, 20)
    while len(specs) < batch_size and attempts < max_attempts:
        attempts += 1
        trial = study.ask()
        params = fill_fixed_params(sample_params(trial))
        fp = params_fingerprint(params)
        if fp in existing_fps:
            study.tell(trial, optuna.trial.TrialState.PRUNED)
            continue
        existing_fps.add(fp)
        tid = _alloc_trial_id()
        specs.append((params, tid, trial))
    return specs


def main():
    parser = argparse.ArgumentParser(description="UATVR 超参数并行搜索 v2")
    parser.add_argument("--mode", choices=["bayesian", "grid", "random"], default="bayesian")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="贝叶斯/随机搜索的试验次数")
    parser.add_argument("--gpus", type=str, default="1,2,3,4",
                        help="可用 GPU，逗号分隔")
    parser.add_argument("--parallel", type=int, default=2,
                        help="同时跑几组试验（GPU 自动均分）")
    parser.add_argument("--epochs", type=int, default=3,
                        help="每次试验的训练 epoch 数")
    parser.add_argument("--grid_params", type=str, default=None,
                        help="网格搜索的参数名，逗号分隔")
    parser.add_argument("--report_only", action="store_true",
                        help="仅查看已有结果")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.report_only:
        print_report()
        return

    # GPU 分组
    n_groups = args.parallel
    gpu_groups = partition_gpus(args.gpus, n_groups)
    group_sizes = {len(group.split(",")) for group in gpu_groups}
    if len(group_sizes) != 1:
        raise ValueError(f"GPU 分组不均匀：{gpu_groups}。请调整 --gpus 或 --parallel，避免不同 trial 的 batch 结构不可比。")

    print(f"模式: {args.mode}")
    print(f"GPU: {args.gpus} → {n_groups} 组并行: {gpu_groups}")
    print(f"Epochs/trial: {args.epochs}")

    # 已有结果的指纹集合（用于去重）
    existing_fps = load_existing_fingerprints()

    # 已有最大 trial_id，用于续接递增 ID
    existing_results = load_existing_results()
    existing_ids = {r["trial_id"] for r in existing_results}
    global _id_counter
    _id_counter = max(existing_ids, default=-1) + 1

    if args.mode == "bayesian":
        # Optuna SQLite 自动持久化，load_if_exists 会恢复所有历史 trial
        # 不需要手动 enqueue_trial
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            study_name="uatvr_search",
            storage=f"sqlite:///{RESULTS_DIR}/optuna.db",
            load_if_exists=True,
        )

        existing_trial_count = len(study.trials)
        print(f"Optuna 已有 {existing_trial_count} 个 trial")

        # 分批并行
        remaining = args.n_trials
        while remaining > 0:
            batch_size = min(n_groups, remaining)
            specs = build_bayesian_specs(study, batch_size, existing_fps)
            if not specs:
                print("未能采样到新的参数组合，提前结束。")
                break

            trial_specs = [(p, t) for p, t, _ in specs]
            results = run_parallel_trials(trial_specs, gpu_groups, args.epochs)

            for params, tid, trial in specs:
                matching = [r for r in results if r["trial_id"] == tid]
                if matching:
                    r1 = matching[0].get("best_t2v_r1", 0.0)
                    if is_successful_result(matching[0]):
                        study.tell(trial, r1)
                    else:
                        study.tell(trial, optuna.trial.TrialState.FAIL)
                else:
                    study.tell(trial, optuna.trial.TrialState.FAIL)

            remaining -= len(specs)
            done = args.n_trials - remaining
            print(f"\n--- 进度: {done}/{args.n_trials} ---\n")

        completed_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
        ]
        if completed_trials:
            print(f"\n最优 R@1: {study.best_value:.1f}")
            print(f"最优参数: {json.dumps(study.best_params, indent=2)}")
        else:
            print("\n暂无成功完成的 Optuna trial。")

    elif args.mode == "grid":
        selected = args.grid_params.split(",") if args.grid_params else None
        combos = grid_combinations(selected)
        print(f"网格组合数: {len(combos)}")

        # 用指纹去重，避免搜索空间变化导致的 ID 错位
        todo = []
        for params in combos:
            params = fill_fixed_params(params)
            fp = params_fingerprint(params)
            if fp not in existing_fps:
                tid = _alloc_trial_id()
                todo.append((params, tid))
        print(f"待执行: {len(todo)}（已完成: {len(combos) - len(todo)}）")

        for batch_start in range(0, len(todo), n_groups):
            batch = todo[batch_start:batch_start + n_groups]
            run_parallel_trials(batch, gpu_groups, args.epochs)
            done = min(batch_start + n_groups, len(todo))
            print(f"\n--- 进度: {done}/{len(todo)} ---\n")

    elif args.mode == "random":
        rng = random.Random(args.seed)
        todo = []
        for _ in range(args.n_trials):
            params = fill_fixed_params(sample_random(rng))
            fp = params_fingerprint(params)
            if fp not in existing_fps:
                tid = _alloc_trial_id()
                todo.append((params, tid))
                existing_fps.add(fp)  # 防止同批内随机采样重复
        print(f"待执行: {len(todo)}")

        for batch_start in range(0, len(todo), n_groups):
            batch = todo[batch_start:batch_start + n_groups]
            run_parallel_trials(batch, gpu_groups, args.epochs)
            done = min(batch_start + n_groups, len(todo))
            print(f"\n--- 进度: {done}/{len(todo)} ---\n")

    print_report()


if __name__ == "__main__":
    main()
