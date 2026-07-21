# RSPR End-to-End Runtime and Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 RSPR 核心训练固化为从 OpenAI CLIP 权重开始的单次端到端作业，并让训练/评估统一使用 `tvr` 环境，同时输出紧凑、可验收的 RSPR 数值诊断。

**Architecture:** 新建一个只负责解释器选择、启动前校验和运行时摘要的 Bash helper，由训练 controller/worker 与评估入口共同调用。模型侧只生成 detached 的 loss、配对不确定性和 `logvar` 标量摘要，训练循环负责 step 格式化与 epoch 加权聚合；端到端参数可训练性继续由现有 CLIP 分层冻结与 RSPR freeze contract 共同决定。

**Tech Stack:** Bash、Python 3.11、PyTorch、torch.distributed、pytest、Ruff、Git

## Global Constraints

- 规格事实源为 `docs/superpowers/specs/2026-07-22-rspr-training-runtime-and-logging-design.md`。
- 规范训练从 OpenAI CLIP `ViT-B/16` 权重开始，不加载历史 UATVR 全模型 checkpoint，也不使用固定 Stage A/B。
- 规范训练是同一个 optimizer 和学习率计划内连续 5 epochs；`RSPR_WARMUP_EPOCHS=1` 只线性启用 rank/anchor loss，不切换冻结状态或重建 optimizer。
- 规范端到端配置固定 `RSPR_FREEZE_CLIP=0`、`RSPR_FREEZE_DSA=0`、`FREEZE_LAYER_NUM=8`；DSA、WTI、RSPR 与 CLIP 后 4 个 block 从第一个 optimizer step 起可训练。
- A0–A8 必须共享相同 CLIP 起点、trusted split、总 epoch 数和优化日程，只允许消融矩阵定义的差异。
- 默认运行时固定为 `TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python` 与 `TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun`，允许调用者用同名变量成对覆盖。
- controller 必须在 detached worker 创建前校验运行时；训练 worker 和独立评估入口也必须在 split builder 或 `torchrun` 前校验。
- 总损失保持 `L_DSA + λp L_prob + λr warmup L_rank + λa warmup L_anchor`，日志修改不得改变梯度或 loss 数值。
- step/epoch 诊断只能读取当前 forward 已产生的 detached tensor，不得增加 forward、随机采样或概率匹配。
- `off/legacy` 模式不得输出 RSPR 零值或占位诊断；完整参数继续以 `experiment_manifest.json` 为事实源。
- 本计划只完成工程实现和 CPU/静态验收，不启动真实训练；20-step GPU smoke 是后续独立执行步骤。
- 所有 Python 测试和 Ruff 命令使用 `/home/xujie/.conda/envs/tvr/bin/python` 或该环境内的 `ruff`。
- 工作树含用户已有修改、删除和未跟踪文件；每次提交只能暂存当前任务列出的路径，不得清理或覆盖无关改动。
- 每个实现任务严格执行 RED → GREEN → REFACTOR；没有观察到预期失败，不进入生产代码修改。

## File Responsibility Map

- `scripts/tvr_runtime.sh`：唯一的 `tvr` Python/torchrun 默认值、覆盖检测、preflight 和摘要输出。
- `run_train_msrvtt_bg.sh`：controller/worker 生命周期、端到端默认训练参数、split builder 与分布式训练调用。
- `eval.sh`：独立评估 preflight、split builder 和单卡 `torchrun` 调用。
- `modules/modeling.py`：每次 RSPR forward 的 detached loss、配对不确定性和 `logvar` 摘要。
- `main_task_retrieval.py`：CLIP 分层冻结接口、step 日志格式、epoch 诊断聚合、启动参数分组。
- `tests/test_run_train_msrvtt_bg.py`、`tests/test_eval_rspr_entrypoint.py`：Shell runtime 与入口契约。
- `tests/test_rspr_model_integration.py`、`tests/test_rspr_cli.py`：端到端可训练性、loss 不变性与日志契约。
- `tests/test_rspr_documentation_contract.py`：三份 RSPR 事实文档的单次端到端训练语义。

---

### Task 1: 统一训练与评估的 `tvr` 运行时

**Files:**
- Create: `scripts/tvr_runtime.sh`
- Modify: `run_train_msrvtt_bg.sh:1-75,210-220`
- Modify: `eval.sh:20-75,130-145`
- Modify: `tests/test_run_train_msrvtt_bg.py:1-710`
- Modify: `tests/test_eval_rspr_entrypoint.py:1-184`

**Interfaces:**
- Produces: `tvr_load_runtime() -> exit status 0`
- Produces: `tvr_validate_runtime() -> exit status 0 or 2`
- Produces: `tvr_log_runtime() -> stdout line "[Runtime] python=<path> torchrun=<path>"`
- Consumes: optional environment variables `TVR_PYTHON` and `TVR_TORCHRUN`
- Guarantees: successful return leaves non-empty `TVR_PYTHON`, `TVR_TORCHRUN`, `TVR_PYTHON_OVERRIDDEN`, and `TVR_TORCHRUN_OVERRIDDEN` shell variables

- [ ] **Step 1: 为训练入口写 runtime 默认值、覆盖和 fail-fast 测试**

在 `_copy_script()` 中同时复制 `scripts/tvr_runtime.sh`，并让 `_environment()` 显式把两个 fake executable 传给入口：

```python
TVR_RUNTIME = REPOSITORY / "scripts" / "tvr_runtime.sh"


def _copy_script(tmp_path: Path) -> Path:
    copied = tmp_path / SCRIPT.name
    config_directory = tmp_path / "scripts"
    config_directory.mkdir()
    shutil.copy2(SCRIPT, copied)
    shutil.copy2(RSPR_CONFIG, config_directory / RSPR_CONFIG.name)
    shutil.copy2(TVR_RUNTIME, config_directory / TVR_RUNTIME.name)
    return copied


def _environment(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TVR_PYTHON": str(fake_bin / "python3"),
            "TVR_TORCHRUN": str(fake_bin / "torchrun"),
            "DATA_PATH": "/dataset",
            "OUTPUT_DIR": str(tmp_path / "checkpoints"),
            "RUN_ID": "unit-run",
            "EXPERIMENT_PROFILE": "default",
            "CUDA_VISIBLE_DEVICES": "3,5",
            "NPROC": "2",
            "CLIP_GRADIENT_CHECKPOINTING": "1",
        }
    )
    return environment
```

先用不执行 preflight 的 source 测试锁定默认绝对路径：

```python
def test_runtime_helper_defaults_to_tvr_environment():
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{TVR_RUNTIME}"; '
                "unset TVR_PYTHON TVR_TORCHRUN; "
                "tvr_load_runtime; "
                'printf "%s\\n%s\\n" "$TVR_PYTHON" "$TVR_TORCHRUN"'
            ),
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "/home/xujie/.conda/envs/tvr/bin/python",
        "/home/xujie/.conda/envs/tvr/bin/torchrun",
    ]
```

再锁定“只覆盖一个路径”会触发同环境目录检查：

```python
def test_runtime_helper_rejects_one_sided_override(tmp_path):
    fake_python = tmp_path / "python"
    _write_executable(fake_python, "exit 0\n")
    result = subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{TVR_RUNTIME}"; '
                f'TVR_PYTHON="{fake_python}"; unset TVR_TORCHRUN; '
                "tvr_load_runtime; tvr_validate_runtime"
            ),
        ],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "must use the same environment directory" in result.stderr
    assert "override both variables" in result.stderr
```

新增测试覆盖：默认常量文本、controller 传递两个已解析路径、不可执行路径、依赖导入失败均在 `setsid` 前退出 2。fake Python 必须区分 preflight 与 split builder：

```python
def _write_fake_python(path: Path, split_body: str = "exit 0") -> None:
    _write_executable(
        path,
        'if [[ "${1:-}" == "-c" ]]; then exit "${FAKE_IMPORT_STATUS:-0}"; fi\n'
        + split_body
        + "\n",
    )


def test_controller_rejects_runtime_import_failure_before_detaching(tmp_path):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    setsid_marker = tmp_path / "setsid.called"
    _write_fake_python(fake_bin / "python3")
    _write_executable(fake_bin / "torchrun", "exit 0\n")
    _write_executable(fake_bin / "setsid", 'touch "$SETSID_MARKER"\n')
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {"FAKE_IMPORT_STATUS": "19", "SETSID_MARKER": str(setsid_marker)}
    )

    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=environment,
        text=True, capture_output=True, check=False,
    )

    assert result.returncode == 2
    assert "cannot import torch and cv2" in result.stderr
    assert "TVR_PYTHON" in result.stderr
    assert not setsid_marker.exists()
```

更新 controller 的 `setsid_args` 期望，在 `RUN_ID` 后包含：

```python
f"TVR_PYTHON={fake_bin / 'python3'}",
f"TVR_TORCHRUN={fake_bin / 'torchrun'}",
```

- [ ] **Step 2: 为评估入口写相同 runtime 与调用路径测试**

让评估测试的 `_environment()` 同样设置两个覆盖变量；MSR-VTT fake Python 用上一步的 `-c` 分支。新增以下断言：

```python
def test_eval_rejects_non_executable_runtime_before_split_or_torchrun(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    split_marker = tmp_path / "split.called"
    torchrun_marker = tmp_path / "torchrun.called"
    python_path = fake_bin / "python3"
    python_path.write_text("not executable")
    _write_executable(fake_bin / "torchrun", 'touch "$TORCHRUN_MARKER"\n')
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {
            "DATATYPE": "msrvtt",
            "SPLIT_MARKER": str(split_marker),
            "TORCHRUN_MARKER": str(torchrun_marker),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=REPOSITORY, env=environment,
        text=True, capture_output=True, check=False,
    )

    assert result.returncode == 2
    assert "TVR_PYTHON is not executable" in result.stderr
    assert not split_marker.exists()
    assert not torchrun_marker.exists()
```

现有成功路径还要断言 stdout 含完整 `[Runtime]` 行，且 split marker 由 `TVR_PYTHON` 创建、torchrun arguments 由 `TVR_TORCHRUN` 创建。

- [ ] **Step 3: 运行入口测试并确认 RED**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_run_train_msrvtt_bg.py tests/test_eval_rspr_entrypoint.py
```

Expected: FAIL；首个失败是 `scripts/tvr_runtime.sh` 不存在或入口未传递/使用 `TVR_PYTHON`、`TVR_TORCHRUN`，且不是测试 fixture 自身错误。

- [ ] **Step 4: 实现共享 runtime helper**

创建 `scripts/tvr_runtime.sh`：

```bash
#!/usr/bin/env bash

tvr_load_runtime() {
    TVR_PYTHON_OVERRIDDEN=0
    TVR_TORCHRUN_OVERRIDDEN=0
    if [[ ${TVR_PYTHON+x} == x ]]; then
        TVR_PYTHON_OVERRIDDEN=1
    else
        TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python
    fi
    if [[ ${TVR_TORCHRUN+x} == x ]]; then
        TVR_TORCHRUN_OVERRIDDEN=1
    else
        TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun
    fi
}


tvr_validate_runtime() {
    if [[ ! -x "${TVR_PYTHON}" ]]; then
        echo "TVR_PYTHON is not executable: ${TVR_PYTHON}" >&2
        return 2
    fi
    if [[ ! -x "${TVR_TORCHRUN}" ]]; then
        echo "TVR_TORCHRUN is not executable: ${TVR_TORCHRUN}" >&2
        return 2
    fi
    if (( ! (TVR_PYTHON_OVERRIDDEN && TVR_TORCHRUN_OVERRIDDEN) )) \
        && [[ "$(dirname -- "${TVR_PYTHON}")" != "$(dirname -- "${TVR_TORCHRUN}")" ]]; then
        echo "TVR_PYTHON and TVR_TORCHRUN must use the same environment directory; override both variables to opt out" >&2
        return 2
    fi
    if ! "${TVR_PYTHON}" -c 'import torch; import cv2' >/dev/null 2>&1; then
        echo "TVR_PYTHON cannot import torch and cv2: ${TVR_PYTHON}; override TVR_PYTHON and TVR_TORCHRUN together" >&2
        return 2
    fi
}


tvr_log_runtime() {
    echo "[Runtime] python=${TVR_PYTHON} torchrun=${TVR_TORCHRUN}"
}
```

- [ ] **Step 5: 接入训练 controller/worker 与评估入口**

两个入口都 source helper。controller 在任何 `mkdir`/`setsid` 前执行：

```bash
tvr_load_runtime
tvr_validate_runtime || return $?
```

controller 的 detached 环境显式增加：

```bash
TVR_PYTHON="${TVR_PYTHON}" \
TVR_TORCHRUN="${TVR_TORCHRUN}" \
```

worker 再次 load/validate 后调用 `tvr_log_runtime`，并替换命令：

```bash
"${TVR_PYTHON}" "${ROOT_DIR}/scripts/build_msrvtt_trusted_split.py" ...
"${TVR_TORCHRUN}" --nproc_per_node="${NPROC}" ...
```

`eval.sh` 在参数校验后、创建日志目录和 split 之前 load/validate/log，MSR-VTT split 与末尾 torchrun 同样替换为两个变量。

- [ ] **Step 6: 运行入口测试并确认 GREEN**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_run_train_msrvtt_bg.py tests/test_eval_rspr_entrypoint.py
bash -n scripts/tvr_runtime.sh run_train_msrvtt_bg.sh eval.sh
```

Expected: 全部 PASS；Bash 无语法错误。

- [ ] **Step 7: 提交共享 runtime**

```bash
git add scripts/tvr_runtime.sh run_train_msrvtt_bg.sh eval.sh tests/test_run_train_msrvtt_bg.py tests/test_eval_rspr_entrypoint.py
git commit -m "fix: pin rspr entrypoints to tvr runtime"
```

### Task 2: 固化单次端到端训练与 optimizer 参数契约

**Files:**
- Modify: `run_train_msrvtt_bg.sh:80-110,225-240`
- Modify: `main_task_retrieval.py:620-670,1454-1462`
- Modify: `tests/test_run_train_msrvtt_bg.py:330-650`
- Modify: `tests/test_rspr_model_integration.py:540-620`

**Interfaces:**
- Produces: `apply_clip_layer_freeze_contract(model: nn.Module, args: Namespace) -> None`
- Consumes: `args.freeze_layer_num` in `[-1, 12]` and `args.linear_patch`
- Guarantees: `freeze_layer_num=8` freezes CLIP blocks 0–7 while leaving blocks 8–11 and existing norm/projection exceptions trainable
- Composes with: `apply_rspr_freeze_contract(model, args)`; canonical `rspr_freeze_clip=False`, `rspr_freeze_dsa=False` leaves DSA/WTI/RSPR trainable

- [ ] **Step 1: 写端到端默认命令和参数可训练性测试**

把 worker 完整参数期望中的 `--freeze_layer_num` 值从 `0` 改为 `8`，并增加：

```python
assert "--init_model" not in torchrun_args
assert "--resume_model" not in torchrun_args
assert "--rspr_freeze_clip" not in torchrun_args
assert "--rspr_freeze_dsa" not in torchrun_args
```

在模型集成测试中构造 12 层 CLIP，并验证同一个 optimizer 参数集合：

```python
def _end_to_end_harness():
    model = _freeze_harness()
    model.clip = nn.Module()
    model.clip.transformer = nn.Module()
    model.clip.transformer.resblocks = nn.ModuleList(
        nn.Linear(4, 4) for _ in range(12)
    )
    model.clip.ln_final = nn.LayerNorm(4)
    return model


def test_end_to_end_contract_optimizes_clip_tail_dsa_wti_and_rspr():
    model = _end_to_end_harness()
    args = SimpleNamespace(
        freeze_layer_num=8,
        linear_patch="2d",
        rspr_mode="stochastic",
        rspr_freeze_clip=False,
        rspr_freeze_dsa=False,
    )

    main_task_retrieval.apply_clip_layer_freeze_contract(model, args)
    main_task_retrieval.apply_rspr_freeze_contract(model, args)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.1,
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert optimized_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert not any(name.startswith("clip.transformer.resblocks.7.") for name in trainable)
    assert any(name.startswith("clip.transformer.resblocks.8.") for name in trainable)
    assert any(name.startswith("transformerClip.") for name in trainable)
    assert any(name.startswith("text_weight_fc.") for name in trainable)
    assert any(name.startswith("video_weight_fc.") for name in trainable)
    assert any(name.startswith("rspr.") for name in trainable)
```

- [ ] **Step 2: 运行定向测试并确认 RED**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_run_train_msrvtt_bg.py::test_worker_runs_split_builder_and_torchrun_without_recursing tests/test_rspr_model_integration.py::test_end_to_end_contract_optimizes_clip_tail_dsa_wti_and_rspr
```

Expected: FAIL；shell 仍传 `freeze_layer_num=0`，且 `apply_clip_layer_freeze_contract` 尚不存在。

- [ ] **Step 3: 提取 CLIP 分层冻结函数并设置 shell 默认值**

在 `main_task_retrieval.py` 中加入：

```python
def apply_clip_layer_freeze_contract(model, args):
    if not hasattr(model, "clip") or args.freeze_layer_num <= -1:
        return
    for name, parameter in model.clip.named_parameters():
        if not _should_keep_clip_parameter_trainable(name, args):
            parameter.requires_grad = False
```

用该函数替换 `main()` 中现有内联循环；保留原来的范围断言，并确保它仍在 `prep_optimizer()` 前执行。训练 worker 改为：

```bash
FREEZE_LAYER_NUM=${FREEZE_LAYER_NUM:-8}
```

不要自动增加 `--init_model`、`--resume_model` 或任一 `--rspr_freeze_*` flag。

- [ ] **Step 4: 运行端到端契约测试并确认 GREEN**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_run_train_msrvtt_bg.py tests/test_rspr_model_integration.py
/home/xujie/.conda/envs/tvr/bin/ruff check main_task_retrieval.py tests/test_rspr_model_integration.py tests/test_run_train_msrvtt_bg.py
```

Expected: 全部 PASS；Ruff 无诊断。

- [ ] **Step 5: 提交端到端训练契约**

```bash
git add run_train_msrvtt_bg.sh main_task_retrieval.py tests/test_run_train_msrvtt_bg.py tests/test_rspr_model_integration.py
git commit -m "fix: make rspr training end to end"
```

### Task 3: 在模型侧生成 detached `logvar` 完整诊断

**Files:**
- Modify: `modules/modeling.py:490-520`
- Modify: `tests/test_rspr_model_integration.py:271-320`

**Interfaces:**
- Produces: `_detached_logvar_diagnostics(prefix: str, logvar: Tensor) -> dict[str, Tensor]`
- Produces keys: `<prefix>_logvar_min`, `_mean`, `_p50`, `_p95`, `_max`, `<prefix>_batch_size`
- Extends: `UATVR.last_loss_diagnostics` with text/video summaries and `pair_uncertainty_mean`
- Guarantees: every stored value is detached; `_assemble_training_loss()` return formula is byte-for-byte equivalent in mathematical terms

- [ ] **Step 1: 把 loss 集成测试改成精确 `logvar` 统计契约**

将现有 variance mean 期望替换为：

```python
expected_diagnostics = {
    "dsa": 1.25,
    "prob": 2.5,
    "rank": 3.75,
    "anchor": 4.5,
    "pair_uncertainty_mean": 0.3,
    "text_logvar_min": math.log(2.0),
    "text_logvar_mean": 0.5 * (math.log(2.0) + math.log(4.0)),
    "text_logvar_p50": 0.5 * (math.log(2.0) + math.log(4.0)),
    "text_logvar_p95": math.log(2.0) + 0.95 * (math.log(4.0) - math.log(2.0)),
    "text_logvar_max": math.log(4.0),
    "text_batch_size": 1.0,
    "video_logvar_min": math.log(3.0),
    "video_logvar_mean": 0.5 * (math.log(3.0) + math.log(5.0)),
    "video_logvar_p50": 0.5 * (math.log(3.0) + math.log(5.0)),
    "video_logvar_p95": math.log(3.0) + 0.95 * (math.log(5.0) - math.log(3.0)),
    "video_logvar_max": math.log(5.0),
    "video_batch_size": 1.0,
}
```

保留原总损失精确断言，并对所有 diagnostics 继续断言 `requires_grad is False`。

- [ ] **Step 2: 运行 loss 测试并确认 RED**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_rspr_model_integration.py::test_assemble_training_loss_uses_exact_unweighted_components_and_scales
```

Expected: FAIL；实际 key 仍是 `text_variance_mean`/`video_variance_mean`。

- [ ] **Step 3: 实现 detached 分位数 helper 并接入 loss diagnostics**

在 `modules/modeling.py` 的模型类定义前增加：

```python
def _detached_logvar_diagnostics(prefix, logvar):
    values = logvar.detach().float().reshape(-1)
    quantiles = torch.quantile(
        values,
        values.new_tensor([0.5, 0.95]),
    )
    return {
        f"{prefix}_logvar_min": values.amin(),
        f"{prefix}_logvar_mean": values.mean(),
        f"{prefix}_logvar_p50": quantiles[0],
        f"{prefix}_logvar_p95": quantiles[1],
        f"{prefix}_logvar_max": values.amax(),
        f"{prefix}_batch_size": values.new_tensor(float(logvar.size(0))),
    }
```

`_assemble_training_loss()` 使用以下组装，不保留 variance aliases：

```python
self.last_loss_diagnostics = {
    "dsa": dsa_loss.detach(),
    "prob": probability_loss.detach(),
    "rank": rank_loss.detach(),
    "anchor": rspr_output.anchor_kl.detach(),
    "pair_uncertainty_mean": rspr_output.pair_uncertainty.mean().detach(),
    **_detached_logvar_diagnostics(
        "text", rspr_output.text_distribution.logvar
    ),
    **_detached_logvar_diagnostics(
        "video", rspr_output.video_distribution.logvar
    ),
}
```

总损失 return 表达式不做其他编辑。

- [ ] **Step 4: 运行模型集成测试并确认 GREEN**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_rspr_model_integration.py
/home/xujie/.conda/envs/tvr/bin/ruff check modules/modeling.py tests/test_rspr_model_integration.py
```

Expected: 全部 PASS；Ruff 无诊断。

- [ ] **Step 5: 提交模型诊断**

```bash
git add modules/modeling.py tests/test_rspr_model_integration.py
git commit -m "feat: expose detached rspr logvar diagnostics"
```

### Task 4: 精简启动日志并聚合 step/epoch RSPR 诊断

**Files:**
- Modify: `main_task_retrieval.py:496-579,830-965`
- Modify: `tests/test_rspr_cli.py:120-151`
- Modify: `tests/test_rspr_model_integration.py:620-681`

**Interfaces:**
- Produces: `_RSPRDiagnosticsAccumulator.update(diagnostics: dict) -> bool`
- Produces: `_RSPRDiagnosticsAccumulator.summary() -> dict[str, float] | None`
- Produces: `_format_rspr_diagnostics(model) -> str`
- Produces: `_format_rspr_epoch_diagnostics(summary) -> str`
- Consumes: Task 3 的 diagnostics key；mean/p50/p95 按 `<modality>_batch_size` 加权，min/max 取 epoch 极值

- [ ] **Step 1: 写启动日志、step 格式和 epoch 加权测试**

在 CLI 日志测试中增加：

```python
all_lines = [record.getMessage() for record in caplog.records]
assert not any("[Other]" in line for line in all_lines)
```

将 `_DiagnosticTrainModel.forward()` 的 diagnostics 改为 Task 3 keys，并在训练日志测试中要求同一 step 行包含：

```python
def _diagnostics(**overrides):
    values = {
        "dsa": 1.0,
        "prob": 2.0,
        "rank": 3.0,
        "anchor": 4.0,
        "pair_uncertainty_mean": 5.0,
        "text_logvar_min": -4.0,
        "text_logvar_mean": -2.0,
        "text_logvar_p50": -1.5,
        "text_logvar_p95": -1.0,
        "text_logvar_max": 0.0,
        "text_batch_size": 2.0,
        "video_logvar_min": -5.0,
        "video_logvar_mean": -3.0,
        "video_logvar_p50": -2.5,
        "video_logvar_p95": -2.0,
        "video_logvar_max": -1.0,
        "video_batch_size": 2.0,
    }
    values.update(overrides)
    return {name: torch.tensor(value) for name, value in values.items()}


class _DiagnosticTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.clip = nn.Module()
        self.clip.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        del args, kwargs
        self.last_loss_diagnostics = _diagnostics()
        return self.weight.square()


step_logs = [
    record.getMessage()
    for record in caplog.records
    if " step=" in record.getMessage() and "loss=" in record.getMessage()
]
assert len(step_logs) == 1
for fragment in (
    "loss=1.0000",
    "dsa=1.0000",
    "prob=2.0000",
    "rank=3.0000",
    "anchor=4.0000",
    "u_pair=5.0000",
    "logvar_t=-2.0000/-1.0000",
    "logvar_v=-3.0000/-2.0000",
    "lr=1.00e-02/1.00e-02",
):
    assert fragment in step_logs[0]

epoch_logs = [
    record.getMessage()
    for record in caplog.records
    if record.getMessage().startswith("[RSPR epoch]")
]
assert len(epoch_logs) == 1
assert "logvar_t min/mean/p50/p95/max=" in epoch_logs[0]
assert "logvar_v min/mean/p50/p95/max=" in epoch_logs[0]
assert "u_pair_mean=5.0000" in epoch_logs[0]
```

新增纯聚合测试：

```python
def test_rspr_epoch_diagnostics_weight_batches_and_preserve_extrema():
    accumulator = main_task_retrieval._RSPRDiagnosticsAccumulator()
    first = _diagnostics(text_batch_size=2.0, video_batch_size=2.0)
    second = _diagnostics(
        text_logvar_min=-7.0,
        text_logvar_mean=-1.0,
        text_logvar_p50=-0.5,
        text_logvar_p95=0.5,
        text_logvar_max=1.0,
        video_logvar_min=-6.0,
        video_logvar_mean=-2.0,
        video_logvar_p50=-1.5,
        video_logvar_p95=0.0,
        video_logvar_max=2.0,
        pair_uncertainty_mean=8.0,
        text_batch_size=1.0,
        video_batch_size=1.0,
    )
    assert accumulator.update(first)
    assert accumulator.update(second)

    summary = accumulator.summary()

    assert summary["text_logvar_min"] == -7.0
    assert summary["text_logvar_max"] == 1.0
    assert summary["text_logvar_mean"] == pytest.approx(-5.0 / 3.0)
    assert summary["pair_uncertainty_mean"] == pytest.approx(6.0)
```

另加 off/legacy 形状测试：没有完整 diagnostics dict 时，step formatter 返回空字符串，epoch accumulator 的 `summary()` 返回 `None`。

- [ ] **Step 2: 运行日志测试并确认 RED**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_rspr_cli.py::test_effective_parameter_log_has_explicit_rspr_group tests/test_rspr_model_integration.py::test_train_log_appends_unweighted_rspr_diagnostics tests/test_rspr_model_integration.py::test_rspr_epoch_diagnostics_weight_batches_and_preserve_extrema
```

Expected: FAIL；仍有 `[Other]`、旧 variance 字段，且 accumulator 不存在。

- [ ] **Step 3: 删除 `[Other]` 参数倾倒**

在 `set_seed_logger()` 中保留四个 key group 的循环，删除 `printed_keys` 和末尾 `rest` 组装：

```python
for group, keys in key_params.items():
    vals = [f"{key}={args.__dict__[key]}" for key in keys if key in args.__dict__]
    if vals:
        logger.info("  [%s] %s", group, " | ".join(vals))
```

不得改动 `build_experiment_manifest()` 的完整参数记录。

- [ ] **Step 4: 实现 epoch accumulator 与两个 formatter**

实现 `_RSPRDiagnosticsAccumulator`，内部只保存 Python float：

```python
class _RSPRDiagnosticsAccumulator:
    _STATS = ("mean", "p50", "p95")

    def __init__(self):
        self._weights = {"text": 0.0, "video": 0.0}
        self._weighted = {
            modality: {stat: 0.0 for stat in self._STATS}
            for modality in ("text", "video")
        }
        self._minimum = {"text": float("inf"), "video": float("inf")}
        self._maximum = {"text": float("-inf"), "video": float("-inf")}
        self._pair_weight = 0.0
        self._pair_sum = 0.0

    def update(self, diagnostics):
        required = {
            "pair_uncertainty_mean",
            *(
                f"{modality}_logvar_{stat}"
                for modality in ("text", "video")
                for stat in ("min", "mean", "p50", "p95", "max")
            ),
            "text_batch_size",
            "video_batch_size",
        }
        if not isinstance(diagnostics, dict) or not required.issubset(diagnostics):
            return False
        for modality in ("text", "video"):
            weight = float(diagnostics[f"{modality}_batch_size"])
            self._weights[modality] += weight
            for stat in self._STATS:
                self._weighted[modality][stat] += (
                    weight * float(diagnostics[f"{modality}_logvar_{stat}"])
                )
            self._minimum[modality] = min(
                self._minimum[modality],
                float(diagnostics[f"{modality}_logvar_min"]),
            )
            self._maximum[modality] = max(
                self._maximum[modality],
                float(diagnostics[f"{modality}_logvar_max"]),
            )
        pair_weight = float(diagnostics["text_batch_size"])
        self._pair_weight += pair_weight
        self._pair_sum += pair_weight * float(
            diagnostics["pair_uncertainty_mean"]
        )
        return True

    def summary(self):
        if self._pair_weight == 0:
            return None
        result = {
            "pair_uncertainty_mean": self._pair_sum / self._pair_weight
        }
        for modality in ("text", "video"):
            result[f"{modality}_logvar_min"] = self._minimum[modality]
            result[f"{modality}_logvar_max"] = self._maximum[modality]
            for stat in self._STATS:
                result[f"{modality}_logvar_{stat}"] = (
                    self._weighted[modality][stat] / self._weights[modality]
                )
        return result
```

`_format_rspr_diagnostics()` 要求四项 loss、`pair_uncertainty_mean` 及两模态 mean/p95，返回：

```python
return (
    " | dsa={dsa:.4f} prob={prob:.4f} rank={rank:.4f} anchor={anchor:.4f}"
    " u_pair={pair_uncertainty_mean:.4f}"
    " logvar_t={text_logvar_mean:.4f}/{text_logvar_p95:.4f}"
    " logvar_v={video_logvar_mean:.4f}/{video_logvar_p95:.4f}"
).format(**{name: float(diagnostics[name]) for name in required})
```

epoch formatter 必须返回单行：

```python
return (
    "[RSPR epoch] logvar_t min/mean/p50/p95/max="
    "{text_logvar_min:.4f}/{text_logvar_mean:.4f}/{text_logvar_p50:.4f}/"
    "{text_logvar_p95:.4f}/{text_logvar_max:.4f} | "
    "logvar_v min/mean/p50/p95/max="
    "{video_logvar_min:.4f}/{video_logvar_mean:.4f}/{video_logvar_p50:.4f}/"
    "{video_logvar_p95:.4f}/{video_logvar_max:.4f} | "
    "u_pair_mean={pair_uncertainty_mean:.4f}"
).format(**summary)
```

- [ ] **Step 5: 接入 `train_epoch()` 的单行 step 与 epoch 输出**

在 epoch 开头创建 accumulator；每次 forward 后取 unwrap model 的 `last_loss_diagnostics` 更新。optimizer step 日志改为一行：

```python
logger.info(
    "[Epoch %d/%d] step=%d/%d progress=%.0f%% | loss=%.4f%s | "
    "lr=%.2e/%.2e | time=%.2fs eta=%s",
    epoch + 1,
    args.epochs,
    step + 1,
    num_steps,
    progress,
    float(loss),
    _format_rspr_diagnostics(model),
    lr_clip,
    lr_new,
    time_per_step,
    _fmt_time(eta),
)
```

epoch training-complete 行之后仅在 `summary is not None` 时增加一行 `_format_rspr_epoch_diagnostics(summary)`。不得保存任何完整 `logvar` tensor。

- [ ] **Step 6: 运行日志测试并确认 GREEN**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_rspr_cli.py tests/test_rspr_model_integration.py
/home/xujie/.conda/envs/tvr/bin/ruff check main_task_retrieval.py tests/test_rspr_cli.py tests/test_rspr_model_integration.py
```

Expected: 全部 PASS；off/legacy 无 RSPR 诊断片段；每个 epoch 最多一行 `[RSPR epoch]`。

- [ ] **Step 7: 提交精简日志**

```bash
git add main_task_retrieval.py tests/test_rspr_cli.py tests/test_rspr_model_integration.py
git commit -m "feat: compact rspr training diagnostics"
```

### Task 5: 同步 RSPR 训练事实文档

**Files:**
- Create: `tests/test_rspr_documentation_contract.py`
- Modify: `docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md:650-676`
- Modify: `docs/superpowers/plans/2026-07-19-rspr-core-implementation.md:11-25,730-780,904-1045`
- Modify: `docs/experiments/rspr-core-stage1.md:1-35,55-60`

**Interfaces:**
- Consumes: canonical runtime and training values from Global Constraints
- Guarantees: original design、implementation plan、experiment protocol 都只描述单次端到端核心训练；A4 不依赖历史 checkpoint

- [ ] **Step 1: 写文档契约测试**

创建：

```python
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
DOCUMENTS = [
    REPOSITORY / "docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md",
    REPOSITORY / "docs/superpowers/plans/2026-07-19-rspr-core-implementation.md",
    REPOSITORY / "docs/experiments/rspr-core-stage1.md",
]


def test_rspr_documents_define_one_end_to_end_training_job():
    combined = "\n".join(path.read_text() for path in DOCUMENTS)
    assert "FREEZE_LAYER_NUM=8" in combined
    assert "RSPR_WARMUP_EPOCHS=1" in combined
    assert "TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python" in combined
    assert "TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun" in combined
    assert "Stage A is the one-epoch initialization command" not in combined
    assert "Stage B starts from the Stage A checkpoint" not in combined
    assert "rspr_stage_a" not in combined
    assert "加载复现完成的 UATVR 确定性 checkpoint" not in combined


def test_stage1_protocol_does_not_require_stage_transition_checkpoint():
    protocol = (REPOSITORY / "docs/experiments/rspr-core-stage1.md").read_text()
    fixed_training = protocol.split("## Record sheet", maxsplit=1)[0]
    assert "--init_model" not in fixed_training
    assert "RSPR_FREEZE_CLIP=0" in fixed_training
    assert "RSPR_FREEZE_DSA=0" in fixed_training
```

- [ ] **Step 2: 运行文档测试并确认 RED**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_rspr_documentation_contract.py
```

Expected: FAIL；三份旧事实文档仍包含 Stage A/B 和历史 UATVR checkpoint 描述。

- [ ] **Step 3: 修订原始设计的训练流程**

将原始设计第 8.1/8.2 节替换为以下语义：

```markdown
### 8.1 单次端到端核心训练

- 从 OpenAI CLIP `ViT-B/16` 权重初始化，不要求 UATVR 全模型 checkpoint；
- 在同一个作业和 optimizer 中连续训练 5 epochs；
- `FREEZE_LAYER_NUM=8`，CLIP 后 4 个 block 与 DSA、WTI、RSPR 从第一步联合训练；
- $L_{DSA}$ 与 $L_{prob}$ 从第一步使用完整权重；
- $\lambda_r$ 和 $\lambda_a$ 在第一个 epoch 内线性 warm-up；
- 每个 query 使用显式 `pair_id` 排除多正样本后选择困难负样本；
- 以验证集 R@1 为主选择 checkpoint，同时记录 `logvar` 与错误的关系。

### 8.2 初始化与数值保护

- 概率 mean head 以确定性中心为残差起点，最后一层零初始化；
- `logvar` 从 `prior_std` 对应值开始并限制在 $[-8,2]$；
- CLIP 使用 `coef_lr=1e-3` 的较低学习率；
- 任一 loss 出现 NaN/Inf 或 `logvar` 连续一个 epoch 饱和时停止实验并诊断。
```

保留第 8.3 的 Beta evidence 为后续研究边界，但将标题改为“后续扩展：可信负样本”，不得写成当前核心训练阶段。

- [ ] **Step 4: 修订原实施计划与实验协议**

原实施计划的 Global Constraints 增加当前 `tvr` 绝对路径，删除“历史环境不存在所以使用 python3”的旧约束；Task 6 的 `freeze_layer_num` 说明改为端到端 CLIP 分层冻结；Task 8 的两个阶段命令替换为：

```bash
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
RSPR_MODE=stochastic \
RSPR_FREEZE_CLIP=0 \
RSPR_FREEZE_DSA=0 \
RSPR_WARMUP_EPOCHS=1 \
FREEZE_LAYER_NUM=8 \
RUN_ID=rspr_a3_seed0 \
./run_train_msrvtt_bg.sh
```

实验协议的 “Fixed training stages” 改为 “Fixed end-to-end training”，只保留同一命令；说明 `--init_model` 不用于固定前置阶段，A0–A8 全部从同一 CLIP 起点执行。最终 checklist 增加 runtime、单 optimizer 和 `logvar` 日志要求，保留“不启动训练”的当前状态说明。

- [ ] **Step 5: 运行文档契约与格式扫描并确认 GREEN**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_rspr_documentation_contract.py
rg -n "Stage A is|Stage B starts|rspr_stage_a|加载复现完成的 UATVR 确定性 checkpoint" docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md docs/superpowers/plans/2026-07-19-rspr-core-implementation.md docs/experiments/rspr-core-stage1.md
```

Expected: pytest PASS；`rg` 退出码 1 且无输出。

- [ ] **Step 6: 提交事实文档同步**

```bash
git add tests/test_rspr_documentation_contract.py docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md docs/superpowers/plans/2026-07-19-rspr-core-implementation.md docs/experiments/rspr-core-stage1.md
git commit -m "docs: define rspr end-to-end training protocol"
```

### Task 6: 完整 CPU 与静态验收

**Files:**
- Verify only: all files changed in Tasks 1–5

**Interfaces:**
- Consumes: all prior task outputs
- Produces: fresh verification evidence only; does not launch training or mutate checkpoints/data

- [ ] **Step 1: 运行 RSPR 与入口定向测试**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_reparameterized_distribution.py tests/test_stochastic_prototype_ranking.py tests/test_rspr_core.py tests/test_rspr_pair_ids.py tests/test_rspr_cli.py tests/test_rspr_model_integration.py tests/test_rspr_rerank.py tests/test_rspr_ablation_matrix.py tests/test_run_train_msrvtt_bg.py tests/test_eval_rspr_entrypoint.py tests/test_rspr_documentation_contract.py
```

Expected: 全部 PASS，无 skipped/failure/error。

- [ ] **Step 2: 运行完整 CPU 测试集**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q
```

Expected: 全部 PASS；若出现与本任务无关且由现有 dirty worktree 引起的失败，必须先记录精确测试名和错误证据，不得修改无关用户文件规避。

- [ ] **Step 3: 运行 Ruff、编译与 Bash 校验**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/ruff check main_task_retrieval.py modules/modeling.py scripts tests/test_rspr_cli.py tests/test_rspr_model_integration.py tests/test_run_train_msrvtt_bg.py tests/test_eval_rspr_entrypoint.py tests/test_rspr_documentation_contract.py
/home/xujie/.conda/envs/tvr/bin/python -m py_compile main_task_retrieval.py modules/modeling.py
bash -n scripts/tvr_runtime.sh scripts/rspr_shell_config.sh run_train_msrvtt_bg.sh eval.sh
```

Expected: 三条命令均退出 0，无诊断。

- [ ] **Step 4: 运行最终语义扫描**

Run:

```bash
rg -n "python3 .*build_msrvtt_trusted_split|^[[:space:]]*torchrun " run_train_msrvtt_bg.sh eval.sh
rg -n "\[Other\]|text_variance_mean|video_variance_mean" main_task_retrieval.py modules/modeling.py tests/test_rspr_*.py
rg -n "Stage A is|Stage B starts|rspr_stage_a|加载复现完成的 UATVR 确定性 checkpoint" docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md docs/superpowers/plans/2026-07-19-rspr-core-implementation.md docs/experiments/rspr-core-stage1.md
```

Expected: 每条 `rg` 均退出码 1 且无输出；入口只通过 `TVR_PYTHON`/`TVR_TORCHRUN` 调用运行时。

- [ ] **Step 5: 检查提交范围和工作树保护**

Run:

```bash
git status --short
git log --oneline -6
git diff --check HEAD~5..HEAD
```

Expected: 最近五个实现提交分别对应 runtime、端到端训练、模型诊断、日志、文档；`git diff --check` 退出 0。原有无关 dirty 文件仍保持原状态，没有被暂存、删除或还原。

## Final Verification Checklist

- [ ] 训练 controller 在 detached worker 前验证 `tvr` Python/torchrun 和 `torch`/`cv2` import。
- [ ] worker 的 trusted split builder 使用 `TVR_PYTHON`，训练使用 `TVR_TORCHRUN`；`eval.sh` 使用同一契约。
- [ ] 规范训练命令不含阶段衔接 `--init_model`，默认 `FREEZE_LAYER_NUM=8` 且连续训练 5 epochs。
- [ ] CLIP 后 4 层、DSA、WTI、RSPR 同时进入 optimizer；CLIP 使用 `coef_lr=1e-3` 的低学习率。
- [ ] DSA/prob 从第一步完整生效，rank/anchor 只在同一作业的第一个 epoch 内线性 warm-up。
- [ ] `_assemble_training_loss()` 总损失和梯度保持不变，所有 diagnostics 已 detached。
- [ ] step 单行包含 total、四项未加权 loss、`u_pair`、text/video logvar mean/p95、两组 LR、time/ETA。
- [ ] epoch 单行包含 text/video logvar min/mean/p50/p95/max 与 `u_pair_mean`，不保存完整 tensor。
- [ ] `off/legacy` 不输出 RSPR 占位诊断，启动日志不再输出 `[Other]`。
- [ ] 三份事实文档只定义从 CLIP 开始的单次端到端训练；A0–A8 共享起点和日程。
- [ ] CPU 测试、Ruff、py_compile、Bash syntax 和语义扫描全部通过。
- [ ] 本实施未启动真实训练、未创建新 checkpoint、未修改数据集。
