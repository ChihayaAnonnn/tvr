# RSPR Research-First End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用最少工程改动固化 RSPR 单次端到端训练、`tvr` 运行环境和精简日志，并以真实 20-step GPU smoke 验证训练链路。

**Architecture:** 不新增 runtime helper、诊断聚合器或测试文件。训练/评估脚本直接选用 `tvr` Python/torchrun，训练循环复用现有 detached diagnostics；文档和 `AGENT.md` 同步科研优先原则，最后直接启动 A3 smoke。

**Tech Stack:** Bash、Python 3.11、PyTorch、torchrun、Git

## Global Constraints

- 规格事实源为 `docs/superpowers/specs/2026-07-22-rspr-training-runtime-and-logging-design.md`。
- 本仓库按科研实验项目处理：优先让真实训练跑通，不新增生产级测试矩阵、抽象层、兼容层和重复 preflight。
- 不新增自动化测试；已有测试仅在旧期望阻碍后续工作时做最小文本更新。
- 训练从 OpenAI CLIP `ViT-B/16` 权重开始，不加载历史 UATVR 全模型 checkpoint，不使用固定 Stage A/B。
- 规范训练在同一个 optimizer 和学习率计划内连续 5 epochs；`RSPR_WARMUP_EPOCHS=1` 只线性启用 rank/anchor。
- 默认配置固定 `RSPR_FREEZE_CLIP=0`、`RSPR_FREEZE_DSA=0`、`FREEZE_LAYER_NUM=8`。
- 默认运行时固定为 `/home/xujie/.conda/envs/tvr/bin/python` 和 `/home/xujie/.conda/envs/tvr/bin/torchrun`，仍允许 `TVR_PYTHON`、`TVR_TORCHRUN` 环境变量覆盖。
- 日志不增加新统计计算，只复用 `last_loss_diagnostics` 中已有的四项 loss、`pair_uncertainty_mean` 和两模态 variance mean。
- `off/legacy` 缺少 RSPR diagnostics 时不打印占位字段。
- 不运行 pytest、Ruff、py_compile、文档扫描或专用 dry-run；真实 20-step GPU smoke 是本次主要验收。
- smoke 必须使用独立 `RUN_ID`、`OUTPUT_DIR` 和 `TRAIN_PID_FILE`，达到首个 20-step 日志后终止 worker 进程组，不继续完整训练。
- 工作树含用户已有修改、删除和未跟踪文件；只暂存本任务明确列出的文件，不清理或覆盖无关改动。

---

### Task 1: 直接完成端到端入口、精简日志和科研原则

**Files:**
- Modify: `AGENT.md`
- Modify: `run_train_msrvtt_bg.sh`
- Modify: `eval.sh`
- Modify: `main_task_retrieval.py`
- Modify: `docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md`
- Modify: `docs/superpowers/plans/2026-07-19-rspr-core-implementation.md`
- Modify: `docs/experiments/rspr-core-stage1.md`

**Interfaces:**
- `TVR_PYTHON=${TVR_PYTHON:-/home/xujie/.conda/envs/tvr/bin/python}`
- `TVR_TORCHRUN=${TVR_TORCHRUN:-/home/xujie/.conda/envs/tvr/bin/torchrun}`
- `_format_rspr_diagnostics(model) -> str`：diagnostics 完整时返回单行片段，否则返回空字符串
- 训练 step 行：total、dsa/prob/rank/anchor、`u_pair`、text/video variance mean、CLIP/other LR、time/ETA

- [ ] **Step 1: 将科研最小工程原则写入 `AGENT.md`**

在“项目环境”和“文档路由”之间加入：

```markdown
## 科研项目工作原则

- 本仓库是科研实验项目，首要目标是让研究假设快速、可复现地进入真实训练与评估，不按生产服务标准扩展工程设施。
- 对 shell 参数、日志格式、文档和局部配置等低风险改动，默认不新增测试矩阵、抽象层、兼容层或重复 preflight。
- 对核心数学公式、梯度路径、数据划分、正负样本语义和 checkpoint 兼容性仍做必要验证，因为这些错误会直接使实验结论失效。
- 验收优先使用最小真实训练/评估 smoke；只有出现可复现故障，或用户明确要求，才补充针对性测试和诊断。
- 日志只保留判断 loss、学习率、速度和关键研究变量是否正常所需的信息，不为可能的未来分析预先增加统计。
```

- [ ] **Step 2: 训练和评估入口直接使用 `tvr` 环境**

在两个入口完成 `ROOT_DIR` 解析后直接定义：

```bash
TVR_PYTHON=${TVR_PYTHON:-/home/xujie/.conda/envs/tvr/bin/python}
TVR_TORCHRUN=${TVR_TORCHRUN:-/home/xujie/.conda/envs/tvr/bin/torchrun}
```

`run_train_msrvtt_bg.sh` 将 split builder 的 `python3` 替换为：

```bash
"${TVR_PYTHON}" "${ROOT_DIR}/scripts/build_msrvtt_trusted_split.py" \
```

并将训练命令的 `torchrun` 替换为：

```bash
"${TVR_TORCHRUN}" --nproc_per_node="${NPROC}" --master_addr=127.0.0.9 --master_port=29547 \
```

`eval.sh` 对 MSR-VTT split builder 和末尾单卡 torchrun 做相同替换。两个脚本在现有配置摘要附近各输出一次：

```bash
echo "[Runtime] python=${TVR_PYTHON} torchrun=${TVR_TORCHRUN}"
```

不新增路径检查、import 检查或共享 helper。

- [ ] **Step 3: 固化单次端到端默认值**

训练 worker 将默认值改为：

```bash
FREEZE_LAYER_NUM=${FREEZE_LAYER_NUM:-8}
```

保留 `RSPR_FREEZE_CLIP=0`、`RSPR_FREEZE_DSA=0` 和 `RSPR_WARMUP_EPOCHS=1.0` 的现有默认值。规范命令不传 `--init_model` 或 `--resume_model`；现有显式参数能力不删除。

- [ ] **Step 4: 删除无用启动参数并压缩 step 日志**

`set_seed_logger()` 保留 `Training`、`Model`、`RSPR`、`Protocol` 四个分组，删除 `printed_keys`、`rest` 和 `[Other]` 输出。

`_format_rspr_diagnostics()` 改为：

```python
def _format_rspr_diagnostics(model):
    diagnostic_model = model.module if hasattr(model, "module") else model
    diagnostics = getattr(diagnostic_model, "last_loss_diagnostics", None)
    required = (
        "dsa",
        "prob",
        "rank",
        "anchor",
        "pair_uncertainty_mean",
        "text_variance_mean",
        "video_variance_mean",
    )
    if not isinstance(diagnostics, dict) or any(
        name not in diagnostics for name in required
    ):
        return ""
    values = {name: float(diagnostics[name]) for name in required}
    return (
        " | dsa={dsa:.4f} prob={prob:.4f} rank={rank:.4f} anchor={anchor:.4f}"
        " u_pair={pair_uncertainty_mean:.4f}"
        " variance_t={text_variance_mean:.4f}"
        " variance_v={video_variance_mean:.4f}"
    ).format(**values)
```

optimizer step 日志压缩为现有单行的字段重命名，不新增 epoch summary：

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

不要修改 `modules/modeling.py` 的 loss 公式或 diagnostics 生成逻辑。

- [ ] **Step 5: 同步三份 RSPR 事实文档**

将原始设计第 8 节和原实施计划 Task 6/Task 8 中的 Stage A/B 描述改成以下唯一训练语义：

```markdown
- 从 OpenAI CLIP `ViT-B/16` 权重初始化，不要求 UATVR 全模型 checkpoint；
- 在同一个作业和 optimizer 中连续训练 5 epochs；
- `FREEZE_LAYER_NUM=8`，CLIP 后 4 个 block 与 DSA、WTI、RSPR 从第一步联合训练；
- $L_{DSA}$ 与 $L_{prob}$ 从第一步使用完整权重；
- $\lambda_r$ 与 $\lambda_a$ 在第一个 epoch 内线性 warm-up。
```

`docs/experiments/rspr-core-stage1.md` 删除两个阶段命令，只保留：

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

A0–A8 说明统一为相同 CLIP 起点、trusted split、5 epochs 和优化日程，只改变消融矩阵定义的参数；A4 不要求历史 UATVR checkpoint。

- [ ] **Step 6: 目视检查改动并提交**

只检查目标 diff，不运行自动化校验：

```bash
git diff -- AGENT.md run_train_msrvtt_bg.sh eval.sh main_task_retrieval.py docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md docs/superpowers/plans/2026-07-19-rspr-core-implementation.md docs/experiments/rspr-core-stage1.md
git status --short
```

确认没有无关文件进入 diff 后提交：

```bash
git add AGENT.md run_train_msrvtt_bg.sh eval.sh main_task_retrieval.py docs/superpowers/specs/2026-07-19-reparameterized-stochastic-prototype-ranking-design.md docs/superpowers/plans/2026-07-19-rspr-core-implementation.md docs/experiments/rspr-core-stage1.md
git commit -m "feat: streamline rspr research training"
```

### Task 2: 用真实 20-step A3 smoke 验收训练链路

**Files:**
- Runtime output only: controller stdout 中 `[run_train_msrvtt_bg] LOG_FILE=` 给出的日志路径
- Runtime output only: `ckpts/ckpt_msrvtt_${RUN_ID}/`
- Runtime PID only: `/tmp/tvr_${RUN_ID}.pid`

**Interfaces:**
- Consumes: Task 1 的训练入口和当前 MSR-VTT trusted split/TQFS cache
- Produces: 第一个 `step=20` 日志及有限的 total/dsa/prob/rank/anchor/`u_pair`
- Stop contract: 对已验证的数字 PID 使用 `kill -TERM -- "-${worker_pid}"`，只终止该次 `setsid` worker 进程组

- [ ] **Step 1: 检查 GPU 空闲状态并创建独立 smoke 标识**

只读取 GPU 状态：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

选择当前训练脚本默认的 `CUDA_VISIBLE_DEVICES=0,1,2,3`。用当前时间生成一次性标识，并把实际展开值记录到执行日志；不得复用正式实验目录：

```bash
SMOKE_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="rspr_smoke_${SMOKE_STAMP}"
OUTPUT_DIR="ckpts/ckpt_msrvtt_${RUN_ID}"
TRAIN_PID_FILE="/tmp/tvr_${RUN_ID}.pid"
```

- [ ] **Step 2: 启动规范 A3 训练**

```bash
TVR_PYTHON=/home/xujie/.conda/envs/tvr/bin/python \
TVR_TORCHRUN=/home/xujie/.conda/envs/tvr/bin/torchrun \
RSPR_MODE=stochastic \
RSPR_FREEZE_CLIP=0 \
RSPR_FREEZE_DSA=0 \
RSPR_WARMUP_EPOCHS=1 \
FREEZE_LAYER_NUM=8 \
RUN_ID="${RUN_ID}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_PID_FILE="${TRAIN_PID_FILE}" \
./run_train_msrvtt_bg.sh
```

保持日志监视，直到出现 `step=20`，或出现 traceback、CUDA OOM、NaN/Inf、worker 非零退出。

- [ ] **Step 3: 到达 20 steps 后停止独立 worker 进程组**

从 PID 文件读取并验证数字 PID，再停止这一组进程：

```bash
worker_pid="$(tr -d '[:space:]' < "${TRAIN_PID_FILE}")"
[[ "${worker_pid}" =~ ^[1-9][0-9]*$ ]]
kill -TERM -- "-${worker_pid}"
```

轮询 `kill -0 -- "-${worker_pid}"`，确认进程组退出；若 10 秒内未退出，再报告残留进程，不扩大终止范围。

- [ ] **Step 4: 读取 smoke 结果并交付**

从该次独立日志中报告：

- `tvr` Python 与 torchrun 路径；
- 模型初始化来源为 CLIP、没有 `--init_model`；
- 第 20 step 的 total、dsa、prob、rank、anchor、`u_pair`；
- CLIP/other LR、step time、ETA；
- 是否出现 OOM、NaN/Inf 或 traceback；
- worker 进程组是否已停止；
- smoke log 和输出目录的绝对路径。

smoke 成功后不启动 5-epoch 正式训练，不删除日志或输出目录。

## Final Verification Checklist

- [ ] `AGENT.md` 已包含科研实验优先、最小工程校验原则。
- [ ] 训练和评估使用 `TVR_PYTHON`/`TVR_TORCHRUN`，没有新增 runtime helper。
- [ ] 默认 `FREEZE_LAYER_NUM=8`，规范训练没有 Stage A/B 或阶段衔接 checkpoint。
- [ ] 启动日志无 `[Other]`，step 日志只复用已有 diagnostics。
- [ ] 三份事实文档与单次端到端语义一致。
- [ ] 没有新增测试文件、诊断聚合器或通用校验框架。
- [ ] A3 smoke 到达第 20 optimizer step，关键 loss 有限且 worker 已停止。
