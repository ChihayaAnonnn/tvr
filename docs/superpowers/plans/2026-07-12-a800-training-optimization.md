# A800 Training Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 MSRVTT trusted-v1 P0 科研口径的前提下，为当前 5×A800 80GB 机器提供拓扑友好的四卡启动、NVMe TQFS 缓存和可追溯的数据流水线调优入口。

**Architecture:** 保留现有训练与缓存实现，只在三个边界增加机器级配置：CLI/DataLoader 传递 worker 与 prefetch，experiment manifest 记录运行工程事实，shell 在启动前保护四卡 P0 口径。TQFS 缓存仍使用现有原子格式，但默认迁移到 NVMe 并在创建 worker pool 前检查空间。

**Tech Stack:** Bash、Python 3.10、PyTorch 2.1.2+cu121、pytest、Ruff、OpenCV、scikit-learn、timm 0.9.16。

## Global Constraints

- 科研决策只以 `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 为准。
- P0 固定 4 GPU、每卡 micro-batch 64、global forward batch 256、gradient accumulation 1。
- 默认 OpenAI CLIP 视觉前 4 层 activation checkpointing；checkpoint-off 仅作独立吞吐对照。
- 默认 GPU 集合为 `0,1,2,4`，GPU 3 不加入 P0 DDP。
- 默认 TQFS 缓存为 `/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224`。
- 不启动长期训练；只运行自动测试和少量视频 cache smoke。
- 不修改 torch；只补装缺失的 `timm==0.9.16`，不调整其他已安装包版本。
- 保留用户对 `docs/reference/uatvr_backbone_upgrade_strategy.md` 的删除，不提交或恢复它。

---

### Task 1: DataLoader worker/prefetch CLI 契约

**Files:**
- Modify: `tests/test_msrvtt_dataloader_contract.py`
- Modify: `tests/test_main_task_hard_negative_args.py`
- Modify: `main_task_retrieval.py`
- Modify: `dataloaders/data_dataloaders.py`

**Interfaces:**
- Consumes: `args.num_thread_reader: int` 和新增 `args.prefetch_factor: int`。
- Produces: `_video_loader_kwargs(args) -> dict`；worker 大于 0 时包含用户指定的 `prefetch_factor`，worker 为 0 时不包含 multiprocessing-only 参数。

- [ ] **Step 1: 写 DataLoader 失败测试**

在 `tests/test_msrvtt_dataloader_contract.py` 中把 worker 设置测试改为显式传入 prefetch，并增加零 worker 契约：

```python
def test_video_loader_worker_settings_limit_oversubscription():
    kwargs = builders._video_loader_kwargs(
        SimpleNamespace(num_thread_reader=8, prefetch_factor=4)
    )

    assert kwargs["num_workers"] == 8
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4
    assert kwargs["worker_init_fn"] is builders._configure_video_worker


def test_video_loader_zero_workers_omits_multiprocessing_options():
    kwargs = builders._video_loader_kwargs(
        SimpleNamespace(num_thread_reader=0, prefetch_factor=4)
    )

    assert kwargs["num_workers"] == 0
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs
    assert "worker_init_fn" not in kwargs
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_msrvtt_dataloader_contract.py -k 'video_loader_worker or video_loader_zero'
```

Expected: 自定义值测试因实际 `prefetch_factor == 2` 失败。

- [ ] **Step 3: 写 CLI 校验失败测试**

在 `tests/test_main_task_hard_negative_args.py` 中通过现有最小合法 argv helper 增加：

```python
@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--num_thread_reader", "-1", "--num_thread_reader must be non-negative"),
        ("--prefetch_factor", "0", "--prefetch_factor must be positive"),
    ],
)
def test_dataloader_cli_rejects_invalid_worker_settings(
    monkeypatch, flag, value, message
):
    argv = _minimal_hygiene_train_argv() + [flag, value]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match=message):
        retrieval.get_args()
```

- [ ] **Step 4: 运行 CLI 测试并确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_main_task_hard_negative_args.py -k dataloader_cli
```

Expected: `--prefetch_factor` 尚未定义，测试失败。

- [ ] **Step 5: 实现最小 CLI 和 DataLoader 传递**

在 `main_task_retrieval.py` 的 `--num_thread_reader` 附近增加：

```python
parser.add_argument(
    "--prefetch_factor",
    type=int,
    default=2,
    help="Number of batches prefetched by each DataLoader worker.",
)
```

在 `get_args()` 校验区增加：

```python
if args.num_thread_reader < 0:
    raise ValueError("--num_thread_reader must be non-negative")
if args.prefetch_factor <= 0:
    raise ValueError("--prefetch_factor must be positive")
```

把 `dataloaders/data_dataloaders.py` 的固定值改为：

```python
prefetch_factor=int(args.prefetch_factor),
```

- [ ] **Step 6: 运行定向测试并确认 GREEN**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_msrvtt_dataloader_contract.py tests/test_main_task_hard_negative_args.py
```

Expected: 两个文件全部通过。

- [ ] **Step 7: 提交 Task 1**

```bash
git add main_task_retrieval.py dataloaders/data_dataloaders.py tests/test_msrvtt_dataloader_contract.py tests/test_main_task_hard_negative_args.py
git commit -m "perf: make video prefetch configurable"
```

### Task 2: Manifest 记录数据流水线与 GPU 工程事实

**Files:**
- Modify: `tests/test_experiment_tracking.py`
- Modify: `experiment_tracking.py`
- Modify: `main_task_retrieval.py`

**Interfaces:**
- Consumes: `args.num_thread_reader`、`args.prefetch_factor`、`args.n_gpu`、进程环境 `CUDA_VISIBLE_DEVICES`。
- Produces: manifest 新增顶层 `runtime` 字段；现有 `backbone`、`data`、`batch` 字段不改语义。

- [ ] **Step 1: 写 manifest 失败测试**

扩展 `tests/test_experiment_tracking.py::_args`：

```python
num_thread_reader=8,
prefetch_factor=4,
n_gpu=4,
```

在 manifest 测试中设置环境并断言：

```python
monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,4")
assert payload["runtime"] == {
    "cuda_visible_devices": "0,1,2,4",
    "num_dataloader_workers_per_rank": 8,
    "prefetch_factor": 4,
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": True,
}
```

同时把顶层字段集合加入 `"runtime"`。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_experiment_tracking.py -k manifest_contains
```

Expected: `runtime` 不存在而失败。

- [ ] **Step 3: 实现 runtime manifest**

在 `experiment_tracking.py` 中构造：

```python
workers = int(getattr(args, "num_thread_reader", 0))
runtime = {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "num_dataloader_workers_per_rank": workers,
    "prefetch_factor": int(getattr(args, "prefetch_factor", 2)),
    "pin_memory": bool(torch.cuda.is_available()),
    "persistent_workers": workers > 0,
}
```

为避免让 tracking 模块引入 torch 依赖，最终实现应由调用方在
`main_task_retrieval.py` 把 `args.pin_memory = torch.cuda.is_available()` 设为运行事实，
tracking 中读取 `args.pin_memory`：

```python
"pin_memory": bool(getattr(args, "pin_memory", False)),
```

把 `runtime` 加入返回 payload。

- [ ] **Step 4: 运行测试并确认 GREEN**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_experiment_tracking.py
```

Expected: 全部通过。

- [ ] **Step 5: 提交 Task 2**

```bash
git add experiment_tracking.py main_task_retrieval.py tests/test_experiment_tracking.py
git commit -m "chore: track training runtime configuration"
```

### Task 3: A800 四卡 shell 默认值和启动前保护

**Files:**
- Modify: `tests/test_main_task_hard_negative_args.py`
- Modify: `train_msrvtt.sh`

**Interfaces:**
- Consumes: `CUDA_VISIBLE_DEVICES`、`NPROC`、`TRAIN_NUM_WORKERS`、`TRAIN_PREFETCH_FACTOR`、`TRAIN_BATCH_SIZE`、`TRAIN_GRADIENT_ACCUMULATION_STEPS`、`TQFS_CACHE_DIR`、`EXPERIMENT_PROFILE`。
- Produces: 通过 `torchrun` 传递 `--num_thread_reader`、`--prefetch_factor` 和 NVMe `--tqfs_cache_dir`；hygiene P0 非四卡配置在 torchrun 前退出 2。

- [ ] **Step 1: 扩展 fake torchrun helper 并写失败测试**

让训练脚本 helper 默认传入 `CUDA_VISIBLE_DEVICES=0,1,2,4` 和 `NPROC=4`，避免新增 P0 gate 破坏既有训练脚本测试；eval helper 继续使用单卡。

新增测试：

```python
def test_train_script_forwards_a800_pipeline_settings(tmp_path):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        extra_env={
            "TRAIN_NUM_WORKERS": "12",
            "TRAIN_PREFETCH_FACTOR": "4",
            "TRAIN_BATCH_SIZE": "256",
            "TRAIN_GRADIENT_ACCUMULATION_STEPS": "1",
            "TQFS_CACHE_DIR": "/nvme/tqfs",
        },
    )
    assert result.returncode == 0, result.stderr
    args = capture_path.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--num_thread_reader") + 1] == "12"
    assert args[args.index("--prefetch_factor") + 1] == "4"
    assert args[args.index("--batch_size") + 1] == "256"
    assert args[args.index("--gradient_accumulation_steps") + 1] == "1"
    assert args[args.index("--tqfs_cache_dir") + 1] == "/nvme/tqfs"


@pytest.mark.parametrize(
    ("visible", "nproc", "message"),
    [
        ("0,1,2", "3", "requires exactly 4 GPUs"),
        ("0,1,2,2", "4", "duplicate GPU IDs"),
        ("0,1,2,4", "3", "NPROC=3 does not match"),
    ],
)
def test_hygiene_train_script_rejects_invalid_gpu_world(
    tmp_path, visible, nproc, message
):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh",
        tmp_path,
        "0",
        extra_env={"CUDA_VISIBLE_DEVICES": visible, "NPROC": nproc},
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not capture_path.exists()


@pytest.mark.parametrize(
    ("extra_env", "message"),
    [
        ({"TRAIN_BATCH_SIZE": "320"}, "requires TRAIN_BATCH_SIZE=256"),
        (
            {"TRAIN_GRADIENT_ACCUMULATION_STEPS": "2"},
            "requires TRAIN_GRADIENT_ACCUMULATION_STEPS=1",
        ),
    ],
)
def test_hygiene_train_script_rejects_changed_batch_protocol(
    tmp_path, extra_env, message
):
    result, capture_path = _run_with_fake_torchrun(
        "train_msrvtt.sh", tmp_path, "0", extra_env=extra_env
    )
    assert result.returncode == 2
    assert message in result.stderr
    assert not capture_path.exists()
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_main_task_hard_negative_args.py -k 'a800_pipeline or invalid_gpu_world'
```

Expected: shell 尚未传递自定义设置，也未拒绝非法 world。

- [ ] **Step 3: 实现 shell 默认值与校验**

在 `train_msrvtt.sh` 设置：

```bash
TQFS_CACHE_DIR=${TQFS_CACHE_DIR:-/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
TRAIN_GRADIENT_ACCUMULATION_STEPS=${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4}"
```

解析 GPU 后，对 hygiene profile 实现四卡、唯一性和 NPROC 一致性校验；对 worker 和
prefetch 实现非负/正整数 shell 校验；hygiene profile 还必须拒绝 batch 非 256 或
accumulation 非 1。调用参数改为：

```bash
--do_train --num_thread_reader="${TRAIN_NUM_WORKERS}" \
--prefetch_factor="${TRAIN_PREFETCH_FACTOR}" --epochs=5 \
--batch_size="${TRAIN_BATCH_SIZE}" \
--gradient_accumulation_steps="${TRAIN_GRADIENT_ACCUMULATION_STEPS}" \
```

启动日志打印 GPU、NPROC、worker、prefetch、cache 和 checkpoint 配置；checkpoint
关闭时打印 `A800 throughput comparison: activation checkpointing disabled`。

- [ ] **Step 4: shell 语法和定向测试 GREEN**

Run:

```bash
bash -n train_msrvtt.sh run_train_msrvtt_bg.sh
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_main_task_hard_negative_args.py
```

Expected: shell syntax exit 0，测试文件全部通过。

- [ ] **Step 5: 提交 Task 3**

```bash
git add train_msrvtt.sh tests/test_main_task_hard_negative_args.py
git commit -m "perf: configure trusted training for A800 topology"
```

### Task 4: NVMe TQFS cache 默认值和空间门禁

**Files:**
- Create: `tests/test_build_msrvtt_tqfs_cache.py`
- Modify: `scripts/build_msrvtt_tqfs_cache.py`

**Interfaces:**
- Consumes: cache path 与 `shutil.disk_usage(path).free`。
- Produces: `ensure_cache_space(cache_dir: Path, minimum_free_bytes: int) -> None`；空间不足抛 `RuntimeError`，发生于 `ProcessPoolExecutor` 创建前。

- [ ] **Step 1: 写空间检查失败测试**

```python
from types import SimpleNamespace

import pytest

from scripts import build_msrvtt_tqfs_cache as builder


def test_cache_space_gate_rejects_insufficient_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(
        builder.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=49 * 1024**3),
    )
    with pytest.raises(RuntimeError, match="at least 50 GiB"):
        builder.ensure_cache_space(tmp_path / "cache", 50 * 1024**3)


def test_cache_space_gate_creates_parent_and_accepts_capacity(monkeypatch, tmp_path):
    cache_dir = tmp_path / "nested" / "cache"
    monkeypatch.setattr(
        builder.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=60 * 1024**3),
    )
    builder.ensure_cache_space(cache_dir, 50 * 1024**3)
    assert cache_dir.is_dir()
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_build_msrvtt_tqfs_cache.py
```

Expected: `ensure_cache_space` 不存在。

- [ ] **Step 3: 实现空间检查与 NVMe 默认值**

在脚本中导入 `shutil`，定义：

```python
MINIMUM_CACHE_FREE_BYTES = 50 * 1024**3
DEFAULT_CACHE_DIR = Path.home() / ".cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224"


def ensure_cache_space(cache_dir, minimum_free_bytes=MINIMUM_CACHE_FREE_BYTES):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"TQFS cache filesystem requires at least 50 GiB free; "
            f"available={free_bytes / 1024**3:.1f} GiB path={cache_dir}"
        )
```

把 `--cache-dir` 默认值设为 `str(DEFAULT_CACHE_DIR)`，并在构造
`TQFSFrameCache` 和 worker pool 前调用 `ensure_cache_space`。

- [ ] **Step 4: 运行定向测试并确认 GREEN**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_build_msrvtt_tqfs_cache.py tests/test_tqfs_cache.py
```

Expected: 全部通过。

- [ ] **Step 5: 提交 Task 4**

```bash
git add scripts/build_msrvtt_tqfs_cache.py tests/test_build_msrvtt_tqfs_cache.py
git commit -m "perf: place TQFS cache on NVMe"
```

### Task 5: 安装缺失依赖并运行 cache smoke

**Files:**
- No repository file changes.

**Interfaces:**
- Consumes: conda 环境 `/home/xujie/miniconda3/envs/ret` 和真实 MSRVTT 视频。
- Produces: 可导入的 `timm==0.9.16`；NVMe cache 中少量配置一致的 `.npy` 项目。

- [ ] **Step 1: 确认 torch 与 timm 前置状态**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/python -c "import torch; print(torch.__version__)"
/home/xujie/miniconda3/envs/ret/bin/python -c "import timm"
```

Expected: torch 输出 `2.1.2+cu121`；timm 导入失败。

- [ ] **Step 2: 仅安装 timm**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/python -m pip install --no-deps timm==0.9.16
```

Expected: 安装 timm，不触碰 torch 和其他包版本。

- [ ] **Step 3: 验证依赖状态**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/python -c "import torch, timm; print(torch.__version__, timm.__version__)"
/home/xujie/miniconda3/envs/ret/bin/python -m pip check
```

Expected: 输出 `2.1.2+cu121 0.9.16`；`pip check` 仍只报告实施前已存在的 chex/orbax JAX 缺依赖。

- [ ] **Step 4: 首次 cache smoke**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/python scripts/build_msrvtt_tqfs_cache.py --workers 2 --limit 2
```

Expected: `total=2 built=2 hit=0`，或若此前已有条目则相应为 hit；无解码异常。

- [ ] **Step 5: 二次 cache smoke**

Run: 重复上一步命令。

Expected: `total=2 built=0 hit=2`。

### Task 6: 全量静态与回归验收

**Files:**
- Verify only.

**Interfaces:**
- Consumes: Tasks 1–5 的完整工作树和 `ret` 环境。
- Produces: 新鲜的测试、静态检查、shell 与环境证据；不启动训练。

- [ ] **Step 1: Ruff 检查修改的 Python 文件**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/ruff check main_task_retrieval.py experiment_tracking.py dataloaders/data_dataloaders.py scripts/build_msrvtt_tqfs_cache.py tests/test_msrvtt_dataloader_contract.py tests/test_main_task_hard_negative_args.py tests/test_experiment_tracking.py tests/test_build_msrvtt_tqfs_cache.py
```

Expected: exit 0。

- [ ] **Step 2: shell 语法检查**

Run:

```bash
bash -n train_msrvtt.sh run_train_msrvtt_bg.sh
```

Expected: exit 0。

- [ ] **Step 3: 完整项目测试**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests
```

Expected: 0 failures；不运行根目录无范围 pytest。

- [ ] **Step 4: 机器与依赖复核**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi topo -m
/home/xujie/miniconda3/envs/ret/bin/python -c "import torch, timm; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), timm.__version__)"
df -h /home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224
```

Expected: 5 张 A800 80GB、torch 未改变、timm 0.9.16、NVMe cache 文件系统保留足够空间。

- [ ] **Step 5: 检查变更范围**

Run:

```bash
git status --short
git diff --check 47dc46c..HEAD
git log -5 --oneline
```

Expected: 用户删除的 reference 文档仍未纳入提交；只有计划内实现提交和该既有删除。

- [ ] **Step 6: 交付命令但不执行长期训练**

完整 cache：

```bash
/home/xujie/miniconda3/envs/ret/bin/python scripts/build_msrvtt_tqfs_cache.py --workers 16
```

可信基线：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,4 NPROC=4 TQFS_CACHE_DIR=/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224 bash run_train_msrvtt_bg.sh
```

A800 checkpoint-off 吞吐对照：

```bash
RUN_ID=a800_no_ckpt_$(date +%Y%m%d_%H%M%S) CUDA_VISIBLE_DEVICES=0,1,2,4 NPROC=4 CLIP_GRADIENT_CHECKPOINTING=0 TQFS_CACHE_DIR=/home/xujie/.cache/uatvr/tqfs/msrvtt_trusted_v1_f1_m8_r224 bash run_train_msrvtt_bg.sh
```
