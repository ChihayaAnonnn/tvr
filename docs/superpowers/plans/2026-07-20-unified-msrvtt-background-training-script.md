# Unified MSR-VTT Background Training Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `run_train_msrvtt_bg.sh` 与 `train_msrvtt.sh` 合并为单一后台训练入口，同时保持参数校验、GPU/torchrun 配置、日志重定向、PID 文件和前台日志跟随行为。

**Architecture:** `run_train_msrvtt_bg.sh` 采用 controller/worker 双模式。普通调用进入 controller，由 `setsid` 设置保留的内部环境变量并重新启动同一脚本；worker 检测该变量后执行原 `train_msrvtt.sh` 的数据准备、配置校验和 `torchrun`，从而在一个文件中保持原有后台边界。

**Tech Stack:** Bash、GNU `setsid`、PyTorch `torchrun`、Python 3、pytest、Git

## Global Constraints

- 规格事实源为 `docs/superpowers/specs/2026-07-20-unified-msrvtt-background-training-script-design.md`。
- 合并后只保留 `run_train_msrvtt_bg.sh`；必须删除 `train_msrvtt.sh`。
- 公共启动方式保持 `./run_train_msrvtt_bg.sh [CLI args...]`。
- 后台命令必须使用 `setsid`；`Ctrl-C` 只能停止 `tail`，不得终止训练。
- 日志保持 `logs/${RUN_DATE}/${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}_train_msrvtt.log`。
- `TRAIN_PID_FILE` 非空时必须写入后台会话 PID。
- 内部模式变量固定为 `RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1`，进入 worker 后必须 `unset`，不得泄漏到 `torchrun` 或 Python。
- 所有尾随参数必须通过 `"$@"` 逐项传递，禁止字符串拼接或 `eval`。
- Worker 必须迁移当前工作树版本的 `train_msrvtt.sh`，包括 `msrvtt_trusted_v1_seed0.json` 和 `--run_final_test`；不得从 HEAD 的旧版本恢复。
- 不改变训练默认值、hygiene 保护、A800 检查、GPU 拓扑、split 构建或 `torchrun` 参数。
- 不新增 RSPR shell 环境变量；该工作仍属于 RSPR Task 8。
- 自动化测试不得启动真实 split 构建、`torchrun` 或训练进程。
- 工作树已有修改属于用户；提交只允许包含本任务明确列出的文件和明确迁移的当前脚本行为。

---

### Task 1: 合并 controller/worker、删除旧入口并验证行为等价

**Files:**
- Modify: `run_train_msrvtt_bg.sh:1-43`
- Delete: `train_msrvtt.sh:1-194`
- Create: `tests/test_run_train_msrvtt_bg.py`
- Modify: `scripts/diagnose_msrvtt_hard_negative_runtime.py:202`
- Modify in working tree: `docs/superpowers/plans/2026-07-19-rspr-core-implementation.md:898-1016`

**Interfaces:**
- Public command: `./run_train_msrvtt_bg.sh [main_task_retrieval.py CLI arguments...]`
- Internal dispatch: `RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1 bash run_train_msrvtt_bg.sh [CLI arguments...]`
- Controller inputs: `RUN_DATE`, `RUN_TIME`, `RUN_TAG`, `RUN_ID`, `TRAIN_PID_FILE`, trailing CLI arguments
- Worker inputs: all existing `train_msrvtt.sh` environment variables and trailing CLI arguments
- Worker output: one foreground `torchrun` process whose output is redirected by the controller

- [ ] **Step 1: 记录受保护工作树基线**

Run:

```bash
git rev-parse HEAD
git status --short -- run_train_msrvtt_bg.sh train_msrvtt.sh \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  docs/superpowers/plans/2026-07-19-rspr-core-implementation.md
sha256sum run_train_msrvtt_bg.sh train_msrvtt.sh
git diff -- train_msrvtt.sh scripts/diagnose_msrvtt_hard_negative_runtime.py
```

Expected:

- `run_train_msrvtt_bg.sh` 无任务前修改；
- `train_msrvtt.sh` 的用户差异明确包含 seed42→seed0 和新增 `--run_final_test`；
- `scripts/diagnose_msrvtt_hard_negative_runtime.py` 的其他用户修改被记录，后续不得整文件暂存；
- 暂存区为空。

- [ ] **Step 2: 写 controller/worker 失败测试**

Create `tests/test_run_train_msrvtt_bg.py` with the following helpers and tests:

```python
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "run_train_msrvtt_bg.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _read_nul_arguments(path: Path) -> list[str]:
    payload = path.read_bytes()
    return [item.decode() for item in payload.split(b"\0") if item]


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 2.0
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def _copy_script(tmp_path: Path) -> Path:
    copied = tmp_path / SCRIPT.name
    shutil.copy2(SCRIPT, copied)
    return copied


def _environment(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
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


def test_controller_relaunches_same_script_and_tails_log(tmp_path):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    setsid_arguments = tmp_path / "setsid.args"
    tail_arguments = tmp_path / "tail.args"
    pid_file = tmp_path / "train.pid"
    _write_executable(
        fake_bin / "setsid",
        'printf "%s\\0" "$@" > "$SETSID_ARGUMENTS"\n',
    )
    _write_executable(
        fake_bin / "tail",
        'printf "%s\\0" "$@" > "$TAIL_ARGUMENTS"\n',
    )
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {
            "RUN_DATE": "20260720",
            "RUN_TIME": "121314",
            "RUN_TAG": "rspr",
            "RUN_ID": "20260720_121314_rspr",
            "TRAIN_PID_FILE": str(pid_file),
            "SETSID_ARGUMENTS": str(setsid_arguments),
            "TAIL_ARGUMENTS": str(tail_arguments),
        }
    )

    subprocess.run(
        [
            "bash",
            str(script),
            "--rspr_mode",
            "stochastic",
            "--experiment_desc",
            "two words",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    _wait_for(setsid_arguments)
    setsid_args = _read_nul_arguments(setsid_arguments)
    assert setsid_args == [
        "env",
        "RUN_ID=20260720_121314_rspr",
        "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1",
        "bash",
        str(script),
        "--rspr_mode",
        "stochastic",
        "--experiment_desc",
        "two words",
    ]
    assert _read_nul_arguments(tail_arguments) == [
        "-n",
        "50",
        "-F",
        "logs/20260720/121314_rspr_train_msrvtt.log",
    ]
    assert pid_file.read_text().strip().isdigit()


def test_worker_runs_split_builder_and_torchrun_without_recursing(tmp_path):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python_arguments = tmp_path / "python.args"
    torchrun_arguments = tmp_path / "torchrun.args"
    worker_environment = tmp_path / "worker.env"
    forbidden_call = tmp_path / "forbidden"
    _write_executable(
        fake_bin / "python3",
        'printf "%s\\0" "$@" > "$PYTHON_ARGUMENTS"\n',
    )
    _write_executable(
        fake_bin / "torchrun",
        (
            'printf "%s\\0" "$@" > "$TORCHRUN_ARGUMENTS"\n'
            'printf "%s" "${RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER-unset}" '
            '> "$WORKER_ENVIRONMENT"\n'
        ),
    )
    for command in ("setsid", "tail"):
        _write_executable(
            fake_bin / command,
            'touch "$FORBIDDEN_CALL"\nexit 97\n',
        )
    environment = _environment(tmp_path, fake_bin)
    environment.update(
        {
            "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER": "1",
            "PYTHON_ARGUMENTS": str(python_arguments),
            "TORCHRUN_ARGUMENTS": str(torchrun_arguments),
            "WORKER_ENVIRONMENT": str(worker_environment),
            "FORBIDDEN_CALL": str(forbidden_call),
        }
    )

    subprocess.run(
        [
            "bash",
            str(script),
            "--rspr_mode",
            "stochastic",
            "--experiment_desc",
            "two words",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )

    assert not forbidden_call.exists()
    assert worker_environment.read_text() == "unset"
    python_args = _read_nul_arguments(python_arguments)
    assert python_args[0] == str(tmp_path / "scripts/build_msrvtt_trusted_split.py")
    manifest_index = python_args.index("--manifest")
    assert python_args[manifest_index + 1] == str(
        tmp_path / "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
    )
    torchrun_args = _read_nul_arguments(torchrun_arguments)
    assert "--nproc_per_node=2" in torchrun_args
    assert str(tmp_path / "main_task_retrieval.py") in torchrun_args
    assert "--run_final_test" in torchrun_args
    assert torchrun_args[-4:] == [
        "--rspr_mode",
        "stochastic",
        "--experiment_desc",
        "two words",
    ]


@pytest.mark.parametrize(
    ("environment_update", "arguments", "message"),
    (
        (
            {"EXPERIMENT_PROFILE": "default", "CUDA_VISIBLE_DEVICES": "gpu0"},
            (),
            "malformed CUDA_VISIBLE_DEVICES",
        ),
        (
            {
                "EXPERIMENT_PROFILE": "hygiene",
                "CUDA_VISIBLE_DEVICES": "0,1,2,4",
                "NPROC": "4",
            },
            ("--batch_size", "128"),
            "hygiene cannot override protected baseline option",
        ),
    ),
)
def test_worker_preserves_validation_failures(
    tmp_path, environment_update, arguments, message
):
    script = _copy_script(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    torchrun_marker = tmp_path / "torchrun.called"
    _write_executable(fake_bin / "python3", "exit 0\n")
    _write_executable(
        fake_bin / "torchrun",
        'touch "$TORCHRUN_MARKER"\n',
    )
    environment = _environment(tmp_path, fake_bin)
    environment.update(environment_update)
    environment.update(
        {
            "RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER": "1",
            "TORCHRUN_MARKER": str(torchrun_marker),
        }
    )

    result = subprocess.run(
        ["bash", str(script), *arguments],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not torchrun_marker.exists()


def test_repository_has_one_supported_msrvtt_training_script():
    assert SCRIPT.is_file()
    assert not (REPOSITORY / "train_msrvtt.sh").exists()
    subprocess.run(
        ["bash", "-n", str(SCRIPT), str(REPOSITORY / "run_train_bg.sh")],
        check=True,
    )
```

- [ ] **Step 3: 运行新测试并确认 RED**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_run_train_msrvtt_bg.py
```

Expected: 5 tests are collected and at least these contracts fail for the expected reasons:

- controller still invokes `train_msrvtt.sh` instead of itself;
- internal worker marker is ignored, so the worker test reaches forbidden `setsid/tail`;
- `train_msrvtt.sh` still exists.

Failures caused by missing fake-command environment variables, Python syntax errors, or test collection errors are not valid RED evidence and must be corrected before production edits.

- [ ] **Step 4: 将后台 wrapper 改为双模式单文件骨架**

Replace the wrapper-only structure in `run_train_msrvtt_bg.sh` with this controller and dispatch structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${ROOT_DIR}/run_train_msrvtt_bg.sh"
cd "${ROOT_DIR}"

run_controller() {
    mkdir -p logs

    RUN_DATE="${RUN_DATE:-$(date +%Y%m%d)}"
    RUN_TIME="${RUN_TIME:-$(date +%H%M%S)}"
    RUN_TAG="${RUN_TAG:-}"
    if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Unsupported RUN_TAG=${RUN_TAG}; use letters, digits, dot, underscore, or hyphen" >&2
        return 2
    fi
    RUN_SUFFIX="${RUN_TIME}${RUN_TAG:+_${RUN_TAG}}"
    RUN_ID="${RUN_ID:-${RUN_DATE}_${RUN_SUFFIX}}"
    LOG_DIR="logs/${RUN_DATE}"
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/${RUN_SUFFIX}_train_msrvtt.log"
    TRAIN_PID_FILE="${TRAIN_PID_FILE:-}"

    echo "[run_train_msrvtt_bg] RUN_DATE=${RUN_DATE} RUN_TIME=${RUN_TIME} RUN_TAG=${RUN_TAG}"
    echo "[run_train_msrvtt_bg] LOG_FILE=${LOG_FILE}"
    echo "[run_train_msrvtt_bg] Starting internal training worker (completely detached)"

    setsid env \
        RUN_ID="${RUN_ID}" \
        RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER=1 \
        bash "${SCRIPT_PATH}" "$@" >"${LOG_FILE}" 2>&1 &

    TRAIN_PID=$!
    if [[ -n "${TRAIN_PID_FILE}" ]]; then
        echo "${TRAIN_PID}" > "${TRAIN_PID_FILE}"
    fi
    echo "[run_train_msrvtt_bg] PID=${TRAIN_PID}"
    echo "[run_train_msrvtt_bg] MSRVTT 训练已在后台启动。你可以安全关闭 Cursor。"
    echo "[run_train_msrvtt_bg] 随时可以运行以下命令查看日志："
    echo "tail -f ${LOG_FILE}"

    tail -n 50 -F "${LOG_FILE}"
}


run_worker() {
    unset RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER
    # Step 5 moves the exact current training body here.
}


if [[ "${RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER:-0}" == "1" ]]; then
    run_worker "$@"
else
    run_controller "$@"
fi
```

Do not leave the comment inside the final `run_worker`; Step 5 replaces it with the concrete current training body.

- [ ] **Step 5: 将当前训练实现逐行迁移进 worker 并删除旧脚本**

Move the current working-tree `train_msrvtt.sh` body from immediately after `set -euo pipefail` through its final `torchrun` command into `run_worker()`. Apply only these structural edits:

1. Keep the global `ROOT_DIR` already defined by the merged script; do not redefine it in the worker.
2. Put `unset RUN_TRAIN_MSRVTT_BG_INTERNAL_WORKER` first.
3. Preserve every current environment default, validation branch and argument in its existing order.
4. Preserve exactly:

```bash
SPLIT_MANIFEST="${ROOT_DIR}/dataloaders/splits/msrvtt_trusted_v1_seed0.json"
```

5. Preserve `--do_train --run_final_test` in the `torchrun` command.
6. Preserve `"$@"` as the final command element.
7. Change only the two stale worker log prefixes from `[train_msrvtt.sh]` to `[run_train_msrvtt_bg:worker]`.
8. Delete `train_msrvtt.sh` after its full behavior is present in `run_worker()`.

Use `apply_patch` for both the merged file and deletion. Do not use `cp`, `mv`, heredoc writes, or destructive Git commands.

- [ ] **Step 6: 运行聚焦测试并确认 GREEN**

Run:

```bash
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_run_train_msrvtt_bg.py
```

Expected:

```text
.....                                                                    [100%]
5 passed
```

If a controller test is intermittent, investigate the process boundary and synchronization; do not add arbitrary sleeps or retries beyond `_wait_for`'s bounded file wait.

- [ ] **Step 7: 更新活动引用，不改写历史归档**

Apply these exact reference changes:

- In `scripts/diagnose_msrvtt_hard_negative_runtime.py:202`, change “matching train_msrvtt.sh” to “matching run_train_msrvtt_bg.sh”. Preserve every other user hunk in that file.
- In only Task 8 of `docs/superpowers/plans/2026-07-19-rspr-core-implementation.md`, replace the active training target, examples, syntax checks and `git add` entry from `train_msrvtt.sh`/`./train_msrvtt.sh` to `run_train_msrvtt_bg.sh`/`./run_train_msrvtt_bg.sh`.
- Do not edit the completed 2026-07-12/14 historical plans or the approved merge design's description of the old chain.
- Leave `run_train_bg.sh` unchanged because it already redirects to `run_train_msrvtt_bg.sh`.

Because the current RSPR plan is untracked at task start, update it in the working tree but do not silently stage the entire pre-existing document. Report its pre/post hash and leave its commit decision to the controller.

- [ ] **Step 8: 运行语法、引用和回归验证**

Run:

```bash
bash -n run_train_msrvtt_bg.sh run_train_bg.sh
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_run_train_msrvtt_bg.py
/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests
/home/xujie/.conda/envs/tvr/bin/python -m py_compile scripts/diagnose_msrvtt_hard_negative_runtime.py tests/test_run_train_msrvtt_bg.py
rg -n 'bash train_msrvtt\.sh|\./train_msrvtt\.sh|Modify: `train_msrvtt\.sh`|bash -n train_msrvtt\.sh' \
  run_train_bg.sh run_train_msrvtt_bg.sh scripts docs/project \
  docs/superpowers/plans/2026-07-19-rspr-core-implementation.md
git diff --check
```

Expected:

- both `bash -n` checks exit 0;
- focused suite reports `5 passed`;
- all tests collected under the project's `tests/` directory pass;
- `py_compile` exits 0;
- the active-reference scan has no matches;
- references in historical plans/specs are intentionally outside the scan;
- `git diff --check` exits 0.

- [ ] **Step 9: 审核工作区隔离并提交实现**

Before staging, inspect:

```bash
git status --short
git diff -- run_train_msrvtt_bg.sh train_msrvtt.sh \
  tests/test_run_train_msrvtt_bg.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py
```

Stage only the implementation files and only the one diagnostic-docstring hunk. The diagnostic file already contains an unrelated manifest-path edit, so apply the task-owned one-line patch directly to the index instead of staging the whole file.

```bash
git add run_train_msrvtt_bg.sh train_msrvtt.sh tests/test_run_train_msrvtt_bg.py
git apply --cached --unidiff-zero <<'PATCH'
diff --git a/scripts/diagnose_msrvtt_hard_negative_runtime.py b/scripts/diagnose_msrvtt_hard_negative_runtime.py
--- a/scripts/diagnose_msrvtt_hard_negative_runtime.py
+++ b/scripts/diagnose_msrvtt_hard_negative_runtime.py
@@ -202 +202 @@
-    """Build a main_task_retrieval Namespace matching train_msrvtt.sh."""
+    """Build a main_task_retrieval Namespace matching run_train_msrvtt_bg.sh."""
PATCH
git diff --cached --name-status
git diff --cached --check
git commit -m "refactor: unify msrvtt background training entry"
```

Expected staged paths:

```text
M  run_train_msrvtt_bg.sh
M  scripts/diagnose_msrvtt_hard_negative_runtime.py
D  train_msrvtt.sh
A  tests/test_run_train_msrvtt_bg.py
```

After committing:

```bash
git diff --cached --quiet
git show --name-status --format='%H%n%s' HEAD
git status --short
```

Expected:

- index is empty;
- commit contains exactly the four implementation paths above;
- unrelated user modifications remain in the working tree;
- the updated untracked RSPR plan remains unstaged and is called out in the handoff.
