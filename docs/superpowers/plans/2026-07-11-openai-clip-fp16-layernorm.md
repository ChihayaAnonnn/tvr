# OpenAI CLIP FP16 LayerNorm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 OpenAI CLIP 自定义 LayerNorm 默认在 CUDA FP16 输入上使用 native FP16，同时保留可追溯的 FP32 回退，并验证数值、梯度和显存行为。

**Architecture:** `modules.module_clip.LayerNorm` 负责算子级精度选择，保留 FP32 master affine 参数并仅在 forward 转换小向量。`UATVR` 将 CLI 配置传播到 OpenAI CLIP 内的自定义 LayerNorm；训练/评估脚本与 experiment manifest 显式记录该配置。EVA、全模型 AMP、activation checkpointing 和检索目标均保持不变。

**Tech Stack:** Python 3.10、PyTorch 2.1.2、CUDA 12.1、A800、pytest、Ruff、Bash、Git。

## Global Constraints

- 默认精度必须是 `fp16`，唯一回退值为 `fp32`。
- 仅 `CUDA + torch.float16` 输入走 native FP16 LayerNorm；FP32、CPU FP16 和 BF16 输入走旧 FP32 路径。
- LayerNorm 的 master `weight`/`bias` 保持 FP32；不得永久调用 `.half()` 修改参数存储。
- 不接入 autocast、GradScaler、Apex AMP、activation checkpointing、GradCache 或 backbone 冻结。
- 不改变 checkpoint 参数键、WTI、多正例 InfoNCE、SAP、EVA adapter 或数据协议。
- 精度配置必须写入有效参数日志、训练/评估脚本输出和 `experiment_manifest.json`。
- 不启动长期训练；只运行单元测试、受控 CUDA 显存测试和 CPU/CUDA smoke。

---

## File Structure

- `modules/module_clip.py`：定义 LayerNorm 精度契约、native FP16/旧 FP32 两条计算路径及批量配置 helper。
- `modules/modeling_mulit.py`：读取 task config，并只对 OpenAI CLIP 实例传播 LayerNorm 精度。
- `main_task_retrieval.py`：定义 CLI 默认值和有效参数日志字段。
- `experiment_tracking.py`：将精度写入安全 allowlist provenance。
- `train_msrvtt.sh`、`eval.sh`：提供默认环境变量、合法值校验、显式参数传递和启动日志。
- `tests/test_clip_layer_norm_precision.py`：算子级 CPU/CUDA、数值、梯度及 saved-tensor 门禁。
- `tests/test_modeling_mulit_losses.py`：模型配置传播与 EVA 不受影响的回归。
- `tests/test_main_task_hard_negative_args.py`：CLI 默认/回退及 shell 参数传递。
- `tests/test_experiment_tracking.py`：manifest provenance 回归。
- `AGENTS.md`、设计规格：记录最终默认行为和回退方式。

---

### Task 1: 建立 LayerNorm 精度内核与算子级回归

**Files:**
- Modify: `modules/module_clip.py:236-243`
- Create: `tests/test_clip_layer_norm_precision.py`

**Interfaces:**
- Produces: `validate_layer_norm_precision(value: str) -> str`
- Produces: `LayerNorm(..., precision: str = "fp16")`
- Produces: `LayerNorm.set_precision(precision: str) -> None`
- Produces: `set_layer_norm_precision(module: nn.Module, precision: str) -> int`

- [ ] **Step 1: 写默认值、回退值和非法值的失败测试**

Create `tests/test_clip_layer_norm_precision.py`:

```python
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from modules.module_clip import LayerNorm, set_layer_norm_precision


def _legacy_fp32(layer, value):
    return F.layer_norm(
        value.float(),
        layer.normalized_shape,
        layer.weight.float() if layer.weight is not None else None,
        layer.bias.float() if layer.bias is not None else None,
        layer.eps,
    ).to(value.dtype)


def test_layer_norm_defaults_to_fp16_and_accepts_fp32_fallback():
    layer = LayerNorm(8)
    assert layer.precision == "fp16"
    layer.set_precision("fp32")
    assert layer.precision == "fp32"


def test_layer_norm_rejects_unknown_precision():
    with pytest.raises(ValueError, match="layer norm precision"):
        LayerNorm(8, precision="tf32")


def test_set_layer_norm_precision_only_updates_custom_layers():
    module = nn.Sequential(LayerNorm(8), nn.LayerNorm(8), LayerNorm(8))
    assert set_layer_norm_precision(module, "fp32") == 2
    assert module[0].precision == "fp32"
    assert module[2].precision == "fp32"
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_clip_layer_norm_precision.py
```

Expected: FAIL because `LayerNorm` lacks `precision` and `set_layer_norm_precision` is not defined.

- [ ] **Step 3: 实现精度校验、默认值和批量配置 helper**

In `modules/module_clip.py`, replace the current `LayerNorm` definition and add:

```python
LAYER_NORM_PRECISIONS = frozenset({"fp16", "fp32"})


def validate_layer_norm_precision(value):
    value = str(value).lower()
    if value not in LAYER_NORM_PRECISIONS:
        raise ValueError(
            f"unsupported layer norm precision={value}; expected fp16 or fp32"
        )
    return value


class LayerNorm(nn.LayerNorm):
    """CLIP LayerNorm with configurable CUDA FP16 execution."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, precision="fp16"):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.precision = validate_layer_norm_precision(precision)

    def set_precision(self, precision):
        self.precision = validate_layer_norm_precision(precision)

    def forward(self, x):
        precision = validate_layer_norm_precision(self.precision)
        if precision == "fp16" and x.is_cuda and x.dtype == torch.float16:
            weight = None if self.weight is None else self.weight.to(dtype=x.dtype)
            bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
            return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
        original_dtype = x.dtype
        result = F.layer_norm(
            x.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        )
        return result.to(dtype=original_dtype)


def set_layer_norm_precision(module, precision):
    precision = validate_layer_norm_precision(precision)
    configured = 0
    for child in module.modules():
        if isinstance(child, LayerNorm):
            child.set_precision(precision)
            configured += 1
    return configured
```

- [ ] **Step 4: 写 CPU 行为与 FP32 等价测试**

Append:

```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cpu_inputs_use_legacy_fp32_computation(dtype):
    layer = LayerNorm(8, precision="fp16")
    value = torch.randn(4, 8, dtype=dtype, requires_grad=True)
    torch.testing.assert_close(layer(value), _legacy_fp32(layer, value), rtol=0, atol=0)


def test_fp32_mode_matches_legacy_formula():
    layer = LayerNorm(8, precision="fp32")
    value = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    torch.testing.assert_close(layer(value), _legacy_fp32(layer, value))
```

- [ ] **Step 5: 写 CUDA FP16 数值、梯度和 saved-tensor 门禁**

Append:

```python
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@CUDA_REQUIRED
@pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (1e-4, 0.0), (1e4, 0.0), (1.0, 1e4)])
def test_cuda_fp16_is_finite_and_close_to_fp32(scale, offset):
    fp16_layer = LayerNorm(768, precision="fp16").cuda()
    fp32_layer = LayerNorm(768, precision="fp32").cuda()
    with torch.no_grad():
        fp16_layer.weight.uniform_(0.5, 1.5)
        fp16_layer.bias.uniform_(-0.2, 0.2)
    fp32_layer.load_state_dict(fp16_layer.state_dict())
    value_fp16 = (
        torch.randn(32, 16, 768, device="cuda", dtype=torch.float16) * scale + offset
    ).requires_grad_(True)
    value_fp32 = value_fp16.detach().clone().requires_grad_(True)
    actual = fp16_layer(value_fp16)
    expected = fp32_layer(value_fp32)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-2)
    probe = torch.randn_like(actual, dtype=torch.float32)
    (actual.float() * probe).sum().backward()
    (expected.float() * probe).sum().backward()
    assert torch.isfinite(value_fp16.grad).all()
    cosine = F.cosine_similarity(
        value_fp16.grad.float().flatten()[None],
        value_fp32.grad.float().flatten()[None],
    ).item()
    assert cosine >= 0.9999


@CUDA_REQUIRED
def test_cuda_fp16_does_not_save_full_fp32_input_copy():
    layer = LayerNorm(768, precision="fp16").cuda()
    value = torch.randn(64, 16, 768, device="cuda", dtype=torch.float16, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: saved.append((tensor.dtype, tensor.numel())) or tensor,
        lambda tensor: tensor,
    ):
        layer(value).float().sum().backward()
    assert (torch.float32, value.numel()) not in saved
    assert layer.weight.dtype == torch.float32
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype == torch.float32
```

- [ ] **Step 6: 运行 Task 1 测试与静态检查**

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_clip_layer_norm_precision.py
/home/xujie/miniconda3/envs/ret/bin/ruff check modules/module_clip.py tests/test_clip_layer_norm_precision.py
python3 -m py_compile modules/module_clip.py tests/test_clip_layer_norm_precision.py
git diff --check
```

Expected: all tests PASS; static checks exit 0.

- [ ] **Step 7: 提交核心内核**

```bash
git add modules/module_clip.py tests/test_clip_layer_norm_precision.py
git diff --cached --check
git commit -m "perf: add native FP16 CLIP LayerNorm"
```

---

### Task 2: 将精度配置接入 UATVR 与 CLI

**Files:**
- Modify: `main_task_retrieval.py:299-333,653-660`
- Modify: `modules/modeling_mulit.py:15,297-397`
- Modify: `tests/test_main_task_hard_negative_args.py`
- Modify: `tests/test_modeling_mulit_losses.py`

**Interfaces:**
- Consumes: `validate_layer_norm_precision(value) -> str`
- Consumes: `set_layer_norm_precision(module, precision) -> int`
- Produces: `args.clip_layer_norm_precision: Literal["fp16", "fp32"]`
- Produces: `UATVR.clip_layer_norm_precision: str`
- Produces: `UATVR.clip_layer_norm_module_count: int`

- [ ] **Step 1: 写 CLI 默认值和显式回退的失败测试**

Append to `tests/test_main_task_hard_negative_args.py`:

```python
def test_clip_layer_norm_precision_defaults_to_fp16(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--do_train", "--output_dir", "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences",
        ],
    )
    assert get_args().clip_layer_norm_precision == "fp16"


def test_clip_layer_norm_precision_accepts_fp32(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog", "--do_train", "--output_dir", "/tmp/uatvr-test-out",
            "--expand_msrvtt_sentences", "--clip_layer_norm_precision", "fp32",
        ],
    )
    assert get_args().clip_layer_norm_precision == "fp32"
```

- [ ] **Step 2: 运行 CLI 测试并确认 RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_main_task_hard_negative_args.py -k clip_layer_norm_precision
```

Expected: FAIL because the parser does not expose `clip_layer_norm_precision`.

- [ ] **Step 3: 新增 CLI 并加入有效参数日志**

Add after `--backbone_type` in `main_task_retrieval.py`:

```python
parser.add_argument(
    "--clip_layer_norm_precision",
    default="fp16",
    choices=["fp16", "fp32"],
    help=(
        "Execution precision for project OpenAI CLIP LayerNorm modules. "
        "fp16 is used only for CUDA FP16 inputs; fp32 preserves legacy behavior."
    ),
)
```

Add `"clip_layer_norm_precision"` to the `Model` keys in `set_seed_logger`.

- [ ] **Step 4: 写 OpenAI 传播和 EVA 不生效的失败测试**

Append to `tests/test_modeling_mulit_losses.py`:

```python
from modules.module_clip import LayerNorm


def test_openai_clip_layer_norm_precision_is_propagated():
    clip = torch.nn.Sequential(LayerNorm(4), LayerNorm(4))
    count = UATVR.configure_clip_layer_norm_precision(
        clip, backbone_type="openai_clip", precision="fp32"
    )
    assert count == 2
    assert all(module.precision == "fp32" for module in clip)


def test_eva_backbone_is_not_modified_by_clip_layer_norm_precision():
    backbone = torch.nn.Sequential(torch.nn.LayerNorm(4))
    assert UATVR.configure_clip_layer_norm_precision(
        backbone, backbone_type="eva_clip", precision="fp16"
    ) == 0
```

- [ ] **Step 5: 实现模型配置传播**

Update imports in `modules/modeling_mulit.py`:

```python
from modules.module_clip import (
    CLIP,
    convert_weights,
    set_layer_norm_precision,
    validate_layer_norm_precision,
)
```

Add to `UATVR`:

```python
@staticmethod
def configure_clip_layer_norm_precision(backbone, backbone_type, precision):
    precision = validate_layer_norm_precision(precision)
    if backbone_type != "openai_clip":
        return 0
    return set_layer_norm_precision(backbone, precision)
```

At the start of `UATVR.__init__`, after resolving `self.backbone_type`:

```python
self.clip_layer_norm_precision = validate_layer_norm_precision(
    getattr(task_config, "clip_layer_norm_precision", "fp16")
)
self.clip_layer_norm_module_count = 0
```

Immediately after `convert_weights(self.clip)`:

```python
self.clip_layer_norm_module_count = self.configure_clip_layer_norm_precision(
    self.clip,
    backbone_type=self.backbone_type,
    precision=self.clip_layer_norm_precision,
)
show_log(
    task_config,
    "\t OpenAI CLIP LayerNorm precision: {} (modules={})".format(
        self.clip_layer_norm_precision,
        self.clip_layer_norm_module_count,
    ),
)
```

In the EVA branch, log that the option does not alter EVA:

```python
show_log(
    task_config,
    "\t OpenAI CLIP LayerNorm precision: {} (not applied to EVA)".format(
        self.clip_layer_norm_precision
    ),
)
```

- [ ] **Step 6: 运行模型与 CLI 回归**

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_clip_layer_norm_precision.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py
/home/xujie/miniconda3/envs/ret/bin/ruff check modules/module_clip.py modules/modeling_mulit.py main_task_retrieval.py tests/test_clip_layer_norm_precision.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py
python3 -m py_compile modules/module_clip.py modules/modeling_mulit.py main_task_retrieval.py
git diff --check
```

Expected: all tests PASS and static checks exit 0.

- [ ] **Step 7: 提交模型和 CLI 接入**

```bash
git add main_task_retrieval.py modules/modeling_mulit.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py
git diff --cached --check
git commit -m "feat: default OpenAI CLIP LayerNorm to FP16"
```

---

### Task 3: 接入脚本、provenance 与回退入口

**Files:**
- Modify: `experiment_tracking.py:125-158`
- Modify: `train_msrvtt.sh:20-65,95-125`
- Modify: `eval.sh:35-60,105-145,155-185`
- Modify: `tests/test_experiment_tracking.py`
- Modify: `tests/test_main_task_hard_negative_args.py`

**Interfaces:**
- Consumes: `args.clip_layer_norm_precision`
- Produces: `experiment_manifest.json["backbone"]["clip_layer_norm_precision"]`
- Produces: environment variable `CLIP_LAYER_NORM_PRECISION=fp16|fp32`

- [ ] **Step 1: 写 manifest 和 shell 参数传播的失败测试**

In `tests/test_experiment_tracking.py`, add `clip_layer_norm_precision="fp16"` to `_args` and assert:

```python
assert payload["backbone"]["clip_layer_norm_precision"] == "fp16"
```

Extend `_run_with_fake_torchrun` in `tests/test_main_task_hard_negative_args.py` with a `layer_norm_precision="fp16"` argument and environment entry:

```python
"CLIP_LAYER_NORM_PRECISION": layer_norm_precision,
```

Append:

```python
@pytest.mark.parametrize("script_name", ["train_msrvtt.sh", "eval.sh"])
@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_scripts_forward_and_log_clip_layer_norm_precision(
    script_name, precision, tmp_path
):
    result, capture_path = _run_with_fake_torchrun(
        script_name, tmp_path, "0", layer_norm_precision=precision
    )
    assert result.returncode == 0, result.stderr
    captured_args = capture_path.read_text(encoding="utf-8").splitlines()
    index = captured_args.index("--clip_layer_norm_precision")
    assert captured_args[index + 1] == precision
    assert f"CLIP_LAYER_NORM_PRECISION={precision}" in result.stdout


@pytest.mark.parametrize("script_name", ["train_msrvtt.sh", "eval.sh"])
def test_scripts_reject_invalid_clip_layer_norm_precision(script_name, tmp_path):
    result, capture_path = _run_with_fake_torchrun(
        script_name, tmp_path, "0", layer_norm_precision="tf32"
    )
    assert result.returncode == 2
    assert "CLIP_LAYER_NORM_PRECISION=tf32" in result.stderr
    assert not capture_path.exists()
```

- [ ] **Step 2: 运行测试并确认 RED**

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_experiment_tracking.py tests/test_main_task_hard_negative_args.py -k "layer_norm_precision or manifest_contains"
```

Expected: FAIL because scripts and manifest do not expose the precision.

- [ ] **Step 3: 将精度加入 provenance**

Update the `backbone` allowlist in `experiment_tracking.py`:

```python
backbone = {
    "type": getattr(args, "backbone_type", ""),
    "pretrained_clip_name": getattr(args, "pretrained_clip_name", ""),
    "name": getattr(args, "backbone_name", ""),
    "path": getattr(args, "backbone_path", ""),
    "clip_layer_norm_precision": getattr(
        args, "clip_layer_norm_precision", "fp16"
    ),
}
```

- [ ] **Step 4: 修改训练脚本的默认值、校验、日志和 CLI**

In `train_msrvtt.sh`, add alongside backbone variables:

```bash
CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION:-fp16}
if [[ "${CLIP_LAYER_NORM_PRECISION}" != "fp16" && "${CLIP_LAYER_NORM_PRECISION}" != "fp32" ]]; then
    echo "Unsupported CLIP_LAYER_NORM_PRECISION=${CLIP_LAYER_NORM_PRECISION}; expected fp16 or fp32" >&2
    exit 2
fi
```

Include it in the existing backbone echo and pass:

```bash
--clip_layer_norm_precision "${CLIP_LAYER_NORM_PRECISION}" \
```

- [ ] **Step 5: 修改评估脚本的默认值、校验、日志和 CLI**

In `eval.sh`, add the same default and validation next to the backbone variables, append:

```bash
EXTRA_ARGS+=(--clip_layer_norm_precision "${CLIP_LAYER_NORM_PRECISION}")
```

and include the value in the backbone log line.

- [ ] **Step 6: 运行 provenance 与脚本回归**

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_experiment_tracking.py tests/test_main_task_hard_negative_args.py
/home/xujie/miniconda3/envs/ret/bin/ruff check experiment_tracking.py tests/test_experiment_tracking.py tests/test_main_task_hard_negative_args.py
bash -n train_msrvtt.sh
bash -n eval.sh
git diff --check
```

Expected: tests PASS; Ruff, shell syntax and diff check exit 0.

- [ ] **Step 7: 提交 provenance 和脚本入口**

```bash
git add experiment_tracking.py train_msrvtt.sh eval.sh tests/test_experiment_tracking.py tests/test_main_task_hard_negative_args.py
git diff --cached --check
git commit -m "feat: track CLIP LayerNorm precision"
```

---

### Task 4: 文档、CUDA 显存门禁与完整交付验证

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-07-11-openai-clip-fp16-layernorm-design.md`
- Verify: all files from Tasks 1-3

**Interfaces:**
- Consumes: default `CLIP_LAYER_NORM_PRECISION=fp16`
- Produces: verified FP16 default, FP32 rollback command and final clean worktree

- [ ] **Step 1: 更新项目事实与设计状态**

Add to `AGENTS.md` Learned Workspace Facts:

```markdown
- (07-11) OpenAI CLIP 自定义 LayerNorm 默认使用 native FP16；FP32 master affine 参数保留。可通过 `CLIP_LAYER_NORM_PRECISION=fp32` 回退。该设置不等于全模型 AMP，且不影响 EVA 自身 LayerNorm。
```

Change the design spec status from `已确认，待实施` to `已实施`.

- [ ] **Step 2: 运行聚焦与项目全量测试**

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_clip_layer_norm_precision.py tests/test_modeling_mulit_losses.py tests/test_main_task_hard_negative_args.py tests/test_experiment_tracking.py tests/test_backbone_adapter.py
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests
```

Expected: all project tests PASS. Do not use root-level `pytest -q`, because `ref/InternVideo` contains external tests with optional `libmr`.

- [ ] **Step 3: 运行受控 CUDA 显存与数值 smoke**

```bash
CUDA_VISIBLE_DEVICES=3 /home/xujie/miniconda3/envs/ret/bin/pytest -q tests/test_clip_layer_norm_precision.py -k cuda
```

Expected: CUDA FP16 output/gradient tests PASS; saved-tensor test confirms no full-size FP32 input copy. After completion, `nvidia-smi` shows no residual test process.

- [ ] **Step 4: 运行静态和脚本检查**

```bash
/home/xujie/miniconda3/envs/ret/bin/ruff check modules/module_clip.py modules/modeling_mulit.py main_task_retrieval.py experiment_tracking.py tests/test_clip_layer_norm_precision.py tests/test_modeling_mulit_losses.py tests/test_main_task_hard_negative_args.py tests/test_experiment_tracking.py
python3 -m py_compile modules/module_clip.py modules/modeling_mulit.py main_task_retrieval.py experiment_tracking.py
bash -n train_msrvtt.sh
bash -n eval.sh
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 5: 提交文档状态**

```bash
git add AGENTS.md docs/superpowers/specs/2026-07-11-openai-clip-fp16-layernorm-design.md
git diff --cached --check
git commit -m "docs: record default FP16 CLIP LayerNorm"
```

- [ ] **Step 6: 审计提交与工作树**

```bash
git log --oneline -6
git status --short --branch
git diff --check
```

Expected: four implementation commits after design commit; worktree clean.

- [ ] **Step 7: 交付用户手动运行命令，不执行长期训练**

Default FP16:

```bash
EXPERIMENT_PROFILE=hygiene FINAL_SCORE_MODE=wti CLIP_LAYER_NORM_PRECISION=fp16 RUN_ID=20260711_trusted_v1_openai_hygiene_1gpu_fp16ln CUDA_VISIBLE_DEVICES=3 NPROC=1 bash run_train_msrvtt_bg.sh
```

FP32 rollback:

```bash
EXPERIMENT_PROFILE=hygiene FINAL_SCORE_MODE=wti CLIP_LAYER_NORM_PRECISION=fp32 RUN_ID=20260711_trusted_v1_openai_hygiene_1gpu_fp32ln CUDA_VISIBLE_DEVICES=3 NPROC=1 bash run_train_msrvtt_bg.sh --gradient_accumulation_steps 4
```

Do not run either command during implementation.
