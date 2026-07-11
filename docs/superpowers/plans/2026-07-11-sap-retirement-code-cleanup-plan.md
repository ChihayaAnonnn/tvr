# SAP Retirement Code Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从受支持的检索入口彻底删除 SAP 及其旧概率/不确定性依赖链，将训练和评估收敛为 CLIP/EVA token features → WTI → 双向多正例 InfoNCE，同时保留 trusted-v1 与独立 hard-negative 诊断能力。

**Architecture:** `modules/modeling_mulit.py` 只保留确定性 WTI 主分数、精确 `video_id` 多正例损失和可选 hard-negative loss；`main_task_retrieval.py` 删除旧 score/loss/uncertainty/MoE CLI 与日志，同时保留只消费最终 WTI 矩阵的 MUS 诊断。SpatialEnhancer 源文件保留，但活动接线与 CLI 删除，因为它只修改原先由 SAP 消费的 patch hidden，SAP 删除后不会影响 WTI frame tokens；保留该 knob 会制造无效 causal variable。

**Tech Stack:** Python 3、PyTorch、pytest、ruff、Bash、Git、`rg`。

## Global Constraints

- 先完成 `docs/superpowers/plans/2026-07-11-research-synthesis-and-ssot-plan.md`，确保 Roadmap 使用过渡墓碑。
- 不启动训练；训练验收只给用户单行命令，由用户手动运行。
- 不运行根目录无范围的 `pytest -q`；完整测试固定为 `/home/xujie/miniconda3/envs/ret/bin/pytest -q tests`。
- 不删除 `prob_models/`、`modules/modeling.py`、`query_models/module_query.py`、`query_models/head.py` 或 `modules/spatial_enhancer.py`。
- 保留 `modules/spatial_enhancer.py` 文件，但删除 `modules/modeling_mulit.py` 的 import、构造、patch 处理以及 `rope_mode`/deprecated disable CLI；这是对“保留代码”的源文件级落实，不保留对 WTI 无效的活动 knob。
- 保留 `text_weight_fc.*`、`video_weight_fc.*`、`transformerClip.*` checkpoint 键；它们属于 WTI/seqTransf。
- 保留 hard-negative packing、explicit loss、mapping、seed 与 `w_hard_negative`，但不恢复 sweep/repeat 自动启动脚本。
- `experiment_profile=hygiene` 继续约束 trusted-v1；训练和评估分数固定为 WTI，不保留单选项 `final_score_mode=wti` 空壳。
- 旧 CLI 必须由 argparse 返回 `SystemExit(2)`；不提供 alias、兼容 stub 或静默归零。
- 含任何已删除活动参数（SAP/概率/SpatialEnhancer）的 checkpoint 必须在 `init_preweight` 之前明确拒绝；不做自动迁移。
- `log_mus_scores` 与 `modules/mus_util.py` 保留：它们只消费最终 WTI 相似度矩阵，属于 P1 风险诊断，不是 SAP/probability 分支。
- 不修改或提交当前用户改动：`.gitignore`、`docs/reference/uatvr_backbone_upgrade_strategy.md` 的删除、`research_refs/`。
- 所有手工文件编辑使用 `apply_patch`；tracked 历史产物目录的批量机械删除可使用精确路径的 `git rm -r -- search_results`。

---

### Task 1: 锁定并实现旧 checkpoint 拒绝契约

**Files:**

- Modify: `modules/modeling_mulit.py:53-110`
- Modify: `main_task_retrieval.py:724-887`
- Test: `tests/test_modeling_mulit_losses.py:263-307`

**Interfaces:**

- Consumes: 原始 `state_dict: Mapping[str, Tensor]`；retired-key detector 必须识别可选单层 `module.` 前缀，不宣称整个 loader 会为活动键迁移该前缀。
- Produces: `CLIP4ClipPreTrainedModel._validate_retired_checkpoint_keys(state_dict) -> None`；命中旧前缀时抛 `ValueError`，否则不修改输入。

- [ ] **Step 1: 写 checkpoint 拒绝与允许边界测试**

在现有 checkpoint contract 测试区加入：

```python
@pytest.mark.parametrize(
    "key",
    [
        "sap.anchor_tokens",
        "module.sap.anchor_tokens",
        "qc_sap_text_proj.weight",
        "qc_sap_anchor_proj.weight",
        "pie_net_text.query_proj.weight",
        "uncertain_net_text.fc_logsigma.weight",
        "ada_norm_text.scale_net.weight",
        "expansion_tokens",
        "module.expansion_tokens",
        "spatial_enhancer.attention.qkvo.weight",
    ],
)
def test_retired_auxiliary_checkpoint_keys_are_rejected(key):
    with pytest.raises(ValueError, match=r"retired auxiliary checkpoint"):
        UATVR._validate_retired_checkpoint_keys({key: torch.ones(1)})


@pytest.mark.parametrize(
    "key",
    [
        "text_weight_fc.0.weight",
        "video_weight_fc.0.weight",
        "transformerClip.resblocks.0.attn.in_proj_weight",
        "clip.visual.conv1.weight",
        "clip.visual.patch_embed.proj.weight",
    ],
)
def test_active_wti_checkpoint_keys_are_not_rejected(key):
    UATVR._validate_retired_checkpoint_keys({key: torch.ones(1)})


def test_from_pretrained_rejects_retired_keys_before_backbone_loading(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("backbone validation must not run first")

    monkeypatch.setattr(UATVR, "_validate_checkpoint_backbone", fail_if_called)
    task_config = types.SimpleNamespace(backbone_type="openai_clip", local_rank=0)
    with pytest.raises(ValueError, match=r"retired auxiliary checkpoint"):
        UATVR.from_pretrained(
            "cross-base",
            state_dict={"sap.anchor_tokens": torch.ones(1)},
            task_config=task_config,
        )
```

把当前 `test_checkpoint_backbone_contract_allows_upper_layers_without_backbone_identity` 中的 `sap.anchor_queries` 改为 `text_weight_fc.0.weight`；backbone identity 测试不能再把退役键定义为允许。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_modeling_mulit_losses.py \
  -k 'retired_auxiliary_checkpoint or active_wti_checkpoint or retired_keys_before_backbone or checkpoint_backbone_contract'
```

Expected: FAIL，`UATVR` 尚无 `_validate_retired_checkpoint_keys`。

- [ ] **Step 3: 在统一加载入口实现前置拒绝**

在 `CLIP4ClipPreTrainedModel` 中加入以下常量与方法：

```python
    RETIRED_CHECKPOINT_PREFIXES = (
        "sap.",
        "qc_sap_text_proj.",
        "qc_sap_anchor_proj.",
        "pie_net_text.",
        "uncertain_net_text.",
        "ada_norm_text.",
        "spatial_enhancer.",
    )
    RETIRED_CHECKPOINT_ROOT_KEYS = frozenset({"expansion_tokens"})

    @classmethod
    def _validate_retired_checkpoint_keys(cls, state_dict):
        matched = []
        for raw_key in state_dict:
            key = raw_key[7:] if raw_key.startswith("module.") else raw_key
            if key in cls.RETIRED_CHECKPOINT_ROOT_KEYS or key.startswith(
                cls.RETIRED_CHECKPOINT_PREFIXES
            ):
                matched.append(raw_key)
        if matched:
            preview = sorted(matched)[:8]
            raise ValueError(
                "retired auxiliary checkpoint is unsupported; "
                f"matched_keys={preview}, total={len(matched)}. "
                "No automatic migration is provided; use the matching Git revision."
            )
```

在 `from_pretrained` 中，`state_dict` 归一化为空字典之后、调用 `_validate_checkpoint_backbone` 之前，按以下顺序校验：

```python
        if state_dict is None:
            state_dict = {}
        cls._validate_retired_checkpoint_keys(state_dict)
        backbone_type = getattr(task_config, "backbone_type", "openai_clip")
        cls._validate_checkpoint_backbone(state_dict, backbone_type)
```

在 `load_model` 中把 `logger.info("Model loaded from %s", model_file)` 移到 `UATVR.from_pretrained(...)` 和 `model.to(device)` 成功之后，避免拒绝失败前打印“loaded”。

- [ ] **Step 4: 运行 checkpoint 契约测试确认 GREEN**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_modeling_mulit_losses.py \
  -k 'retired_auxiliary_checkpoint or active_wti_checkpoint or retired_keys_before_backbone or checkpoint_backbone_contract'
```

Expected: PASS。

- [ ] **Step 5: 静态检查并提交**

Run:

```bash
set -euo pipefail
/home/xujie/miniconda3/envs/ret/bin/ruff check \
  modules/modeling_mulit.py main_task_retrieval.py tests/test_modeling_mulit_losses.py
git diff --check -- \
  modules/modeling_mulit.py main_task_retrieval.py tests/test_modeling_mulit_losses.py
```

Expected: 退出码 0。

Commit:

```bash
git add -- modules/modeling_mulit.py main_task_retrieval.py tests/test_modeling_mulit_losses.py
git commit -m "feat: reject retired SAP probability checkpoints" -- \
  modules/modeling_mulit.py main_task_retrieval.py tests/test_modeling_mulit_losses.py
```

---

### Task 2: 将活动模型收敛为 WTI 与独立 hard-negative

**Files:**

- Modify: `modules/modeling_mulit.py:1-1885`
- Modify: `main_task_retrieval.py:1542-1805`
- Modify: `scripts/build_msrvtt_model_mined_hard_negatives.py:592-690`
- Modify: `scripts/diagnose_msrvtt_hard_negative_runtime.py:370-410`
- Modify: `scripts/diagnose_msrvtt_validation_errors.py:330-375`
- Modify: `tests/test_modeling_mulit_losses.py:19-1106`
- Create: `tests/test_prob_models_legacy.py`
- Delete: `tests/test_eval_optional_features.py`
- Test: `tests/test_build_msrvtt_model_mined_hard_negatives.py`
- Test: `tests/test_diagnose_hard_negative_runtime.py`
- Test: `tests/test_diagnose_msrvtt_validation_errors.py`

**Interfaces:**

- Consumes: `text_token [Bt,T,D]`、`visual_output [Bv,V,D]`、二值 masks、训练时 `video_group_id [B]`，以及可选 hard-negative video tokens/mask/valid。
- Produces: 推理时 `WTI logits [Bt,Bv]`；训练时字典 `retrieve_logits`、`video_group_id`、`hard_negative_loss`。顶层 `forward` 返回 `total`、`sim_loss`、`hard_negative_loss` 与三项 batch protocol telemetry。

- [ ] **Step 1: 写唯一 WTI 模型图的失败测试**

删除测试文件顶部对 `PIENet` 与 uncertainty module 的活动主模型导入，并加入：

把仍独立验证 `prob_models/` 历史实现的两个测试移到新文件，不让它们继续冒充活动 `UATVR` 测试：

```python
import torch

from prob_models.pie_model import PIENet
from prob_models.uncertainty_module import UncertaintyModuleText


def test_pienet_padding_mask_removes_padding_from_attention():
    torch.manual_seed(0)
    pie = PIENet(1, 4, 4, 2)
    out = torch.zeros(1, 4)
    x = torch.randn(1, 4, 4)
    pad_mask = torch.tensor([[False, False, True, True]])
    _out, attn, _residual = pie(out, x, pad_mask=pad_mask)
    assert torch.allclose(attn[0, 2:, 0], torch.zeros(2), atol=1e-6)
    assert torch.isclose(attn[0, :2, 0].sum(), torch.tensor(1.0), atol=1e-6)


def test_uncertainty_module_text_uses_true_padding_mask_for_lengths():
    torch.manual_seed(0)
    module = UncertaintyModuleText(4, 4, 2)
    out = torch.randn(2, 4)
    x = torch.randn(2, 5, 4)
    pad_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, True, True, True, True],
        ]
    )
    result = module(out, x, pad_mask=pad_mask)
    assert result["logsigma"].shape == (2, 4)
    assert torch.allclose(
        result["attention"][0, 3:, 0], torch.zeros(2), atol=1e-6
    )
    assert torch.allclose(
        result["attention"][1, 1:, 0], torch.zeros(4), atol=1e-6
    )
```

同时明确删除或改写以下旧活动模型测试：

- 删除 `test_hygiene_similarity_skips_sap_and_probability_modules`，由新的唯一 `_wti_similarity` score-source 测试替代；
- 删除 `test_hygiene_eva_capability_requires_text_hidden_even_without_visual_hidden` 与 `test_hygiene_eva_capability_allows_missing_visual_hidden_for_wti`，由新的单参数 capability 测试替代；
- 删除 `test_default_eva_capability_requires_visual_hidden_for_sap`；
- 删除 `test_model_weight_defaults_match_uncertainty_only_setting`；
- 删除所有 evidential/UACL/QC/final-score/freeze-prefix 测试，只保留 WTI、mask、gather、多正例、HN、backbone 与 LayerNorm contract。

然后加入：

```python
@pytest.mark.parametrize(
    "name",
    [
        "probabilistic_text",
        "compose_final_retrieval_logits",
        "compute_query_conditioned_sap_logits",
        "resolve_loss_activations",
        "evidential_matrix_loss",
        "_select_uacl_gaussian_sample",
        "_uacl_intra_contrastive_loss",
        "_logvar_kl",
        "_evidential_similarity",
        "_evidential_nll_loss",
        "_evidential_neg_reg_loss",
    ],
)
def test_retired_auxiliary_model_api_is_absent(name):
    assert not hasattr(UATVR, name)


def test_wti_eva_capability_never_requires_visual_patch_hidden():
    spec = types.SimpleNamespace(
        supports_text_hidden=True,
        supports_visual_hidden=False,
    )
    assert UATVR._validate_eva_spec_capabilities(spec) == (True, False)


def test_wti_eva_capability_still_requires_text_hidden():
    spec = types.SimpleNamespace(
        supports_text_hidden=False,
        supports_visual_hidden=True,
    )
    with pytest.raises(ValueError, match="text hidden"):
        UATVR._validate_eva_spec_capabilities(spec)


def test_visual_hidden_is_absent_from_active_similarity_api():
    assert "visual_output_hidden" not in inspect.signature(
        UATVR.get_similarity_logits
    ).parameters
    assert "visual_output_hidden" not in inspect.signature(
        UATVR._loose_similarity
    ).parameters


def test_constructor_source_has_no_retired_auxiliary_fields():
    source = inspect.getsource(UATVR.__init__)
    for retired in (
        "self.sap",
        "self.qc_sap_",
        "self.pie_net_text",
        "self.uncertain_net_text",
        "self.ada_norm_text",
        "self.expansion_tokens",
        "self.spatial_enhancer",
    ):
        assert retired not in source


def test_hard_negative_activation_requires_positive_weight():
    assert UATVR._resolve_hard_negative_enabled(True, 0.0) is False
    assert UATVR._resolve_hard_negative_enabled(False, 0.1) is False
    assert UATVR._resolve_hard_negative_enabled(True, 0.1) is True
    with pytest.raises(ValueError, match="non-negative"):
        UATVR._resolve_hard_negative_enabled(True, -0.1)
```

把 `_make_forward_only_model` 的 `model.loss_activations` 删除，加入：

```python
    model.use_explicit_hard_negative_loss = False
    model.hard_negative_enabled = False
```

同时把 fake visual encoder 改为单 tensor 返回：

```python
    def get_visual_output(_self, video, video_mask, shaped, video_frame):
        batch = video_mask.size(0)
        return torch.zeros(batch, 1, 2)
```

其 fake similarity 返回改为：

```python
        return {
            "retrieve_logits": retrieval_logits,
            "video_group_id": group_ids,
            "hard_negative_loss": retrieval_logits.new_tensor(hard_negative_loss),
        }
```

并把 multi-positive forward 测试的精确键集合锁定为：

```python
    assert set(loss_dict) == {
        "total",
        "sim_loss",
        "hard_negative_loss",
        "unique_video_count",
        "duplicate_sample_count",
        "mean_positive_count",
    }
```

新增两个直接 score-source 测试：

```python
def _make_wti_unit_model():
    model = UATVR.__new__(UATVR)
    torch.nn.Module.__init__(model)
    model.task_config = Namespace(world_size=1, rank=0)
    model.clip = _FakeClip()
    model.text_weight_fc = torch.nn.Linear(2, 1, bias=False)
    model.video_weight_fc = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.text_weight_fc.weight)
    torch.nn.init.zeros_(model.video_weight_fc.weight)
    model.use_explicit_hard_negative_loss = False
    model.hard_negative_enabled = False
    model.w_hard_negative = 0.25
    model._diag_step = 0
    model._diag_interval = 1000
    model._diag_chain = {}
    model._hard_negative_chain = {}
    return model


def _wti_inputs():
    text = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    video = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    mask = torch.ones(2, 1, dtype=torch.long)
    groups = torch.tensor([10, 20])
    return text, video, mask, groups


def test_train_and_eval_use_identical_wti_score_source():
    model = _make_wti_unit_model()
    text, video, mask, groups = _wti_inputs()
    model.eval()
    eval_logits = model._wti_similarity(text, video, mask, mask)
    model.train()
    train_result = model._wti_similarity(
        text, video, mask, mask, video_group_id=groups
    )
    torch.testing.assert_close(train_result["retrieve_logits"], eval_logits)
    assert train_result["video_group_id"].tolist() == [10, 20]


def test_hard_negative_is_separate_and_weighted_once():
    model = _make_wti_unit_model()
    text, video, mask, groups = _wti_inputs()
    hard_video = video.flip(0)
    hard_valid = torch.tensor([True, True])
    model.eval()
    expected_retrieve = model._wti_similarity(text, video, mask, mask)
    model.train()
    model.use_explicit_hard_negative_loss = True
    model.hard_negative_enabled = True
    result = model._wti_similarity(
        text,
        video,
        mask,
        mask,
        video_group_id=groups,
        hard_visual_output=hard_video,
        hard_video_mask=mask,
        hard_valid=hard_valid,
    )
    expected_hard_logits = model.weighted_token_wise_intersection(
        torch.nn.functional.normalize(text, dim=-1),
        torch.nn.functional.normalize(hard_video, dim=-1),
        mask,
        mask,
    ) * model.clip.logit_scale.exp()
    expected_raw = model._hard_negative_infonce_loss(
        result["retrieve_logits"], expected_hard_logits, hard_valid
    )
    torch.testing.assert_close(result["retrieve_logits"], expected_retrieve)
    torch.testing.assert_close(
        result["hard_negative_loss"], model.w_hard_negative * expected_raw
    )
    assert result["video_group_id"].tolist() == [10, 20]
```

- [ ] **Step 2: 运行模型契约确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_modeling_mulit_losses.py \
  -k 'retired_auxiliary_model_api or wti_eva_capability or visual_hidden_is_absent or constructor_source or train_and_eval or hard_negative or bidirectional_multi_positive'
```

Expected: FAIL；旧 helper 仍存在，默认 EVA 路径仍依赖 SAP visual hidden，loss dict 仍有退役键。

- [ ] **Step 3: 删除旧 imports、构造字段和活动接线**

从 `modules/modeling_mulit.py` 删除：

```text
MILNCELoss_BoF
SemanticAnchorProbing
SpatialEnhancer
PIENet
l2_normalize
sample_gaussian_tensors
UncertaintyAdaNorm
UncertaintyModuleImage
UncertaintyModuleText
UncertaintyModuleTextMamba
math（只被 SpatialEnhancer patch reshape 使用）
```

从 `UATVR.__init__` 删除以下字段及日志：

```text
HYGIENE_FROZEN_PARAMETER_PREFIXES
pie_net_text, uncertain_net_text, use_ada_norm, ada_norm_text
n_video_samples, n_text_samples
spatial_enhancer
qc_sap_text_proj, qc_sap_anchor_proj
num_anchors, sap, num_expansion_tokens, expansion_tokens
loss_MIL_fct
w_mil, w_evidential, w_neg_reg, w_uncertainty_reg, w_orth
final_score_mode, lambda_prob, lambda_anchor, lambda_qc_sap, qc_sap_temperature
use_uacl_intra_alignment, w_uacl_intra, w_uacl_kl, uacl_temperature, uacl_sample_strategy
anneal_warmup_epochs, _current_epoch, uncertainty_mode, loss_activations
hygiene_wti_only, log_sigma_min, log_sigma_max
_prob_chain, _aux_chain
```

保留并明确初始化：

```python
    @staticmethod
    def _resolve_hard_negative_enabled(use_explicit, weight):
        weight = float(weight)
        if weight < 0:
            raise ValueError("w_hard_negative must be non-negative")
        return bool(use_explicit) and weight > 0

    # in __init__
        self.loss_fct = MultiPositiveCrossEn()
        self.use_explicit_hard_negative_loss = bool(
            getattr(self.task_config, "use_explicit_hard_negative_loss", False)
        )
        self.w_hard_negative = float(
            getattr(self.task_config, "w_hard_negative", 5e-2)
        )
        self.hard_negative_enabled = self._resolve_hard_negative_enabled(
            self.use_explicit_hard_negative_loss, self.w_hard_negative
        )
        self.experiment_profile = getattr(
            self.task_config, "experiment_profile", "default"
        )
        self._diag_step = 0
        self._diag_interval = getattr(task_config, "diag_interval", 10)
        self._diag_chain = {}
        self._hard_negative_chain = {}
```

删除 `freeze_inactive_parameters_for_profile()` 的构造调用；它与相应 helper 一起删除。

- [ ] **Step 4: 收敛 EVA 与视觉输出能力契约**

把 `_validate_eva_spec_capabilities` 替换为：

```python
    @staticmethod
    def _validate_eva_spec_capabilities(spec):
        if not getattr(spec, "supports_text_hidden", False):
            raise ValueError("EVA backbone must support text hidden tokens")
        return True, False
```

构造 EVA backbone 时改为：

```python
            require_text_hidden, require_visual_hidden = (
                self._validate_eva_spec_capabilities(spec)
            )
```

从 `get_visual_output` 签名删除 `require_hidden`，固定不请求 patch hidden，删除 SpatialEnhancer patch 处理块，并把返回协议收敛为单个 frame-token tensor：

```python
        encoded = self.clip.encode_image(
            video, return_hidden=False, video_frame=video_frame
        )
        visual_cls = encoded.float()
        visual_cls = visual_cls.view(bs_pair, -1, visual_cls.size(-1))
        return visual_cls
```

把旧 visual-encoder spy 测试改为断言返回单一 tensor，且 `_FakeClip.return_hidden_values == [False]`。

`get_sequence_visual_output` 相应返回三元组：

```python
        sequence_output, hidden_word = self.get_sequence_output(
            input_ids, token_type_ids, attention_mask, shaped=False
        )
        visual_output = self.get_visual_output(
            video, video_mask, shaped=True, video_frame=video_frame
        )
        return sequence_output, hidden_word, visual_output
```

- [ ] **Step 5: 把 `_wti_only_similarity` 替换为唯一 `_wti_similarity`**

使用以下完整实现；不要返回退役零值键：

```python
    def _wti_similarity(
        self,
        text_token,
        visual_output,
        attention_mask,
        video_mask,
        video_group_id=None,
        hard_visual_output=None,
        hard_video_mask=None,
        hard_valid=None,
    ):
        if self.training:
            visual_output = allgather_with_grad(
                visual_output.contiguous(), self.task_config
            )
            video_mask = allgather_no_grad(
                video_mask.contiguous(), self.task_config
            )
            text_token = allgather_with_grad(
                text_token.contiguous(), self.task_config
            )
            attention_mask = allgather_no_grad(
                attention_mask.contiguous(), self.task_config
            )
            if video_group_id is None:
                raise ValueError("WTI training requires video_group_id")
            video_group_id = allgather_no_grad(
                video_group_id.contiguous().view(-1), self.task_config
            )
            if hard_visual_output is not None and hard_video_mask is not None:
                hard_visual_output = allgather_with_grad(
                    hard_visual_output.contiguous(), self.task_config
                )
                hard_video_mask = allgather_no_grad(
                    hard_video_mask.contiguous(), self.task_config
                )
                if hard_valid is not None:
                    hard_valid = allgather_no_grad(
                        hard_valid.contiguous().view(-1), self.task_config
                    )

        text_token = F.normalize(text_token, dim=-1)
        visual_output = F.normalize(visual_output, dim=-1)
        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = self.weighted_token_wise_intersection(
            text_token, visual_output, attention_mask, video_mask
        ) * logit_scale

        if not self.training:
            return retrieve_logits

        hard_negative_loss = retrieve_logits.new_zeros(())
        hard_diag_mean = 0.0
        hard_pos_gap = 0.0
        if (
            self.hard_negative_enabled
            and hard_visual_output is not None
            and hard_video_mask is not None
        ):
            hard_visual_output = F.normalize(hard_visual_output, dim=-1)
            hard_logits = self.weighted_token_wise_intersection(
                text_token,
                hard_visual_output,
                attention_mask,
                hard_video_mask,
            ) * logit_scale
            if hard_valid is None:
                hard_valid = torch.ones(
                    hard_logits.size(1),
                    dtype=torch.bool,
                    device=hard_logits.device,
                )
            hard_negative_loss = self._hard_negative_infonce_loss(
                retrieve_logits, hard_logits, hard_valid
            )
            valid = hard_valid.to(device=hard_logits.device, dtype=torch.bool)
            if bool(valid.any().item()):
                hard_diag = hard_logits.diagonal()[valid]
                positive_diag = retrieve_logits.diagonal()[valid]
                hard_diag_mean = float(hard_diag.detach().mean().item())
                hard_pos_gap = float(
                    (positive_diag - hard_diag).detach().mean().item()
                )

        positive_mask = video_group_id[:, None].eq(video_group_id[None, :])
        self._diag_step += 1
        if self._diag_step % self._diag_interval == 0:
            stats = self._matrix_gap_stats(
                retrieve_logits, positive_mask=positive_mask
            )
            positive_values = retrieve_logits.detach()[positive_mask]
            self._diag_chain = {
                "pos_mean": stats["diag"],
                "neg_mean": stats["off"],
                "gap": stats["gap"],
                "pos_std": float(
                    positive_values.std(unbiased=False).item()
                ) if positive_values.numel() else 0.0,
            }
            self._hard_negative_chain = {
                "active": int(self.hard_negative_enabled),
                "loss": float(hard_negative_loss.detach().item()),
                "hard_diag_mean": hard_diag_mean,
                "hard_pos_gap": hard_pos_gap,
            }

        return {
            "retrieve_logits": retrieve_logits,
            "video_group_id": video_group_id,
            "hard_negative_loss": self.w_hard_negative * hard_negative_loss,
        }
```

保留 `_loose_similarity` 中 seqTransf 的现有确定性 token 处理，删除其后全部 SAP/概率分支，并把结尾替换为：

```python
        return self._wti_similarity(
            text_token,
            visual_output,
            attention_mask,
            video_mask,
            video_group_id=video_group_id,
            hard_visual_output=hard_visual_output,
            hard_video_mask=hard_video_mask,
            hard_valid=hard_valid,
        )
```

从 `_loose_similarity` 与 `get_similarity_logits` 删除 `visual_output_hidden` 形参，并同步完成以下原子调用方修改：

- `UATVR.forward`：`visual_cls = self.get_visual_output(...)`，hard video 同理；调用 `get_similarity_logits` 时不再传 hidden；
- `main_task_retrieval.py::_run_on_single_gpu`：`batch_visual_output_list` 直接存 tensor，删除 `_to_cpu_optional`、`_cat_optional_tensors` 和 `visual_output_all_full/full_chunk`，相似度调用只传 `cls_chunk`；
- `main_task_retrieval.py::eval_epoch`：multi-sentence 分支 append `visual_output.cpu()`；普通分支从 `get_sequence_visual_output` 解包三项；
- `main_task_retrieval.py::eval_epoch` 的 legacy multi-GPU cache 搬运改为 `devc_batch_list = [tensor.to(devc) for tensor in batch_visual_output_list]`，不得再按 tensor 的 batch 维迭代并构造 tuple；为 `n_gpu > 1 and "LOCAL_RANK" not in os.environ` 分支加入 direct-tensor-cache 测试；
- `scripts/build_msrvtt_model_mined_hard_negatives.py::encode_all_videos`：只累计/返回 `visual_outputs, video_masks`，评分调用不传 hidden；
- `scripts/diagnose_msrvtt_hard_negative_runtime.py::score_checkpoint`：只解包 visual output，评分调用不传 hidden；
- `scripts/diagnose_msrvtt_validation_errors.py::compute_validation_sim_matrix`：video cache 只含 visual output/mask，评分调用不传 hidden；
- 删除只测试 optional hidden cache 的 `tests/test_eval_optional_features.py`；在三个脚本测试中加入 fake model 返回单一 visual tensor 的最小评分测试，证明不会访问 `.cpu()`/`.detach()` 的 hidden 值。

`get_similarity_logits` 的 eval 返回仍为 `(retrieve_logits, {})`，避免改变现有 `logits, _ = ...` 调用语义；只删除 hidden 位置参数。

- [ ] **Step 6: 简化顶层 forward loss 字典**

hard video 编码条件统一使用 `self.hard_negative_enabled`。把旧 loss 累加和返回替换为：

```python
            sim_matrix = res["retrieve_logits"]
            global_group_ids = res["video_group_id"]
            sim_loss_t2v = self.loss_fct(
                sim_matrix, global_group_ids, global_group_ids
            )
            sim_loss_v2t = self.loss_fct(
                sim_matrix.T, global_group_ids, global_group_ids
            )
            sim_loss = (sim_loss_t2v + sim_loss_v2t) / 2
            hard_negative_loss = res["hard_negative_loss"]
            total_loss = sim_loss + hard_negative_loss

            positive_mask = global_group_ids[:, None].eq(
                global_group_ids[None, :]
            )
            positive_counts = positive_mask.sum(dim=1).float()
            unique_count = torch.unique(global_group_ids).numel()
            return {
                "total": total_loss,
                "sim_loss": sim_loss,
                "hard_negative_loss": hard_negative_loss,
                "unique_video_count": sim_matrix.new_tensor(float(unique_count)),
                "duplicate_sample_count": sim_matrix.new_tensor(
                    float(global_group_ids.numel() - unique_count)
                ),
                "mean_positive_count": positive_counts.mean(),
            }
```

- [ ] **Step 7: 删除退役 helpers 与对应测试**

删除以下 helper 及只验证它们的测试：

```text
final_score_uses_sap
final_score_uses_text_probability
compose_final_retrieval_logits
compute_query_conditioned_sap_logits
should_freeze_parameter_for_profile
freeze_inactive_parameters_for_profile
resolve_loss_activations
_positive_weight
_bound_ratio
_select_closest_gaussian_sample
_select_uacl_gaussian_sample
_uacl_intra_contrastive_loss
_logvar_kl
_evidential_similarity
_evidential_nll_loss
_evidential_neg_reg_loss
evidential_matrix_loss
probabilistic_text
_mean_pooling_for_similarity_sequence
_mean_pooling_for_similarity_visual
```

保留 `_matrix_gap_stats`、`_hard_negative_infonce_loss`、`weighted_token_wise_intersection` 及其完整 mask/shape/dtype/device 测试。

删除概率分支后同步移除 `_loose_similarity` 中不再使用的 `frame_num`、`word_num` 局部变量；保留 `sequence_output` 形参以维持现有 text encoder/评估 cache 返回协议，但函数体不再读取它。

- [ ] **Step 8: 运行模型、WTI、多正例和 hard-negative 测试**

Run:

```bash
set -euo pipefail
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_modeling_mulit_losses.py tests/test_prob_models_legacy.py
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
```

Expected: PASS。

Run:

```bash
set -euo pipefail
/home/xujie/miniconda3/envs/ret/bin/ruff check \
  modules/modeling_mulit.py main_task_retrieval.py \
  scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_modeling_mulit_losses.py tests/test_prob_models_legacy.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
git diff --check -- \
  modules/modeling_mulit.py main_task_retrieval.py \
  scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_modeling_mulit_losses.py tests/test_prob_models_legacy.py \
  tests/test_eval_optional_features.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
```

Expected: 退出码 0。

- [ ] **Step 9: 提交唯一 WTI 模型图**

```bash
git add -- \
  modules/modeling_mulit.py main_task_retrieval.py \
  scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_modeling_mulit_losses.py tests/test_prob_models_legacy.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
git add -u -- tests/test_eval_optional_features.py
git commit -m "refactor: converge active model on WTI retrieval" -- \
  modules/modeling_mulit.py main_task_retrieval.py \
  scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_modeling_mulit_losses.py tests/test_prob_models_legacy.py \
  tests/test_eval_optional_features.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
```

---

### Task 3: 清理训练日志、optimizer 分组与实验 manifest

**Files:**

- Modify: `main_task_retrieval.py:632-1805`
- Modify: `main_task_retrieval.py:1861-2054`
- Modify: `experiment_tracking.py:1-170`
- Modify: `tests/test_experiment_tracking.py:1-299`
- Modify: `tests/test_main_task_hard_negative_args.py:431-486`

**Interfaces:**

- Consumes: Task 2 的精简 loss dict 与 `_diag_chain` / `_hard_negative_chain`。
- Produces: 只记录 trusted protocol、WTI retrieval stats 和 hard-negative 配置的训练日志与 manifest；optimizer 删除只服务 uncertainty Mamba 的参数组，旧 optimizer state 不迁移。

- [ ] **Step 1: 写 manifest、日志函数消失和 optimizer 分组测试**

将 `tests/test_experiment_tracking.py::_args` 的旧字段删除，加入：

```python
        use_hard_negative_packing=False,
        use_explicit_hard_negative_loss=False,
        hard_negative_path="cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json",
        hard_negative_pack_seed=42,
        w_hard_negative=0.0,
```

扩展 manifest 测试：

```python
    assert "final_score_mode" not in payload
    assert "losses" not in payload
    assert payload["hard_negative"] == {
        "packing_enabled": False,
        "explicit_loss_enabled": False,
        "mapping_path": "cache_dir/hard_negatives/msrvtt_train_hardneg_clean.json",
        "pack_seed": 42,
        "loss_weight": 0.0,
    }
    encoded = json.dumps(payload)
    for retired in (
        "w_mil",
        "w_evidential",
        "w_neg_reg",
        "w_orth",
        "w_uacl_intra",
        "w_uacl_kl",
        "w_query_sim",
        "w_uncertainty_reg",
        "final_score_mode",
    ):
        assert retired not in encoded
```

在 `tests/test_main_task_hard_negative_args.py` 加入：

```python
@pytest.mark.parametrize(
    "name",
    [
        "_log_moe_weights_tsv",
        "_log_causal_summary_tsv",
        "_log_eval_stats_tsv",
    ],
)
def test_retired_auxiliary_loggers_are_absent(name):
    import main_task_retrieval

    assert not hasattr(main_task_retrieval, name)
```

同时锁定 MUS 仍存在且只消费最终相似度矩阵：

```python
def test_mus_logger_remains_available_for_wti_risk_diagnostics():
    import inspect
    import main_task_retrieval

    assert hasattr(main_task_retrieval, "_log_mus_scores_tsv")
    source = inspect.getsource(main_task_retrieval.eval_epoch)
    assert source.rindex("_log_mus_scores_tsv") > source.rindex(
        "sim_matrix = np.concatenate"
    )


def test_optimizer_source_has_no_uncertainty_mamba_group():
    import inspect
    import main_task_retrieval as retrieval

    source = inspect.getsource(retrieval.prep_optimizer)

    assert "mamba_keywords" not in source
    assert "mamba_lr_ratio" not in source
```

把 `tests/test_experiment_tracking.py` 中“缺 telemetry 的 legacy loss dict 被跳过”测试替换为：

```python
@pytest.mark.parametrize(
    "telemetry",
    [
        {},
        {"unique_video_count": torch.tensor(2.0)},
    ],
)
def test_train_epoch_rejects_missing_batch_protocol_telemetry(
    tmp_path, telemetry
):
    with pytest.raises(ValueError, match="missing required keys"):
        _run_train_epoch_for_sidecar(tmp_path, telemetry)
```

`tests/test_main_task_hard_negative_args.py::CapturingModel.forward` 和 `_run_train_epoch_for_sidecar` 的 fake loss dict 都补齐：

```python
{
    "unique_video_count": loss.new_tensor(2.0),
    "duplicate_sample_count": loss.new_tensor(0.0),
    "mean_positive_count": loss.new_tensor(1.0),
}
```

同时从 fake args 删除 `gate_log_interval` 与 `log_moe_weights`。

- [ ] **Step 2: 运行测试确认 RED**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_experiment_tracking.py \
  tests/test_main_task_hard_negative_args.py \
  -k 'manifest or auxiliary_loggers or mus_logger or optimizer_source or telemetry'
```

Expected: FAIL；manifest 仍含旧字段、旧 logger 仍存在、optimizer 仍含 Mamba 分组，legacy telemetry 仍被跳过。

- [ ] **Step 3: 收敛 experiment manifest**

从 `experiment_tracking.py` 删除 `_LOSS_FIELDS`，把 `build_experiment_manifest` 的旧 `losses/final_score_mode` 替换为：

```python
    hard_negative = {
        "packing_enabled": bool(
            getattr(args, "use_hard_negative_packing", False)
        ),
        "explicit_loss_enabled": bool(
            getattr(args, "use_explicit_hard_negative_loss", False)
        ),
        "mapping_path": getattr(args, "hard_negative_path", ""),
        "pack_seed": getattr(args, "hard_negative_pack_seed", None),
        "loss_weight": getattr(args, "w_hard_negative", 0.0),
    }
```

返回 payload 只包含：`protocol_version`、`git`、`split`、`seed`、`profile`、`backbone`、`data`、`batch`、`hard_negative`。

- [ ] **Step 4: 简化 optimizer 参数组**

删除 `mamba_keywords`、Mamba 四组中间变量和 `args.mamba_lr_ratio`。`prep_optimizer` 的参数组固定为：

```python
    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in decay_clip_param_tp],
            "weight_decay": weight_decay,
            "lr": args.lr * coef_lr,
        },
        {
            "params": [p for _, p in no_decay_clip_param_tp],
            "weight_decay": 0.0,
            "lr": args.lr * coef_lr,
        },
        {
            "params": [p for _, p in decay_other_param_tp],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for _, p in no_decay_other_param_tp],
            "weight_decay": 0.0,
        },
    ]
```

训练日志中的 `lr_new` 兼容测试 fake optimizer：

```python
lr_new = (
    optimizer.param_groups[2]["lr"]
    if len(optimizer.param_groups) > 2
    else lr_clip
)
```

不增加 optimizer schema 或迁移逻辑；旧 optimizer 参数组若不兼容，由 `optimizer.load_state_dict` 的原生明确错误终止，不能静默重排。

- [ ] **Step 5: 删除旧因果链/MoE 日志并保留 WTI/MUS 诊断**

在 `train_epoch`：

- 删除 `gate_log_step`、`core._current_epoch`、`prob/aux` 累积、MoE 调用和所有 Prob/Aux/Hygiene/Score/UACL 文本。
- 保留 batch protocol sidecar、通用 scalar loss details、`_diag_chain` 与 `_hard_negative_chain`。
- optimizer step 日志可输出以下两行，字段不存在时不伪造旧值：

```python
                if retrieval_stats:
                    logger.info(
                        "  [WTI] pos=%.3f neg=%.3f gap=%.3f pos_std=%.3f",
                        retrieval_stats["pos_mean"],
                        retrieval_stats["neg_mean"],
                        retrieval_stats["gap"],
                        retrieval_stats["pos_std"],
                    )
                if hard_negative_stats:
                    logger.info(
                        "  [HardNeg] active=%d loss=%.4f hard_diag=%.3f pos_gap=%.3f",
                        hard_negative_stats["active"],
                        hard_negative_stats["loss"],
                        hard_negative_stats["hard_diag_mean"],
                        hard_negative_stats["hard_pos_gap"],
                    )
```

完全删除函数及其调用：

```text
_log_moe_weights_tsv
_log_causal_summary_tsv
_log_eval_stats_tsv
```

保留 `_log_mus_scores_tsv` 及 eval 完整 `sim_matrix` 上的调用。完全删除 causal TSV 是明确选择：WTI/HN 的训练期统计由上述 console telemetry 提供，离线 HN 行为由 `diagnose_msrvtt_hard_negative_runtime.py` 提供，MUS TSV 继续承担纯 WTI 风险诊断，不保留混合概率列。

`train_epoch` 不再接受旧 test double 的缺 telemetry 兼容：全局 rank 直接对模型返回调用 `extract_batch_protocol_stats(loss_dict)`；缺字段抛 `ValueError`，同步更新 test double 始终提供三项 telemetry。

- [ ] **Step 6: 运行 runtime/tracking 测试确认 GREEN**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_experiment_tracking.py \
  tests/test_main_task_hard_negative_args.py
```

Expected: PASS。

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/ruff check \
  main_task_retrieval.py experiment_tracking.py \
  tests/test_experiment_tracking.py tests/test_main_task_hard_negative_args.py
```

Expected: 退出码 0。

- [ ] **Step 7: 提交 runtime 与 provenance 清理**

```bash
git add -- \
  main_task_retrieval.py experiment_tracking.py \
  tests/test_experiment_tracking.py tests/test_main_task_hard_negative_args.py
git commit -m "refactor: remove retired retrieval telemetry" -- \
  main_task_retrieval.py experiment_tracking.py \
  tests/test_experiment_tracking.py tests/test_main_task_hard_negative_args.py
```

---

### Task 4: 删除旧 CLI 并同步全部受支持脚本

**Files:**

- Modify: `main_task_retrieval.py:36-590`
- Modify: `main_task_retrieval.py:632-704`
- Modify: `train_msrvtt.sh`
- Modify: `eval.sh`
- Modify: `run_train_msrvtt_bg.sh`
- Modify: `train_msvd.sh`
- Modify: `scripts/build_msrvtt_model_mined_hard_negatives.py`
- Modify: `scripts/diagnose_msrvtt_hard_negative_runtime.py`
- Modify: `scripts/diagnose_msrvtt_validation_errors.py`
- Modify: `tests/test_main_task_hard_negative_args.py`
- Modify: `tests/test_trusted_eval_protocol.py`
- Modify: `tests/test_build_msrvtt_model_mined_hard_negatives.py`
- Modify: `tests/test_diagnose_hard_negative_runtime.py`
- Modify: `tests/test_diagnose_msrvtt_validation_errors.py`

**Interfaces:**

- Consumes: 只含 deterministic retrieval、backbone、trusted protocol、attributes 和独立 HN 的命令行。
- Produces: 旧参数统一 `SystemExit(2)`；train/eval scripts 默认 `experiment_profile=hygiene` 且不发出退役参数。

- [ ] **Step 1: 写 exhaustive 旧 CLI 拒绝测试**

在 `tests/test_main_task_hard_negative_args.py` 定义：

```python
RETIRED_CLI_CASES = (
    ("--gate_log_interval", "10"),
    ("--gate_log_dir", "/tmp/gates"),
    ("--log_moe_weights", None),
    ("--moe_log_dir", "/tmp/moe"),
    ("--use_mil", None),
    ("--sampled_use_mil", None),
    ("--n_video_embeddings", "7"),
    ("--n_text_embeddings", "7"),
    ("--mamba_lr_ratio", "0.1"),
    ("--uncertainty_text_head", "text"),
    ("--log_sigma_min", "-1.5"),
    ("--log_sigma_max", "4"),
    ("--rope_mode", "2d"),
    ("--disable_spatial_enhancer", None),
    ("--num_expansion_tokens", "4"),
    ("--use_ada_norm", None),
    ("--eval_branch_mode", "base_only"),
    ("--disable_query_gate_in_retrieval", None),
    ("--fusion_mode", "prob_mos"),
    ("--w_mil", "0"),
    ("--w_evidential", "0"),
    ("--w_neg_reg", "0"),
    ("--final_score_mode", "wti"),
    ("--lambda_prob", "0"),
    ("--lambda_anchor", "0"),
    ("--lambda_qc_sap", "0"),
    ("--qc_sap_temperature", "0.1"),
    ("--w_uncertainty_reg", "0"),
    ("--w_orth", "0"),
    ("--w_query_sim", "0"),
    ("--use_uacl_intra_alignment", None),
    ("--w_uacl_intra", "0"),
    ("--w_uacl_kl", "0"),
    ("--uacl_temperature", "0.07"),
    ("--uacl_sample_strategy", "closest"),
    ("--anneal_warmup_epochs", "0"),
    ("--warmup_steps", "500"),
    ("--uncertainty_mode", "none"),
    ("--fusion_temperature", "1.5"),
)


@pytest.mark.parametrize(("flag", "value"), RETIRED_CLI_CASES)
def test_get_args_rejects_retired_cli(flag, value, monkeypatch, capsys):
    argv = [
        "prog",
        "--do_train",
        "--output_dir",
        "/tmp/uatvr-test-out",
        "--expand_msrvtt_sentences",
        flag,
    ]
    if value is not None:
        argv.append(value)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        get_args()
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr
    assert flag in stderr
```

把旧 HN+UACL 解析测试改为只验证：`use_explicit_hard_negative_loss=True`、`w_hard_negative=0.07`。删除旧 evidential/final-score 接受测试和非 MSRVTT 静默归零测试。

- [ ] **Step 2: 写 trusted profile 与脚本参数测试**

`tests/test_trusted_eval_protocol.py::_args` 只保留：datatype、train/eval、split、init model、expanded captions、experiment profile、两个 HN flags。测试改为：

```python
@pytest.mark.parametrize(
    "flag",
    ["use_hard_negative_packing", "use_explicit_hard_negative_loss"],
)
def test_hygiene_rejects_hard_negative_diagnostics(flag):
    with pytest.raises(ValueError, match="hard-negative diagnostic"):
        retrieval.validate_trusted_cli(_args(**{flag: True}))


def test_default_profile_allows_explicit_hard_negative_diagnostic():
    retrieval.validate_trusted_cli(
        _args(
            experiment_profile="default",
            use_explicit_hard_negative_loss=True,
        )
    )
```

先把 `_run_with_fake_torchrun` 的 fake setup 改为：

```python
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    python3_path = fake_bin / "python3"
    python3_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    python3_path.chmod(0o755)
```

这避免脚本测试真的生成 trusted split 文件。随后扩展为参数化 fake torchrun 测试：

```python
@pytest.mark.parametrize(
    "script_name", ["train_msrvtt.sh", "eval.sh", "train_msvd.sh"]
)
def test_supported_scripts_emit_only_supported_retrieval_args(
    script_name, tmp_path
):
    retired_flags = {flag for flag, _ in RETIRED_CLI_CASES}
    result, capture_path = _run_with_fake_torchrun(
        script_name, tmp_path, "0"
    )
    assert result.returncode == 0, result.stderr
    captured = capture_path.read_text(encoding="utf-8").splitlines()
    assert "--experiment_profile" in captured
    profile_index = captured.index("--experiment_profile")
    assert captured[profile_index + 1] == "hygiene"
    emitted_flags = {
        token.split("=", 1)[0]
        for token in captured
        if token.startswith("--")
    }
    assert retired_flags.isdisjoint(emitted_flags)
    if script_name == "eval.sh":
        assert "--log_mus_scores" in emitted_flags
    else:
        assert "--log_mus_scores" not in emitted_flags
```

- [ ] **Step 3: 运行 CLI/script 契约确认 RED**

Run:

```bash
set -euo pipefail
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  -k 'retired_cli or hard_negative_diagnostic or supported_retrieval_args'
```

Expected: FAIL；旧 CLI 仍被接受，脚本仍发送旧参数。

- [ ] **Step 4: 删除 parser 旧参数并简化 trusted 校验**

从 `get_args` 删除 `RETIRED_CLI_CASES` 中所有参数定义。把 `w_hard_negative` 移到 hard-negative 参数组，保留 `experiment_profile`、backbone、extra CLS、attributes 和 trusted 路径。

把 `validate_trusted_cli` 的 hygiene 部分替换为：

```python
    if args.experiment_profile == "hygiene" and (
        args.use_hard_negative_packing
        or args.use_explicit_hard_negative_loss
    ):
        raise ValueError("hygiene forbids hard-negative diagnostic paths")
```

删除非 MSRVTT hygiene 的旧参数静默归零块。`set_seed_logger` 的 key groups 只保留：Training、Model、HardNeg、Protocol；Protocol 包含 `experiment_profile`，不打印已删除属性。

- [ ] **Step 5: 将训练/评估 shell 收敛为 WTI-only**

`train_msrvtt.sh`：

- 默认 `EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE:-hygiene}`。
- 删除 `EXTRA_PROFILE_ARGS`。
- 删除所有采样、概率、score fusion、query、UACL、annealing、MoE、AdaNorm 参数。
- 删除 RoPE/SpatialEnhancer 参数；保留 trusted split build、`--eval_split val`、OpenAI/EVA backbone、extra CLS、`--experiment_profile` 和 `"$@"`。

`eval.sh`：

- 删除 query-only/NIG 示例、`EVAL_BRANCH_MODE`、fusion/AdaNorm/uncertainty/score variables。
- 默认 `EXPERIMENT_PROFILE=hygiene`。
- 删除 branch-dependent 命名，固定为：

```bash
OUTPUT_DIR=${OUTPUT_DIR:-ckpts/eval_${DATATYPE}_${RUN_ID}}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}_${DATATYPE}_ua${USE_ATTRIBUTES}.log}
```

- 删除/改写仍引用 branch/fusion/uncertainty/final-score 变量的 echo；保留 `--log_mus_scores`，删除采样、probability、final-score 参数，保留显式 `EVAL_SPLIT`、trusted-v1 split build、backbone、attributes 和 profile。

`run_train_msrvtt_bg.sh`：把旧参数示例改为：

```bash
# 透传受支持参数给 train_msrvtt.sh；退役参数由 argparse 明确拒绝。
```

`train_msvd.sh`：删除采样、uncertainty、query/gate/RoPE 参数，显式传 `--experiment_profile hygiene`，保留 deterministic WTI/seqTransf 配置。

- [ ] **Step 6: 更新三个模型诊断/挖掘脚本的 checkpoint 与 trusted-val 边界**

`scripts/diagnose_msrvtt_hard_negative_runtime.py` 的 checkpoint 参数改为 required，不再默认指向旧 checkpoint：

```python
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--target_checkpoint", required=True)
```

`build_task_args` 只构造 main parser 仍支持的确定性参数：do_eval、output/init、trusted data paths、batch、frames/words、datatype、expanded captions、backbone、extra CLS、profile。删除所有 RETIRED_CLI_CASES 参数。新增测试断言生成的 Namespace 不具有 `final_score_mode`、`uncertainty_mode`、`n_video_embeddings`、`w_evidential`。

三个脚本把现有 `train_csv`/`val_csv` 定义替换为下列 generated trusted defaults，并新增/透传其余协议参数（路径相对 `PROJECT_ROOT` 解析；不得重复注册同名 argparse flag）：

```python
parser.add_argument(
    "--train_csv",
    default=str(PROJECT_ROOT / "data/generated/msrvtt_trusted_v1/train.csv"),
)
parser.add_argument(
    "--source_train_csv",
    default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_train.9k.csv",
)
parser.add_argument(
    "--test_csv",
    default=f"{DEFAULT_DATA_ROOT}/csv/MSRVTT_JSFUSION_test.csv",
)
parser.add_argument(
    "--split_manifest",
    default=str(
        PROJECT_ROOT
        / "dataloaders/splits/msrvtt_trusted_v1_seed42.json"
    ),
)
parser.add_argument(
    "--val_csv",
    default=str(PROJECT_ROOT / "data/generated/msrvtt_trusted_v1/val.csv"),
)
```

`test_csv` 只用于 `validate_trusted_manifest`，不得传给评分 dataloader。

在 `scripts/diagnose_msrvtt_hard_negative_runtime.py` 增加共享 gate，另外两个脚本直接导入调用：

```python
def validate_trusted_diagnostic_inputs(args, scored_csv=None):
    from dataloaders.msrvtt_protocol import (
        load_trusted_manifest,
        validate_trusted_manifest,
    )

    manifest = load_trusted_manifest(args.split_manifest)
    validate_trusted_manifest(
        manifest,
        args.source_train_csv,
        args.data_path,
        args.test_csv,
    )
    with open(args.train_csv, "r", encoding="utf-8", newline="") as handle:
        train_ids = {
            str(row.get("video_id", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("video_id", "")).strip()
        }
    expected_train_ids = set(manifest["train_video_ids"])
    if train_ids != expected_train_ids:
        raise ValueError(
            "diagnostic train CSV must exactly match trusted-v1 train video IDs; "
            f"expected={len(expected_train_ids)}, got={len(train_ids)}"
        )
    if scored_csv is not None:
        with open(scored_csv, "r", encoding="utf-8", newline="") as handle:
            scored_ids = {
                str(row.get("video_id", "")).strip()
                for row in csv.DictReader(handle)
                if str(row.get("video_id", "")).strip()
            }
        expected_ids = set(manifest["val_video_ids"])
        if scored_ids != expected_ids:
            raise ValueError(
                "diagnostic CSV must exactly match trusted-v1 internal val "
                f"video IDs; expected={len(expected_ids)}, got={len(scored_ids)}"
            )
    return manifest
```

- runtime HN diagnostic 与 model-mined builder 的 `main()` 在加载 checkpoint 前调用 `validate_trusted_diagnostic_inputs(args)`；
- validation-error diagnostic 在读 CSV 或构造 dataloader 前调用 `validate_trusted_diagnostic_inputs(args, scored_csv=args.val_csv)`；
- 不把协议参数仅仅传给 `build_task_args` 后就假设校验已经发生；三个脚本都不调用 `main_task_retrieval.main()`。

在 `tests/test_diagnose_hard_negative_runtime.py` 加入：

```python
import argparse
from pathlib import Path

import pytest


def test_trusted_diagnostic_gate_rejects_non_internal_val_csv(
    tmp_path, monkeypatch
):
    from dataloaders import msrvtt_protocol
    from scripts.diagnose_msrvtt_hard_negative_runtime import (
        validate_trusted_diagnostic_inputs,
    )

    monkeypatch.setattr(
        msrvtt_protocol,
        "load_trusted_manifest",
        lambda _path: {
            "train_video_ids": ["train1"],
            "val_video_ids": ["video1"],
        },
    )
    monkeypatch.setattr(
        msrvtt_protocol,
        "validate_trusted_manifest",
        lambda *_args, **_kwargs: None,
    )
    args = argparse.Namespace(
        split_manifest="manifest.json",
        train_csv=str(tmp_path / "train.csv"),
        source_train_csv="source.csv",
        data_path="annotation.json",
        test_csv="test.csv",
    )
    Path(args.train_csv).write_text(
        "video_id\ntrain1\n", encoding="utf-8"
    )
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("video_id,sentence\nvideo2,caption\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="exactly match trusted-v1 internal val"
    ):
        validate_trusted_diagnostic_inputs(args, scored_csv=wrong)

    valid = tmp_path / "valid.csv"
    valid.write_text("video_id,sentence\nvideo1,caption\n", encoding="utf-8")
    validate_trusted_diagnostic_inputs(args, scored_csv=valid)

    Path(args.train_csv).write_text(
        "video_id\ntrain1\nvideo1\n", encoding="utf-8"
    )
    with pytest.raises(
        ValueError, match="exactly match trusted-v1 train"
    ):
        validate_trusted_diagnostic_inputs(args, scored_csv=valid)
```

三个脚本实际构造 train dataset 时必须使用同一个 `args.train_csv`，默认均为 generated trusted 8500-train；`source_train_csv` 只用于重建/校验 manifest，不能传给训练集 dataloader。

三个脚本的 main wiring 测试分别 monkeypatch gate 为 recorder，并断言任何 checkpoint/model loader 调用前 recorder 已执行。

同一任务还必须更新：

- `scripts/build_msrvtt_model_mined_hard_negatives.py`：删除 `DEFAULT_CKPT`，将 `--checkpoint` 改为 `required=True`；默认 `val_csv` 指向 `data/generated/msrvtt_trusted_v1/val.csv`，不得默认 JSFusion；
- `scripts/diagnose_msrvtt_validation_errors.py`：删除两个默认 checkpoint 常量，将 `--baseline_checkpoint`、`--target_checkpoint` 改为 required；文案与默认 `val_csv` 改为 trusted-v1 internal val，不再称 JSFusion 为 validation；
- 三个脚本的 task args 均传 `--source_train_csv`、`--test_csv`、`--split_manifest` 与 `--eval_split val`，其中 `test_csv` 只供 trusted manifest 校验，脚本不得构造或评分 test dataloader；
- 对应测试断言缺 checkpoint 时 argparse 返回 2、默认/生成的 `val_csv` 不含 `JSFUSION_test`，并检查 Namespace 无退役参数。

- [ ] **Step 7: 运行 parser、shell 与 trusted 协议测试确认 GREEN**

Run:

```bash
set -euo pipefail
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
bash -n train_msrvtt.sh eval.sh run_train_msrvtt_bg.sh train_msvd.sh
```

Expected: 全部 PASS/退出 0。

- [ ] **Step 8: 静态检查并提交 CLI/script 收敛**

Run:

```bash
set -euo pipefail
/home/xujie/miniconda3/envs/ret/bin/ruff check \
  main_task_retrieval.py scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
git diff --check -- \
  main_task_retrieval.py train_msrvtt.sh eval.sh run_train_msrvtt_bg.sh \
  train_msvd.sh scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
```

Commit:

```bash
git add -- \
  main_task_retrieval.py train_msrvtt.sh eval.sh run_train_msrvtt_bg.sh \
  train_msvd.sh scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
git commit -m "refactor: remove retired retrieval interfaces" -- \
  main_task_retrieval.py train_msrvtt.sh eval.sh run_train_msrvtt_bg.sh \
  train_msvd.sh scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
```

---

### Task 5: 删除孤立模块、旧 sweep 与 repeat 产物

**Files:**

- Delete: `query_models/module_sap.py`
- Delete: `hyperparam_search.py`
- Delete: `scripts/wait_gpu_start_clean_hn.sh`
- Delete: `search_results/`
- Preserve: `prob_models/`
- Preserve: `modules/modeling.py`
- Preserve: `modules/spatial_enhancer.py`
- Preserve: `modules/mus_util.py`

**Interfaces:**

- Consumes: Tasks 2–4 已经移除的 imports/CLI/日志消费者。
- Produces: 当前工作树不再包含可执行的旧 SAP/uncertainty sweep 或 HN repeat launcher；Git 历史承担复现。

- [ ] **Step 1: 运行孤立引用扫描**

Run:

```bash
rg -n 'query_models\.module_sap|hyperparam_search|wait_gpu_start_clean_hn|search_results/' \
  --glob '!research_refs/**' \
  --glob '!docs/superpowers/**' \
  --glob '!query_models/module_sap.py' \
  --glob '!hyperparam_search.py' \
  --glob '!scripts/wait_gpu_start_clean_hn.sh' \
  --glob '!search_results/**' \
  .
```

Expected: 无活动消费者；若仍有命中，先回到对应任务清理，不能删除后留下断链。

- [ ] **Step 2: 删除三个孤立源文件与 tracked 结果目录**

使用 `apply_patch` 删除：

```text
query_models/module_sap.py
hyperparam_search.py
scripts/wait_gpu_start_clean_hn.sh
```

使用精确机械删除：

```bash
git rm -r -- search_results
```

不得删除 `prob_models/`、历史 `modules/modeling.py`、`modules/spatial_enhancer.py` 源文件或 hard-negative 只读诊断脚本。

- [ ] **Step 3: 运行活动树引用与 import smoke tests**

Run:

```bash
set -euo pipefail
test ! -e query_models/module_sap.py
test ! -e hyperparam_search.py
test ! -e scripts/wait_gpu_start_clean_hn.sh
test ! -e search_results
/home/xujie/miniconda3/envs/ret/bin/python -c 'from modules.modeling_mulit import UATVR; from main_task_retrieval import get_args; print(UATVR.__name__, get_args.__name__)'
```

Expected: 删除检查退出 0，import 输出 `UATVR get_args`。

Run:

```bash
rg -n 'query_models\.module_sap|hyperparam_search|wait_gpu_start_clean_hn|search_results/' \
  --glob '!research_refs/**' --glob '!docs/superpowers/**' .
```

Expected: 无输出，`rg` 退出 1。

- [ ] **Step 4: 运行限定测试并提交归档删除**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_modeling_mulit_losses.py \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_experiment_tracking.py \
  tests/test_diagnose_hard_negative_runtime.py
```

Expected: PASS。

Commit:

```bash
git add -u -- \
  query_models/module_sap.py hyperparam_search.py \
  scripts/wait_gpu_start_clean_hn.sh search_results
git commit -m "chore: remove retired retrieval artifacts" -- \
  query_models/module_sap.py hyperparam_search.py \
  scripts/wait_gpu_start_clean_hn.sh search_results
```

---

### Task 6: 全量验证、完成墓碑并归档设计资料

**Files:**

- Modify: `AGENTS.md`
- Modify: `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`
- Delete after all checks pass: `docs/superpowers/specs/2026-07-11-sap-retirement-and-research-pivot-design.md`
- Delete after all checks pass: `docs/superpowers/plans/2026-07-11-research-synthesis-and-ssot-plan.md`
- Delete after all checks pass: `docs/superpowers/plans/2026-07-11-sap-retirement-code-cleanup-plan.md`

**Interfaces:**

- Consumes: Plan A 与 Tasks 1–5 全部绿色的工作树。
- Produces: Roadmap 的最终单条墓碑、完整验证证据，以及只通过 Git 历史追溯的已执行规格/计划。

- [ ] **Step 1: 运行完整测试**

Run:

```bash
/home/xujie/miniconda3/envs/ret/bin/pytest -q tests
```

Expected: 全部 PASS，0 failed。不得用历史测试结果替代本次新输出。

- [ ] **Step 2: 运行静态与 shell 检查**

Run:

```bash
set -euo pipefail
/home/xujie/miniconda3/envs/ret/bin/ruff check \
  modules/modeling_mulit.py \
  main_task_retrieval.py \
  experiment_tracking.py \
  scripts/build_msrvtt_model_mined_hard_negatives.py \
  scripts/diagnose_msrvtt_hard_negative_runtime.py \
  scripts/diagnose_msrvtt_validation_errors.py \
  tests/test_modeling_mulit_losses.py \
  tests/test_prob_models_legacy.py \
  tests/test_main_task_hard_negative_args.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_experiment_tracking.py \
  tests/test_build_msrvtt_model_mined_hard_negatives.py \
  tests/test_diagnose_hard_negative_runtime.py \
  tests/test_diagnose_msrvtt_validation_errors.py
bash -n train_msrvtt.sh eval.sh run_train_msrvtt_bg.sh train_msvd.sh
```

Expected: 全部退出 0。

- [ ] **Step 3: 运行退役引用扫描并核对允许例外**

Run:

```bash
set -euo pipefail
retired='SemanticAnchorProbing|query_models\.module_sap|probabilistic_text|qc_sap|pie_net_text|uncertain_net_text|ada_norm_text|expansion_tokens|wti_prob_mu|wti_anchor_wti|wti_qc_sap|prob_mu_logits|anchor_wti_logits|MIL_loss|evidential_loss|neg_reg_loss|orth_loss|uacl_|final_score_mode|lambda_(prob|anchor|qc_sap)|n_(video|text)_embeddings|uncertainty_text_head|log_sigma_(min|max)|fusion_mode|fusion_temperature|use_ada_norm|w_(mil|evidential|neg_reg|uncertainty_reg|orth|query_sim)|gate_log_interval|gate_log_dir|log_moe_weights|moe_log_dir|use_mil|sampled_use_mil|mamba_lr_ratio|rope_mode|disable_spatial_enhancer|eval_branch_mode|disable_query_gate_in_retrieval|anneal_warmup_epochs|warmup_steps|uncertainty_mode'
if rg -n "$retired" \
  main_task_retrieval.py experiment_tracking.py \
  train_msrvtt.sh eval.sh run_train_msrvtt_bg.sh train_msvd.sh \
  scripts --glob '!research_refs/**'; then
  exit 1
fi
```

Expected: 活动 entry point 与 scripts 无输出，整个保护块退出 0；任何命中立即退出 1。

Run:

```bash
set -euo pipefail
retired='SemanticAnchorProbing|query_models\.module_sap|probabilistic_text|qc_sap|pie_net_text|uncertain_net_text|ada_norm_text|expansion_tokens|wti_prob_mu|wti_anchor_wti|wti_qc_sap|prob_mu_logits|anchor_wti_logits|MIL_loss|evidential_loss|neg_reg_loss|orth_loss|uacl_|final_score_mode|lambda_(prob|anchor|qc_sap)|n_(video|text)_embeddings|uncertainty_text_head|log_sigma_(min|max)|fusion_mode|fusion_temperature|use_ada_norm|w_(mil|evidential|neg_reg|uncertainty_reg|orth|query_sim)|gate_log_interval|gate_log_dir|log_moe_weights|moe_log_dir|use_mil|sampled_use_mil|mamba_lr_ratio|rope_mode|disable_spatial_enhancer|eval_branch_mode|disable_query_gate_in_retrieval|anneal_warmup_epochs|warmup_steps|uncertainty_mode'
rg -n "$retired|\bsap\." modules/modeling_mulit.py
rg -n "$retired|\bsap\." tests
if rg -n 'from prob_models|query_models\.module_sap' modules/modeling_mulit.py; then
  exit 1
fi
if rg -n 'SpatialEnhancer|rope_mode|disable_spatial_enhancer' \
  modules/modeling_mulit.py main_task_retrieval.py \
  train_msrvtt.sh eval.sh train_msvd.sh; then
  exit 1
fi
test -f modules/spatial_enhancer.py
```

Expected:

- 模型命中只在 `RETIRED_CHECKPOINT_PREFIXES` / root-key 拒绝器；
- 测试命中只在 `tests/test_modeling_mulit_losses.py`、`tests/test_main_task_hard_negative_args.py`、`tests/test_experiment_tracking.py` 以及三个模型挖掘/诊断测试中明确的 checkpoint/CLI/manifest/API “拒绝或不存在”负契约；
- `from prob_models` / `module_sap` 扫描无输出；
- 活动模型、CLI 与 shell 的 SpatialEnhancer/RoPE 扫描无输出；`modules/spatial_enhancer.py` 文件仍存在，只通过 Git/独立后续规格处理。

`modules/modeling.py` 与 `prob_models/` 明确不在活动实现扫描范围；不能因为负契约测试需要退役字符串而要求 `tests/` 绝对零命中。

- [ ] **Step 4: 写入最终 `AGENTS.md` 事实并完成 Roadmap 墓碑**

代码删除完成后，才用 `apply_patch` 重写 `AGENTS.md`。保留原有标题体系，并确保包含：

```markdown
# UATVR Agent 入口

UATVR 是基于 PyTorch、OpenAI CLIP ViT-B/16、WTI 与 trusted-v1 协议的文本—视频检索研究项目。交流和项目文档统一使用简体中文。
```

必须逐条写入：

- Roadmap 是科研唯一 SSOT；论文综合是外部证据分析、非 SSOT；Query 文档是历史快照；
- MSRVTT 固定 seed 42、8500 train / 500 internal val，JSFusion 1K 只作冻结后显式盲测；
- 主损失为按精确 `video_id` 的双向多正例 InfoNCE，最终 logits 固定 WTI；
- P0 为 OpenAI CLIP hygiene WTI-only，global forward batch 256、4 卡 micro 64、accum 1；
- native FP16 LayerNorm、FP32 master affine 与 `CLIP_LAYER_NORM_PRECISION=fp32` 回退语义；
- Hard negative 主线停止但独立诊断保留；UACL 活动接口已删除；
- 主入口、batch 语义、用户手动训练、不得代跑长期训练、`EVAL_SPLIT=val|test`、训练期不得构造 test dataloader；
- 测试/ruff 精确入口及禁止根目录无范围 pytest；`research_refs/` 不提交。

代码入口表只列训练、评估、主模型、数据协议和测试；不再列退役模块或历史概率目录。文档入口只列 Roadmap、论文综合、Query 历史快照和 Qwen 说明。

用 `apply_patch` 把唯一墓碑行改为：

```markdown
| SAP 及其依赖链 | 已删除、不得恢复 | 历史结构、实验和止损证据仅从 Git 历史追溯 |
```

并把 Plan A 的 UACL 过渡状态从“活动接口待 Plan B 删除”改为“活动接口已删除，不恢复”；不得改回单独实验入口。

Run:

```bash
set -euo pipefail
test "$(rg -n 'SAP|SemanticAnchorProbing|AnchorWTI|QC-SAP|wti_prob_mu|wti_anchor_wti|wti_qc_sap' \
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md | wc -l)" -eq 1
test "$(rg -n 'SAP|SemanticAnchorProbing|module_sap|prob_models/' AGENTS.md | wc -l)" -eq 0
for file in AGENTS.md docs/README.md; do
  rg -Fq 'multimodal_retrieval_research_synthesis' "$file" || exit 1
done
for key in \
  trusted-v1 '8500 train / 500 internal val' '双向多正例 InfoNCE' \
  'global forward batch 256' '每卡 micro-batch 64' 'accum=1' \
  'CLIP_LAYER_NORM_PRECISION=fp32' 'EVAL_SPLIT=val|test' research_refs/
do
  rg -Fq "$key" AGENTS.md || exit 1
done
```

Expected: Roadmap 唯一 SAP 命中为最终墓碑；AGENTS 无 SAP/旧模块入口，并保留所有执行边界。

- [ ] **Step 5: 审计工作树与用户既有改动**

Run:

```bash
set -euo pipefail
git diff --check -- AGENTS.md docs/project/RESEARCH_ISSUES_AND_ROADMAP.md
git status --short --untracked-files=all
git status --short --untracked-files=all | \
  rg '^.. (\.gitignore|docs/reference/uatvr_backbone_upgrade_strategy\.md|research_refs/)'
git diff --cached --name-only
```

Expected: 状态过滤只反映用户原有未提交改动；cached diff 为空。若发现本计划修改或暂存了这些路径，停止并恢复本计划自己的改动，不覆盖用户内容。

- [ ] **Step 6: 归档已执行规格和计划**

只有 Steps 1–5 全部成功后，使用 `apply_patch` 删除：

```text
docs/superpowers/specs/2026-07-11-sap-retirement-and-research-pivot-design.md
docs/superpowers/plans/2026-07-11-research-synthesis-and-ssot-plan.md
docs/superpowers/plans/2026-07-11-sap-retirement-code-cleanup-plan.md
```

Git 历史中的 commits `c3bda04`、`56569c3` 以及计划提交承担追溯职责。

- [ ] **Step 7: 提交最终墓碑与归档**

```bash
git add -- AGENTS.md docs/project/RESEARCH_ISSUES_AND_ROADMAP.md
git add -u -- \
  docs/superpowers/specs/2026-07-11-sap-retirement-and-research-pivot-design.md \
  docs/superpowers/plans/2026-07-11-research-synthesis-and-ssot-plan.md \
  docs/superpowers/plans/2026-07-11-sap-retirement-code-cleanup-plan.md
git commit -m "docs: finalize SAP retirement record" -- \
  AGENTS.md \
  docs/project/RESEARCH_ISSUES_AND_ROADMAP.md \
  docs/superpowers/specs/2026-07-11-sap-retirement-and-research-pivot-design.md \
  docs/superpowers/plans/2026-07-11-research-synthesis-and-ssot-plan.md \
  docs/superpowers/plans/2026-07-11-sap-retirement-code-cleanup-plan.md
```

- [ ] **Step 8: 重新运行提交后最小验证**

Run:

```bash
set -euo pipefail
git show --check --oneline --stat HEAD
/home/xujie/miniconda3/envs/ret/bin/pytest -q \
  tests/test_modeling_mulit_losses.py \
  tests/test_trusted_eval_protocol.py \
  tests/test_experiment_tracking.py
actual_status=$(git status --short --untracked-files=normal)
expected_status=$' M .gitignore\n D docs/reference/uatvr_backbone_upgrade_strategy.md\n?? research_refs/'
test "$actual_status" = "$expected_status"
```

Expected: commit 无空白错误，三组关键测试 PASS；porcelain 状态精确只剩三项已知用户改动。

---

## Plan B Completion Gate

完成条件必须同时成立：

```text
1. modules/modeling_mulit.py 不再 import/instantiate SAP、prob_models 或 SpatialEnhancer；SpatialEnhancer 源文件保留但无活动 CLI/接线。
2. train/eval final logits 只有 WTI；主损失仍是精确 video_id 双向多正例 InfoNCE。
3. 独立 hard-negative 默认关闭，hygiene 明确拒绝；default profile 可用于只读/消融诊断。
4. 旧 CLI 全部由 argparse SystemExit(2) 拒绝。
5. 含 SAP/probability/SpatialEnhancer 已删除参数的旧 model checkpoint 明确拒绝；旧 optimizer state 不迁移，参数组不兼容时原生失败。
6. manifest/log/TSV 不再含旧概率、QC、UACL、MoE 或 final-score 字段；纯 WTI MUS TSV 保留。
7. query_models/module_sap.py、旧 sweep 工具、tracked search_results 与 HN repeat launcher 已删除。
8. prob_models/、modules/modeling.py、modules/spatial_enhancer.py、modules/mus_util.py 和 WTI weight heads 保留。
9. Roadmap 仅有一条最终 SAP 墓碑；AGENTS 与其他长期文档无 SAP 入口。
10. 完整 tests、ruff、bash -n 均以本次新输出通过。
```

本计划不运行 P0 训练。验收后只向用户提供 `run_train_msrvtt_bg.sh` 的单行手动启动命令，并等待用户产生可信基线结果。
