# 可信实验基座 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MSRVTT 主线切换为唯一的 `trusted-v1` 实验协议，消除 test 选模、同视频错误负例、WTI padding 偏差和伪 hygiene 路径，并保留可审计的实验元数据。

**Architecture:** 用独立的 split/protocol 模块生成并校验固定 8500/500 manifest，dataloader 通过稳定 `video_group_id` 把同视频描述传入模型；模型以双向多正例 InfoNCE 作为唯一主检索损失，并为 hygiene 提供不触发 SAP/概率模块的早返回路径。训练入口只构造内部 val 或显式 test 中的一种，实验追踪模块把代码、数据、backbone 与四种 batch 口径写入 sidecar。

**Tech Stack:** Python 3.10+、PyTorch 2.x、torch.distributed、pytest、Bash、JSON/CSV、ruff、Git

## Global Constraints

- MSRVTT 只保留 `trusted-v1`；不新增或保留 legacy 协议开关。
- split 固定为 seed 42、8500 train、500 val；val 每视频恰好保留 20 条官方描述。
- JSFusion 1K test 只能通过独立 `--do_eval --eval_split test --init_model ...` 使用，训练不得构造或评估 test dataloader。
- 正例只由精确相同 `video_id` 定义；不引入语义软正例、伪标签或新 hard-negative 机制。
- 多正例 InfoNCE 是唯一主检索损失；不提供 diagonal CE 兼容开关。
- hygiene WTI-only 不请求视觉 patch hidden states，不执行 SpatialEnhancer、SAP、PIENet、不确定性头或概率辅助损失。
- OpenAI CLIP 保持默认 backbone；EVA adapter 是显式可选路径。
- 当前 dirty 工作树是用户指定的实施基础；先把已有文档和 adapter 修改分开提交，不覆盖或夹带无关修改。
- `RESEARCH_ISSUES_AND_ROADMAP.md` 是科研决策主文档；`STATUS.md` 只保留摘要；`plan.md` 简化为指向 roadmap 的入口。
- 不启动长期 GPU 训练；只运行单元测试、CPU smoke test、split 生成/校验和静态检查。

---

## File Map

### 先归档的现有修改

- `.gitignore`、`AGENTS.md`、`docs/README.md`、`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`、`docs/project/STATUS.md`、`docs/superpowers/specs/2026-07-10-research-roadmap-and-adapter-commit-design.md`：文档口径与开题报告大纲删除。
- `modules/backbone_adapter.py`：EVA-CLIP 到项目 CLIP-like 接口的适配层。
- `modules/modeling_mulit.py`、`main_task_retrieval.py`、`train_msrvtt.sh`、`eval.sh`：adapter 接入、参数、冻结策略和脚本入口。
- `tests/test_backbone_adapter.py`、`tests/test_main_task_hard_negative_args.py`、`tests/test_modeling_mulit_losses.py`：adapter 与相关回归测试。

### 新建文件

- `dataloaders/msrvtt_protocol.py`：trusted split 构建、哈希、校验和派生 CSV。
- `scripts/build_msrvtt_trusted_split.py`：split 构建/校验 CLI。
- `dataloaders/splits/msrvtt_trusted_v1_seed42.json`：版本化事实来源。
- `experiment_tracking.py`：Git/data/backbone/batch 元数据和原子 sidecar 写入。
- `tests/test_msrvtt_trusted_protocol.py`：split 与真实 manifest 测试。
- `tests/test_msrvtt_dataloader_contract.py`：group ID 和 multi-sentence dataloader 契约。
- `tests/test_trusted_eval_protocol.py`：CLI、loader 路由和 test 隔离。
- `tests/test_multi_positive_loss.py`：多正例损失与分布式 ID 聚合。
- `tests/test_experiment_tracking.py`：sidecar、Git 状态和 batch 口径。

### 修改文件

- `dataloaders/dataloader_msrvtt_retrieval.py`：稳定 group ID、val 多描述元数据和强校验。
- `dataloaders/data_dataloaders.py`：独立 train/val/test 构造器。
- `modules/until_module.py`：`MultiPositiveCrossEn` 与无梯度 all-gather。
- `modules/modeling_mulit.py`：group ID 数据流、WTI mask、hygiene 早返回和正例诊断。
- `main_task_retrieval.py`：可信 CLI、loader 选择、val 选模、可选 hidden eval 缓存和追踪接入。
- `train_msrvtt.sh`、`eval.sh`：唯一可信命令口径。
- `.gitignore`：忽略 `data/generated/`。
- `AGENTS.md`、`docs/README.md`、`docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`、`docs/project/STATUS.md`、`docs/project/plan.md`：科研决策与使用说明。
- `docs/superpowers/specs/2026-07-10-trusted-experiment-foundation-design.md`：实施后状态。

---

### Task 1: 归档当前文档口径修改

**Files:**
- Modify/commit: `.gitignore`
- Modify/commit: `AGENTS.md`
- Modify/commit: `docs/README.md`
- Modify/commit: `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`
- Modify/commit: `docs/project/STATUS.md`
- Modify/commit: `docs/superpowers/specs/2026-07-10-research-roadmap-and-adapter-commit-design.md`

**Interfaces:**
- Consumes: 当前 dirty 工作树中已完成的 QC-SAP 止损、roadmap/STATUS 对齐和开题报告索引删除。
- Produces: 一个只包含文档口径的提交；Task 2 的 adapter 提交不再夹带文档。

- [ ] **Step 1: 核对开题报告索引已从活动入口删除**

Run:

```bash
rg -n "开题报告_大纲" AGENTS.md docs/README.md
```

Expected: 无输出，退出码 1。

- [ ] **Step 2: 检查文档 diff 和空白**

Run:

```bash
git diff --check -- .gitignore AGENTS.md docs/README.md docs/project/RESEARCH_ISSUES_AND_ROADMAP.md docs/project/STATUS.md docs/superpowers/specs/2026-07-10-research-roadmap-and-adapter-commit-design.md
git diff --stat -- .gitignore AGENTS.md docs/README.md docs/project/RESEARCH_ISSUES_AND_ROADMAP.md docs/project/STATUS.md docs/superpowers/specs/2026-07-10-research-roadmap-and-adapter-commit-design.md
```

Expected: `git diff --check` 无输出；stat 只列出上述文件。

- [ ] **Step 3: 只暂存文档口径文件并核对集合**

Run:

```bash
git add .gitignore AGENTS.md docs/README.md docs/project/RESEARCH_ISSUES_AND_ROADMAP.md docs/project/STATUS.md docs/superpowers/specs/2026-07-10-research-roadmap-and-adapter-commit-design.md
git diff --cached --name-only
```

Expected: 只出现本任务列出的 6 个路径。

- [ ] **Step 4: 提交文档口径**

Run:

```bash
git commit -m "docs: align research roadmap and current status"
```

Expected: 提交成功；adapter 文件仍保持未提交。

---

### Task 2: 验证并提交当前 EVA backbone adapter

**Files:**
- Create/commit: `modules/backbone_adapter.py`
- Create/commit: `tests/test_backbone_adapter.py`
- Modify/commit: `modules/modeling_mulit.py`
- Modify/commit: `main_task_retrieval.py`
- Modify/commit: `tests/test_main_task_hard_negative_args.py`
- Modify/commit: `tests/test_modeling_mulit_losses.py`
- Modify/commit: `train_msrvtt.sh`
- Modify/commit: `eval.sh`

**Interfaces:**
- Consumes: 当前工作树中 `EvaClipBackboneAdapter.encode_text(..., return_hidden)` 和 `encode_image(..., return_hidden, video_frame)`。
- Produces: 已提交的 `openai_clip|eva_clip` backbone 选择、EVA B/16 spec、权重规范化、冻结策略；后续 hygiene 任务可安全修改统一接口。

- [ ] **Step 1: 运行 adapter 与模型回归测试**

Run:

```bash
pytest -q tests/test_backbone_adapter.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py
```

Expected: 全部 PASS；不下载模型权重。

- [ ] **Step 2: 检查脚本语法和 Python 静态问题**

Run:

```bash
bash -n train_msrvtt.sh
bash -n eval.sh
ruff check modules/backbone_adapter.py modules/modeling_mulit.py main_task_retrieval.py tests/test_backbone_adapter.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py
```

Expected: 三条命令均退出 0。

- [ ] **Step 3: 只暂存 adapter 文件并核对**

Run:

```bash
git add modules/backbone_adapter.py tests/test_backbone_adapter.py modules/modeling_mulit.py main_task_retrieval.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py train_msrvtt.sh eval.sh
git diff --cached --check
git diff --cached --name-only
```

Expected: 只出现本任务列出的 8 个路径。

- [ ] **Step 4: 提交 adapter**

Run:

```bash
git commit -m "feat: add EVA CLIP backbone adapter"
```

Expected: 提交成功；`git status --short` 不再显示 adapter 代码修改。

---

### Task 3: 构建并版本化 trusted-v1 split

**Files:**
- Create: `dataloaders/msrvtt_protocol.py`
- Create: `scripts/build_msrvtt_trusted_split.py`
- Create: `dataloaders/splits/msrvtt_trusted_v1_seed42.json`
- Create: `tests/test_msrvtt_trusted_protocol.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 官方 9000-video train CSV、`MSRVTT_v2.json`、JSFusion 1K CSV。
- Produces:
  - `build_trusted_manifest(train_csv, annotation_json, test_csv, seed=42, val_size=500, expected_captions=20) -> dict`
  - `validate_trusted_manifest(manifest, train_csv, annotation_json, test_csv) -> dict`
  - `write_generated_split_files(manifest, annotation_json, output_dir) -> dict[str, str]`
  - CLI 的 `--check-only` 校验入口。

- [ ] **Step 1: 写 split 的失败测试**

Create `tests/test_msrvtt_trusted_protocol.py` with these core tests:

```python
import csv
import json
from pathlib import Path

import pytest

from dataloaders.msrvtt_protocol import (
    build_trusted_manifest,
    validate_trusted_manifest,
    write_generated_split_files,
)


def _write_fixture(root: Path, train_count=9000, test_count=1000):
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    annotation = root / "annotation.json"
    with train_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id"])
        writer.writeheader()
        writer.writerows({"video_id": f"video{i}"} for i in range(train_count))
    with test_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "sentence"])
        writer.writeheader()
        writer.writerows(
            {"video_id": f"video{i}", "sentence": f"test {i}"}
            for i in range(train_count, train_count + test_count)
        )
    payload = {
        "videos": [{"video_id": f"video{i}"} for i in range(train_count)],
        "sentences": [
            {"video_id": f"video{i}", "caption": f"caption {i}-{j}"}
            for i in range(train_count)
            for j in range(20)
        ],
    }
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    return train_csv, annotation, test_csv


def test_trusted_split_is_deterministic_and_has_exact_counts(tmp_path):
    train_csv, annotation, test_csv = _write_fixture(tmp_path)
    first = build_trusted_manifest(train_csv, annotation, test_csv)
    second = build_trusted_manifest(train_csv, annotation, test_csv)
    assert first == second
    assert first["protocol_version"] == "trusted-v1"
    assert len(first["train_video_ids"]) == 8500
    assert len(first["val_video_ids"]) == 500
    assert set(first["train_video_ids"]).isdisjoint(first["val_video_ids"])


def test_generated_val_contains_twenty_grouped_captions_per_video(tmp_path):
    train_csv, annotation, test_csv = _write_fixture(tmp_path)
    manifest = build_trusted_manifest(train_csv, annotation, test_csv)
    paths = write_generated_split_files(manifest, annotation, tmp_path / "generated")
    with Path(paths["val_csv"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10000
    assert [row["video_id"] for row in rows[:20]] == [manifest["val_video_ids"][0]] * 20


def test_validation_rejects_train_test_overlap(tmp_path):
    train_csv, annotation, test_csv = _write_fixture(tmp_path)
    test_csv.write_text("video_id,sentence\nvideo0,leak\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train/test overlap.*video0"):
        build_trusted_manifest(train_csv, annotation, test_csv)


def test_validation_rejects_missing_caption(tmp_path):
    train_csv, annotation, test_csv = _write_fixture(tmp_path)
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["sentences"].pop()
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 20 captions"):
        build_trusted_manifest(train_csv, annotation, test_csv)
```

- [ ] **Step 2: 运行测试并确认红灯**

Run:

```bash
pytest -q tests/test_msrvtt_trusted_protocol.py
```

Expected: collection FAIL with `ModuleNotFoundError: dataloaders.msrvtt_protocol`。

- [ ] **Step 3: 实现 split 核心**

Create `dataloaders/msrvtt_protocol.py`. Use standard-library-only logic with these exact invariants:

```python
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


PROTOCOL_VERSION = "trusted-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_ids(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["video_id"].strip() for row in rows]
    if any(not video_id for video_id in ids):
        raise ValueError(f"empty video_id in {path}")
    duplicates = sorted(video_id for video_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate video_id in {path}: {duplicates[:5]}")
    return ids


def _caption_map(annotation_json):
    payload = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    captions = defaultdict(list)
    for row in payload["sentences"]:
        captions[row["video_id"]].append(row["caption"])
    return captions


def _raise_overlap(left_name, left_ids, right_name, right_ids):
    overlap = sorted(set(left_ids) & set(right_ids))
    if overlap:
        raise ValueError(
            f"{left_name}/{right_name} overlap count={len(overlap)} "
            f"examples={overlap[:5]}"
        )


def build_trusted_manifest(
    train_csv,
    annotation_json,
    test_csv,
    seed=42,
    val_size=500,
    expected_captions=20,
):
    train_source_ids = _read_csv_ids(train_csv)
    test_ids = _read_csv_ids(test_csv)
    if len(train_source_ids) != 9000:
        raise ValueError(f"expected 9000 source train videos, got {len(train_source_ids)}")
    if len(test_ids) != 1000:
        raise ValueError(f"expected 1000 test videos, got {len(test_ids)}")
    _raise_overlap("train", train_source_ids, "test", test_ids)
    captions = _caption_map(annotation_json)
    for video_id in train_source_ids:
        count = len(captions.get(video_id, []))
        if count != expected_captions:
            raise ValueError(
                f"video_id={video_id} expected {expected_captions} captions, got {count}"
            )
    shuffled = sorted(train_source_ids)
    random.Random(seed).shuffle(shuffled)
    val_ids = shuffled[:val_size]
    train_ids = shuffled[val_size:]
    _raise_overlap("train", train_ids, "val", val_ids)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "algorithm": "sorted video_id then random.Random(seed).shuffle; first val_size is val",
        "val_size": val_size,
        "expected_captions_per_video": expected_captions,
        "source_sha256": {
            "train_csv": sha256_file(train_csv),
            "annotation_json": sha256_file(annotation_json),
            "test_csv": sha256_file(test_csv),
        },
        "counts": {
            "source_train_videos": len(train_source_ids),
            "train_videos": len(train_ids),
            "val_videos": len(val_ids),
            "val_sentences": len(val_ids) * expected_captions,
            "test_videos": len(test_ids),
        },
        "test_video_ids_sha256": hashlib.sha256(
            ("\n".join(test_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "train_video_ids": train_ids,
        "val_video_ids": val_ids,
    }


def validate_trusted_manifest(manifest, train_csv, annotation_json, test_csv):
    expected = build_trusted_manifest(
        train_csv,
        annotation_json,
        test_csv,
        seed=manifest.get("seed"),
        val_size=manifest.get("val_size"),
        expected_captions=manifest.get("expected_captions_per_video"),
    )
    if manifest != expected:
        raise ValueError("trusted split manifest does not match current source files")
    return {
        "protocol_version": manifest["protocol_version"],
        "seed": manifest["seed"],
        "source_sha256": manifest["source_sha256"],
        **manifest["counts"],
        "manifest_sha256": hashlib.sha256(
            (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_generated_split_files(manifest, annotation_json, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captions = _caption_map(annotation_json)
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    summary_json = output_dir / "validation_summary.json"
    with train_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id"])
        writer.writeheader()
        writer.writerows({"video_id": video_id} for video_id in manifest["train_video_ids"])
    with val_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "sentence"])
        writer.writeheader()
        for video_id in manifest["val_video_ids"]:
            writer.writerows(
                {"video_id": video_id, "sentence": caption}
                for caption in captions[video_id]
            )
    _atomic_json(summary_json, manifest["counts"])
    return {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "summary_json": str(summary_json),
    }


def load_trusted_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_trusted_manifest(path, manifest):
    _atomic_json(path, manifest)
```

- [ ] **Step 4: 实现构建/校验 CLI**

Create `scripts/build_msrvtt_trusted_split.py`:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

from dataloaders.msrvtt_protocol import (
    build_trusted_manifest,
    load_trusted_manifest,
    validate_trusted_manifest,
    write_generated_split_files,
    write_trusted_manifest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = load_trusted_manifest(manifest_path)
        validate_trusted_manifest(
            manifest, args.train_csv, args.annotation_json, args.test_csv
        )
    elif args.check_only:
        raise FileNotFoundError(f"trusted manifest not found: {manifest_path}")
    else:
        manifest = build_trusted_manifest(
            args.train_csv, args.annotation_json, args.test_csv
        )
        write_trusted_manifest(manifest_path, manifest)
    if not args.check_only:
        write_generated_split_files(manifest, args.annotation_json, args.output_dir)
    print(
        f"trusted-v1 validated: train={len(manifest['train_video_ids'])} "
        f"val={len(manifest['val_video_ids'])}"
    )


if __name__ == "__main__":
    main()
```

Also add this exact ignore rule to `.gitignore`:

```gitignore
/data/generated/
```

- [ ] **Step 5: 运行单元测试并生成真实 manifest**

Run:

```bash
pytest -q tests/test_msrvtt_trusted_protocol.py
python3 scripts/build_msrvtt_trusted_split.py --train-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv --annotation-json /data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv --manifest dataloaders/splits/msrvtt_trusted_v1_seed42.json --output-dir data/generated/msrvtt_trusted_v1
```

Expected: tests PASS；CLI 输出 `trusted-v1 validated: train=8500 val=500`；生成的 manifest 被 Git 看到，派生 CSV 被忽略。

- [ ] **Step 6: 提交 split 基础**

Run:

```bash
git add .gitignore dataloaders/msrvtt_protocol.py scripts/build_msrvtt_trusted_split.py dataloaders/splits/msrvtt_trusted_v1_seed42.json tests/test_msrvtt_trusted_protocol.py
git diff --cached --check
git commit -m "feat: add trusted MSRVTT split manifest"
```

Expected: 提交成功。

---

### Task 4: 建立 train group ID 与 val 多描述 dataloader 契约

**Files:**
- Modify: `dataloaders/dataloader_msrvtt_retrieval.py:467-811`
- Modify: `dataloaders/data_dataloaders.py:7-146`
- Create: `tests/test_msrvtt_dataloader_contract.py`

**Interfaces:**
- Consumes: `load_trusted_manifest(path) -> dict`，派生 train/val CSV。
- Produces:
  - train batch 末尾稳定 `np.int64 video_group_id`；
  - `MSRVTT_DataLoader(..., multi_sentence_per_video=True, expected_captions_per_video=20)`；
  - 独立 `dataloader_msrvtt_val` 与 `dataloader_msrvtt_test`。

- [ ] **Step 1: 写 dataloader 契约失败测试**

Create `tests/test_msrvtt_dataloader_contract.py` with focused tests that monkeypatch video decoding:

```python
import csv
import json

import numpy as np
import pytest

from dataloaders.dataloader_msrvtt_retrieval import (
    MSRVTT_DataLoader,
    MSRVTT_TrainDataLoader,
)


class Tokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, words):
        return list(range(1, len(words) + 1))


def test_train_sample_returns_manifest_group_id(tmp_path, monkeypatch):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("video_id\nvideo_b\nvideo_a\n", encoding="utf-8")
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({
        "videos": [],
        "sentences": [
            {"video_id": video_id, "caption": f"{video_id}-{index}"}
            for video_id in ("video_b", "video_a")
            for index in range(20)
        ],
    }), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "train_video_ids": ["video_a", "video_b"],
        "val_video_ids": [],
    }), encoding="utf-8")
    monkeypatch.setattr(
        MSRVTT_TrainDataLoader,
        "_get_rawvideo",
        lambda self, ids: (
            np.zeros((1, 1, 1, 3, 2, 2), dtype=np.float32),
            np.ones((1, 1), dtype=np.int64),
        ),
    )
    dataset = MSRVTT_TrainDataLoader(
        csv_path=train_csv,
        json_path=annotation,
        features_path=tmp_path,
        tokenizer=Tokenizer(),
        max_frames=1,
        unfold_sentences=True,
        split_manifest_path=manifest,
    )
    assert len(dataset) == 40
    assert int(dataset[0][-1]) == 1
    assert int(dataset[20][-1]) == 0


def test_val_loader_exposes_multi_sentence_metadata(tmp_path):
    val_csv = tmp_path / "val.csv"
    with val_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "sentence"])
        writer.writeheader()
        for video_id in ("video_a", "video_b"):
            for index in range(20):
                writer.writerow({"video_id": video_id, "sentence": str(index)})
    dataset = MSRVTT_DataLoader(
        csv_path=val_csv,
        features_path=tmp_path,
        tokenizer=Tokenizer(),
        multi_sentence_per_video=True,
        expected_captions_per_video=20,
    )
    assert dataset.multi_sentence_per_video is True
    assert dataset.cut_off_points == [20, 40]
    assert dataset.sentence_num == 40
    assert dataset.video_num == 2


def test_val_loader_rejects_non_contiguous_video_rows(tmp_path):
    val_csv = tmp_path / "val.csv"
    val_csv.write_text(
        "video_id,sentence\nvideo_a,a\nvideo_b,b\nvideo_a,c\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        MSRVTT_DataLoader(
            csv_path=val_csv,
            features_path=tmp_path,
            tokenizer=Tokenizer(),
            multi_sentence_per_video=True,
            expected_captions_per_video=20,
        )
```

- [ ] **Step 2: 运行测试并确认接口缺失**

Run:

```bash
pytest -q tests/test_msrvtt_dataloader_contract.py
```

Expected: FAIL because `split_manifest_path` and `multi_sentence_per_video` are not accepted。

- [ ] **Step 3: 实现稳定 group ID**

In `MSRVTT_TrainDataLoader.__init__`, add mandatory trusted arguments and validation:

```python
from dataloaders.msrvtt_protocol import load_trusted_manifest

# signature additions
split_manifest_path=None,
expected_captions_per_video=20,

if not self.unfold_sentences:
    raise ValueError("trusted-v1 MSRVTT training requires unfold_sentences=True")
if not split_manifest_path:
    raise ValueError("trusted-v1 MSRVTT training requires split_manifest_path")
manifest = load_trusted_manifest(split_manifest_path)
self.video_group_ids = {
    video_id: index for index, video_id in enumerate(manifest["train_video_ids"])
}
if set(self.csv_video_ids) != set(self.video_group_ids):
    raise ValueError("train CSV video IDs do not match trusted manifest train_video_ids")
```

After building `sentences_dict`, validate each train video has exactly 20 entries. In `__getitem__`, define:

```python
video_group_id = np.int64(self.video_group_ids[video_id])
```

Append `video_group_id` to every return tuple:

- base: 6 fields；
- attributes: 9 fields；
- explicit HN base: 10 fields；
- explicit HN attributes: 13 fields。

Keep `sample_index` only on explicit-HN returns; do not reuse it as group ID.

- [ ] **Step 4: 实现 val 多描述元数据**

Add `multi_sentence_per_video=False` and `expected_captions_per_video=None` to `MSRVTT_DataLoader.__init__`. When enabled, construct metadata with this exact grouping rule:

```python
self.multi_sentence_per_video = bool(multi_sentence_per_video)
if self.multi_sentence_per_video:
    counts = []
    seen = set()
    current = None
    current_count = 0
    for video_id in self.video_ids:
        if video_id != current:
            if video_id in seen:
                raise ValueError(f"val rows must be contiguous by video_id: {video_id}")
            if current is not None:
                counts.append(current_count)
            seen.add(video_id)
            current = video_id
            current_count = 1
        else:
            current_count += 1
    if current is not None:
        counts.append(current_count)
    if expected_captions_per_video is not None and any(
        count != expected_captions_per_video for count in counts
    ):
        raise ValueError(
            f"each val video must have {expected_captions_per_video} captions; "
            f"observed counts={sorted(set(counts))}"
        )
    self.cut_off_points = np.cumsum(counts).tolist()
    self.sentence_num = len(self.video_ids)
    self.video_num = len(counts)
```

- [ ] **Step 5: 拆分 val/test 构造器**

In `dataloaders/data_dataloaders.py`, pass `split_manifest_path=args.split_manifest` to train and define:

```python
def dataloader_msrvtt_val(args, tokenizer, subset="val"):
    dataset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_words_attrs=getattr(args, "max_words_attrs", None),
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        use_attributes=getattr(args, "use_attributes", False),
        attributes_path=getattr(args, "msrvtt_attributes_path", ""),
        attr_num_blocks=getattr(args, "attr_num_blocks", 4),
        multi_sentence_per_video=True,
        expected_captions_per_video=20,
    )
    return _build_msrvtt_eval_loader(dataset, args), len(dataset)


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    dataset = MSRVTT_DataLoader(
        csv_path=args.test_csv,
        features_path=args.features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        max_words_attrs=getattr(args, "max_words_attrs", None),
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        use_attributes=getattr(args, "use_attributes", False),
        attributes_path=getattr(args, "msrvtt_attributes_path", ""),
        attr_num_blocks=getattr(args, "attr_num_blocks", 4),
        multi_sentence_per_video=False,
    )
    return _build_msrvtt_eval_loader(dataset, args), len(dataset)
```

Extract the repeated `DataLoader(...)` call into `_build_msrvtt_eval_loader`. Set:

```python
DATALOADER_DICT["msrvtt"] = {
    "train": dataloader_msrvtt_train,
    "val": dataloader_msrvtt_val,
    "test": dataloader_msrvtt_test,
}
```

- [ ] **Step 6: 运行测试并提交**

Run:

```bash
pytest -q tests/test_msrvtt_dataloader_contract.py tests/test_msrvtt_trusted_protocol.py
git add dataloaders/dataloader_msrvtt_retrieval.py dataloaders/data_dataloaders.py tests/test_msrvtt_dataloader_contract.py
git diff --cached --check
git commit -m "feat: enforce trusted MSRVTT dataloader contract"
```

Expected: tests PASS；提交成功。

---

### Task 5: 隔离训练 val 与显式 test 评估

**Files:**
- Modify: `main_task_retrieval.py:25-525`
- Modify: `main_task_retrieval.py:1649-1827`
- Create: `tests/test_trusted_eval_protocol.py`
- Modify: `tests/test_main_task_hard_negative_args.py`

**Interfaces:**
- Consumes: `DATALOADER_DICT[datatype]["train"|"val"|"test"]`。
- Produces:
  - `validate_trusted_cli(args) -> None`
  - `prepare_requested_dataloaders(args, tokenizer) -> tuple[tuple | None, object, int, str]`
  - `select_best_checkpoint(best_score, best_path, candidate_score, candidate_path) -> tuple[float, str]`
  - `select_multi_sentence_video_rows(batch_start, batch_size, cut_off_points) -> list[int]`
  - `--eval_split=val|test`、`--source_train_csv`、`--test_csv`、`--split_manifest`。

- [ ] **Step 1: 写 CLI 和 loader 路由失败测试**

Create `tests/test_trusted_eval_protocol.py`:

```python
from types import SimpleNamespace

import pytest

import main_task_retrieval as retrieval


def _args(**overrides):
    values = {
        "datatype": "msrvtt",
        "do_train": True,
        "do_eval": False,
        "eval_split": "val",
        "expand_msrvtt_sentences": True,
        "experiment_profile": "hygiene",
        "final_score_mode": "wti",
        "w_mil": 0.0,
        "w_evidential": 0.0,
        "w_neg_reg": 0.0,
        "w_orth": 0.0,
        "uncertainty_mode": "none",
        "use_hard_negative_packing": False,
        "use_explicit_hard_negative_loss": False,
        "use_uacl_intra_alignment": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_rejects_test_split():
    with pytest.raises(ValueError, match="training cannot use eval_split=test"):
        retrieval.validate_trusted_cli(_args(eval_split="test"))


def test_training_requires_expanded_captions():
    with pytest.raises(ValueError, match="requires --expand_msrvtt_sentences"):
        retrieval.validate_trusted_cli(_args(expand_msrvtt_sentences=False))


def test_hygiene_rejects_active_auxiliary_loss():
    with pytest.raises(ValueError, match="hygiene requires w_mil=0"):
        retrieval.validate_trusted_cli(_args(w_mil=0.01))


def test_training_constructs_val_but_not_test(monkeypatch):
    calls = []
    monkeypatch.setitem(retrieval.DATALOADER_DICT, "msrvtt", {
        "train": lambda args, tokenizer: (calls.append("train") or ("train", 1, "sampler")),
        "val": lambda args, tokenizer, subset="val": (calls.append("val") or ("val", 2)),
        "test": lambda args, tokenizer, subset="test": (calls.append("test") or ("test", 3)),
    })
    train_bundle, eval_loader, length, split = retrieval.prepare_requested_dataloaders(
        _args(), object()
    )
    assert train_bundle == ("train", 1, "sampler")
    assert (eval_loader, length, split) == ("val", 2, "val")
    assert calls == ["train", "val"]


def test_explicit_eval_constructs_only_test(monkeypatch):
    calls = []
    monkeypatch.setitem(retrieval.DATALOADER_DICT, "msrvtt", {
        "train": lambda args, tokenizer: (calls.append("train") or None),
        "val": lambda args, tokenizer, subset="val": (calls.append("val") or ("val", 2)),
        "test": lambda args, tokenizer, subset="test": (calls.append("test") or ("test", 3)),
    })
    result = retrieval.prepare_requested_dataloaders(
        _args(do_train=False, do_eval=True, eval_split="test"),
        object(),
    )
    assert result == (None, "test", 3, "test")
    assert calls == ["test"]


def test_checkpoint_selection_uses_internal_eval_score():
    assert retrieval.select_best_checkpoint(
        48.0, "epoch1.bin", 48.5, "epoch2.bin"
    ) == (48.5, "epoch2.bin")
    assert retrieval.select_best_checkpoint(
        48.5, "epoch2.bin", 48.1, "epoch3.bin"
    ) == (48.5, "epoch2.bin")


def test_multi_sentence_eval_selects_one_video_row_per_caption_group():
    cut_off_points = [19, 39]
    assert retrieval.select_multi_sentence_video_rows(
        batch_start=0,
        batch_size=25,
        cut_off_points=cut_off_points,
    ) == [19]
    assert retrieval.select_multi_sentence_video_rows(
        batch_start=25,
        batch_size=15,
        cut_off_points=cut_off_points,
    ) == [14]
```

- [ ] **Step 2: 运行测试并确认红灯**

Run:

```bash
pytest -q tests/test_trusted_eval_protocol.py
```

Expected: FAIL because `validate_trusted_cli` and `prepare_requested_dataloaders` do not exist。

- [ ] **Step 3: 添加参数与强校验**

Add to `get_args()`:

```python
parser.add_argument("--test_csv", type=str, default="data/.test.csv")
parser.add_argument("--source_train_csv", type=str, default="data/.source_train.csv")
parser.add_argument(
    "--split_manifest",
    type=str,
    default="dataloaders/splits/msrvtt_trusted_v1_seed42.json",
)
parser.add_argument(
    "--eval_split",
    choices=["val", "test"],
    default="val",
)
```

Replace the current global hygiene normalization block with this rule: MSRVTT uses the strict contract below; non-MSRVTT datasets retain their existing profile normalization until they receive their own trusted protocol:

```python
if args.datatype != "msrvtt" and args.experiment_profile == "hygiene":
    args.w_mil = 0.0
    args.w_evidential = 0.0
    args.w_neg_reg = 0.0
    args.w_orth = 0.0
    args.w_hard_negative = 0.0
    args.w_uacl_intra = 0.0
    args.w_uacl_kl = 0.0
    args.uncertainty_mode = "none"
    args.use_hard_negative_packing = False
    args.use_explicit_hard_negative_loss = False
    args.use_uacl_intra_alignment = False
```

Add the MSRVTT validator:

```python
def validate_trusted_cli(args):
    if args.datatype != "msrvtt":
        return
    if args.do_train and args.eval_split == "test":
        raise ValueError("trusted-v1 training cannot use eval_split=test")
    if args.do_train and not args.expand_msrvtt_sentences:
        raise ValueError(
            "trusted-v1 MSRVTT training requires --expand_msrvtt_sentences"
        )
    if args.do_eval and not args.do_train and not args.init_model:
        raise ValueError("--do_eval requires --init_model")
    if args.experiment_profile == "hygiene":
        required_zero = ("w_mil", "w_evidential", "w_neg_reg", "w_orth")
        for name in required_zero:
            if float(getattr(args, name)) != 0.0:
                raise ValueError(f"hygiene requires {name}=0")
        if args.uncertainty_mode != "none":
            raise ValueError("hygiene requires uncertainty_mode=none")
        if (
            args.use_hard_negative_packing
            or args.use_explicit_hard_negative_loss
            or args.use_uacl_intra_alignment
        ):
            raise ValueError("hygiene forbids HN and UACL paths")
        if args.final_score_mode != "wti":
            raise ValueError("hygiene trusted baseline requires final_score_mode=wti")
```

Call `validate_trusted_cli(args)` after parsing and before mutating batch size. Replace the old normalization test in `tests/test_main_task_hard_negative_args.py` with:

```python
def test_get_args_accepts_explicit_trusted_hygiene_contract(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--do_train",
            "--output_dir",
            "/tmp/uatvr-test-out",
            "--datatype",
            "msrvtt",
            "--experiment_profile",
            "hygiene",
            "--expand_msrvtt_sentences",
            "--final_score_mode",
            "wti",
            "--w_mil",
            "0",
            "--w_evidential",
            "0",
            "--w_neg_reg",
            "0",
            "--w_orth",
            "0",
            "--uncertainty_mode",
            "none",
        ],
    )
    args = get_args()
    assert args.experiment_profile == "hygiene"
    assert args.final_score_mode == "wti"
    assert args.w_mil == 0.0
    assert args.uncertainty_mode == "none"
```

The nonzero rejection remains covered by `test_hygiene_rejects_active_auxiliary_loss` in `tests/test_trusted_eval_protocol.py`.

- [ ] **Step 4: 实现单一 eval loader 选择**

Add:

```python
def select_best_checkpoint(
    best_score, best_path, candidate_score, candidate_path
):
    if candidate_score >= best_score:
        return candidate_score, candidate_path
    return best_score, best_path


def select_multi_sentence_video_rows(
    batch_start, batch_size, cut_off_points
):
    batch_end = batch_start + batch_size
    return [
        cut_off - batch_start
        for cut_off in cut_off_points
        if batch_start <= cut_off < batch_end
    ]


def prepare_requested_dataloaders(args, tokenizer):
    loaders = DATALOADER_DICT[args.datatype]
    train_bundle = loaders["train"](args, tokenizer) if args.do_train else None
    split = "val" if args.do_train else args.eval_split
    factory = loaders.get(split)
    if factory is None:
        raise ValueError(f"{args.datatype} has no {split} dataloader")
    eval_loader, eval_length = factory(args, tokenizer, subset=split)
    return train_bundle, eval_loader, eval_length, split
```

Replace lines 1687-1710 with one call. In the training loop:

```python
R1 = eval_epoch(args, model, eval_dataloader, device, n_gpu)
```

In `eval_epoch`, replace the inline cutoff list comprehension with `select_multi_sentence_video_rows(...)`; this makes the “encode one video row per 20-caption group” rule independently tested.

Rename log labels from “test” to `eval_split`. Delete the try/except that currently logs “Skipping evaluation”; trusted validation errors must propagate. Keep checkpoint selection as:

```python
best_score, best_output_model_file = select_best_checkpoint(
    best_score,
    best_output_model_file,
    R1,
    output_model_file,
)
logger.info(
    "Best val T2V R@1: %.1f | %s",
    best_score,
    best_output_model_file,
)
```

In `--do_eval`, use only the selected `eval_dataloader` and require `--init_model`.

- [ ] **Step 5: 运行协议测试并提交**

Run:

```bash
pytest -q tests/test_trusted_eval_protocol.py tests/test_main_task_hard_negative_args.py
git add main_task_retrieval.py tests/test_trusted_eval_protocol.py tests/test_main_task_hard_negative_args.py
git diff --cached --check
git commit -m "feat: isolate validation from blind test evaluation"
```

Expected: tests PASS；提交成功。

---

### Task 6: 用 video_group_id 接入双向多正例 InfoNCE

**Files:**
- Modify: `modules/until_module.py:210-224,341-357`
- Modify: `modules/modeling_mulit.py:19,390-395,503-580,682-1052`
- Modify: `main_task_retrieval.py:796-923`
- Create: `tests/test_multi_positive_loss.py`
- Modify: `tests/test_main_task_hard_negative_args.py`
- Modify: `tests/test_modeling_mulit_losses.py`

**Interfaces:**
- Consumes: train batch 最后一项 `video_group_id: LongTensor[B]`。
- Produces:
  - `MultiPositiveCrossEn.forward(logits, query_group_ids, candidate_group_ids=None) -> Tensor`
  - `allgather_with_grad(tensor, args) -> Tensor`
  - `allgather_no_grad(tensor, args) -> Tensor`
  - `UATVR.resolve_video_group_ids(video_group_id, local_batch, device, task_config) -> LongTensor`
  - `UATVR.forward(..., video_group_id=...)`
  - loss telemetry：`unique_video_count`、`duplicate_sample_count`、`mean_positive_count`。

- [ ] **Step 1: 写多正例损失失败测试**

Create `tests/test_multi_positive_loss.py`:

```python
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from modules.until_module import (
    MultiPositiveCrossEn,
    allgather_no_grad,
    allgather_with_grad,
)


def test_unique_groups_equal_diagonal_cross_entropy():
    logits = torch.tensor(
        [[4.0, 1.0, -1.0], [0.5, 3.0, 0.0], [-2.0, 1.0, 2.5]],
        requires_grad=True,
    )
    groups = torch.tensor([10, 11, 12])
    actual = MultiPositiveCrossEn()(logits, groups)
    expected = F.cross_entropy(logits, torch.arange(3))
    torch.testing.assert_close(actual, expected)


def test_same_video_columns_are_all_positives():
    logits = torch.tensor([[2.0, 1.0, 0.0], [1.5, 2.5, -1.0]])
    query_groups = torch.tensor([7, 7])
    candidate_groups = torch.tensor([7, 7, 9])
    actual = MultiPositiveCrossEn()(logits, query_groups, candidate_groups)
    expected = -(
        torch.logsumexp(logits[:, :2], dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    torch.testing.assert_close(actual, expected)


def test_missing_positive_fails_with_query_index():
    logits = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="query indices=\\[1\\]"):
        MultiPositiveCrossEn()(
            logits,
            torch.tensor([1, 2]),
            torch.tensor([1, 3]),
        )


def test_extreme_logits_have_finite_loss_and_gradients():
    logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
    loss = MultiPositiveCrossEn()(logits, torch.tensor([1, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_world_size_one_no_grad_gather_is_identity():
    tensor = torch.tensor([4, 5])
    gathered = allgather_no_grad(
        tensor, SimpleNamespace(world_size=1, rank=0)
    )
    assert torch.equal(gathered, tensor)


def test_world_size_one_grad_gather_preserves_gradient():
    tensor = torch.tensor([4.0, 5.0], requires_grad=True)
    gathered = allgather_with_grad(
        tensor, SimpleNamespace(world_size=1, rank=0)
    )
    gathered.sum().backward()
    assert torch.equal(tensor.grad, torch.ones_like(tensor))
```

Add `import pytest`.

- [ ] **Step 2: 运行测试并确认红灯**

Run:

```bash
pytest -q tests/test_multi_positive_loss.py
```

Expected: collection FAIL because `MultiPositiveCrossEn` is missing。

- [ ] **Step 3: 实现数值稳定的多正例损失和 ID 聚合**

Add to `modules/until_module.py`:

```python
class MultiPositiveCrossEn(nn.Module):
    def forward(self, logits, query_group_ids, candidate_group_ids=None):
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D, got shape={tuple(logits.shape)}")
        query_group_ids = query_group_ids.view(-1).to(logits.device)
        if candidate_group_ids is None:
            candidate_group_ids = query_group_ids
        candidate_group_ids = candidate_group_ids.view(-1).to(logits.device)
        if logits.shape != (query_group_ids.numel(), candidate_group_ids.numel()):
            raise ValueError(
                f"logit/group shape mismatch: logits={tuple(logits.shape)} "
                f"query={query_group_ids.numel()} candidate={candidate_group_ids.numel()}"
            )
        positive_mask = query_group_ids[:, None].eq(candidate_group_ids[None, :])
        missing = (~positive_mask.any(dim=1)).nonzero(as_tuple=False).view(-1)
        if missing.numel():
            raise ValueError(
                f"multi-positive target missing positive query indices={missing.tolist()}"
            )
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        return -(
            torch.logsumexp(positive_logits, dim=1)
            - torch.logsumexp(logits, dim=1)
        ).mean()


def allgather_no_grad(tensor, args):
    if int(args.world_size) == 1:
        return tensor
    output = [torch.empty_like(tensor) for _ in range(args.world_size)]
    torch.distributed.all_gather(output, tensor)
    return torch.cat(output, dim=0)


def allgather_with_grad(tensor, args):
    if int(args.world_size) == 1:
        return tensor
    return AllGather.apply(tensor, args)
```

Replace the module-local `allgather = AllGather.apply` calls as follows:

- differentiable floating-point features, means, variances and anchors use `allgather_with_grad(tensor, self.task_config)`；
- `attention_mask`、`video_mask`、`hard_valid` and `video_group_id` use `allgather_no_grad(tensor, self.task_config)`。

This preserves current DDP gradients, avoids attaching autograd to labels/masks, and makes CPU/world-size-one unit tests independent of an initialized process group.

- [ ] **Step 4: 把 group ID 传到全局 logits**

Change `_unpack_train_batch` to include `"video_group_id": None` and decode the new 6/9/10/13-field contracts from Task 4. Pass it from `train_epoch`:

```python
video_group_id=batch_inputs["video_group_id"],
```

Change `UATVR.forward`:

```python
@staticmethod
def resolve_video_group_ids(
    video_group_id, local_batch, device, task_config
):
    if video_group_id is not None:
        return video_group_id.view(-1).to(device=device, dtype=torch.long)
    if getattr(task_config, "datatype", "msrvtt") == "msrvtt":
        raise ValueError("trusted-v1 MSRVTT training requires video_group_id")
    rank = int(getattr(task_config, "rank", 0))
    return (
        torch.arange(local_batch, device=device, dtype=torch.long)
        + rank * 10_000_000
    )


def forward(
    self,
    input_ids,
    token_type_ids,
    attention_mask,
    video,
    video_mask=None,
    video_group_id=None,
    sample_index=None,
    hard_video=None,
    hard_video_mask=None,
    hard_valid=None,
):
    if self.training:
        video_group_id = self.resolve_video_group_ids(
            video_group_id,
            local_batch=input_ids.size(0),
            device=input_ids.device,
            task_config=self.task_config,
        )
```

The non-MSRVTT fallback creates globally unique IDs, so `MultiPositiveCrossEn` reduces exactly to diagonal CE without adding a legacy CLI path. Add:

```python
def test_non_msrvtt_fallback_group_ids_are_disjoint_across_ranks():
    rank0 = UATVR.resolve_video_group_ids(
        None,
        local_batch=2,
        device=torch.device("cpu"),
        task_config=Namespace(datatype="msvd", rank=0),
    )
    rank1 = UATVR.resolve_video_group_ids(
        None,
        local_batch=2,
        device=torch.device("cpu"),
        task_config=Namespace(datatype="msvd", rank=1),
    )
    assert set(rank0.tolist()).isdisjoint(rank1.tolist())
```

Pass `video_group_id` through `get_similarity_logits` into `_loose_similarity`. During the same training all-gather sequence as features:

```python
video_group_id = allgather_no_grad(
    video_group_id.contiguous().view(-1),
    self.task_config,
)
```

Return it as `res["video_group_id"]`.

- [ ] **Step 5: 替换唯一主检索损失并修正诊断**

Initialize:

```python
self.loss_fct = MultiPositiveCrossEn()
```

Replace diagonal loss:

```python
global_group_ids = res["video_group_id"]
sim_loss_t2v = self.loss_fct(
    sim_matrix,
    global_group_ids,
    global_group_ids,
)
sim_loss_v2t = self.loss_fct(
    sim_matrix.T,
    global_group_ids,
    global_group_ids,
)
sim_loss = (sim_loss_t2v + sim_loss_v2t) / 2
positive_mask = global_group_ids[:, None].eq(global_group_ids[None, :])
positive_counts = positive_mask.sum(dim=1).float()
unique_count = torch.unique(global_group_ids).numel()
```

Expose scalar telemetry:

```python
"unique_video_count": sim_matrix.new_tensor(float(unique_count)),
"duplicate_sample_count": sim_matrix.new_tensor(
    float(global_group_ids.numel() - unique_count)
),
"mean_positive_count": positive_counts.mean(),
```

Replace `_matrix_gap_stats` with:

```python
@staticmethod
def _matrix_gap_stats(logits, positive_mask=None):
    if logits is None:
        return {"diag": 0.0, "off": 0.0, "gap": 0.0, "std": 0.0}
    logits = logits.detach()
    if positive_mask is not None:
        positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
        if positive_mask.shape != logits.shape:
            raise ValueError(
                f"positive mask shape={tuple(positive_mask.shape)} "
                f"does not match logits={tuple(logits.shape)}"
            )
        positive = logits[positive_mask]
        negative = logits[~positive_mask]
    else:
        diag_len = min(logits.size(0), logits.size(1))
        diag_index = torch.arange(diag_len, device=logits.device)
        positive = logits[diag_index, diag_index]
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask[diag_index, diag_index] = False
        negative = logits[negative_mask]
    positive_mean = positive.mean() if positive.numel() else logits.new_zeros(())
    negative_mean = negative.mean() if negative.numel() else logits.new_zeros(())
    return {
        "diag": float(positive_mean.item()),
        "off": float(negative_mean.item()),
        "gap": float((positive_mean - negative_mean).item())
        if positive.numel() and negative.numel()
        else 0.0,
        "std": float(logits.std(unbiased=False).item())
        if logits.numel()
        else 0.0,
    }
```

Pass the group-derived mask to every training diagnostic. Extend `compute_query_conditioned_sap_logits(..., positive_mask=None)` and use the supplied mask for both score-gap and gate entropy/top-1 positive/negative statistics; retain diagonal fallback only for unlabeled rectangular eval chunks.

Add this regression test:

```python
def test_matrix_gap_stats_treats_same_video_off_diagonal_as_positive():
    logits = torch.tensor(
        [[4.0, 3.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 6.0]]
    )
    groups = torch.tensor([7, 7, 9])
    mask = groups[:, None].eq(groups[None, :])
    stats = UATVR._matrix_gap_stats(logits, positive_mask=mask)
    assert stats["diag"] == pytest.approx(
        (4.0 + 3.0 + 2.0 + 5.0 + 6.0) / 5
    )
    assert stats["off"] == pytest.approx(0.5)
```

- [ ] **Step 6: 更新 batch 与模型测试，运行全链路单测**

Update existing `_unpack_train_batch` tests to assert `video_group_id` separately from `sample_index`; add a 10-field explicit-HN case. Update `test_forward_accepts_explicit_hard_negative_kwargs` to include `video_group_id`.

Run:

```bash
pytest -q tests/test_multi_positive_loss.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py tests/test_msrvtt_dataloader_contract.py
```

Expected: all PASS。

- [ ] **Step 7: 提交多正例数据流**

Run:

```bash
git add modules/until_module.py modules/modeling_mulit.py main_task_retrieval.py tests/test_multi_positive_loss.py tests/test_main_task_hard_negative_args.py tests/test_modeling_mulit_losses.py
git diff --cached --check
git commit -m "feat: train retrieval with exact-video multi-positive loss"
```

Expected: 提交成功。

---

### Task 7: 修复 WTI padding 最大池化

**Files:**
- Modify: `modules/modeling_mulit.py:1373-1397`
- Modify: `tests/test_modeling_mulit_losses.py`

**Interfaces:**
- Consumes: `text_token[A,T,D]`、`frame_token[B,V,D]`、二值 text/video masks。
- Produces: padding 永远不能参与 max 的 WTI logits；全空样本明确失败。

- [ ] **Step 1: 写负相似度和空 mask 失败测试**

Append to `tests/test_modeling_mulit_losses.py`:

```python
def _tiny_wti_model():
    model = UATVR.__new__(UATVR)
    torch.nn.Module.__init__(model)
    model.text_weight_fc = torch.nn.Linear(2, 1, bias=False)
    model.video_weight_fc = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(model.text_weight_fc.weight)
    torch.nn.init.zeros_(model.video_weight_fc.weight)
    return model


def test_wti_padding_zero_cannot_beat_negative_valid_similarity():
    model = _tiny_wti_model()
    text = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    video = torch.tensor([[[-1.0, 0.0], [0.0, 0.0]]])
    logits = model.weighted_token_wise_intersection(
        text,
        video,
        torch.tensor([[1, 0]]),
        torch.tensor([[1, 0]]),
    )
    torch.testing.assert_close(logits, torch.tensor([[-1.0]]))


@pytest.mark.parametrize(
    ("text_mask", "video_mask", "message"),
    [
        (torch.tensor([[0, 0]]), torch.tensor([[1, 0]]), "no valid text token"),
        (torch.tensor([[1, 0]]), torch.tensor([[0, 0]]), "no valid video frame"),
    ],
)
def test_wti_rejects_empty_samples(text_mask, video_mask, message):
    model = _tiny_wti_model()
    tokens = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match=message):
        model.weighted_token_wise_intersection(
            tokens, tokens, text_mask, video_mask
        )
```

- [ ] **Step 2: 运行测试并确认 padding 回归失败**

Run:

```bash
pytest -q tests/test_modeling_mulit_losses.py -k "wti_padding or wti_rejects"
```

Expected: negative-similarity test FAIL because current result is 0；empty-mask test FAIL because no error is raised。

- [ ] **Step 3: 用 pairwise 布尔 mask 修复 WTI**

Replace `weighted_token_wise_intersection` body with:

```python
def weighted_token_wise_intersection(
    self, text_token, frame_token, attention_mask, video_mask
):
    text_valid = attention_mask.to(device=text_token.device, dtype=torch.bool)
    video_valid = video_mask.to(device=frame_token.device, dtype=torch.bool)
    empty_text = (~text_valid.any(dim=1)).nonzero(as_tuple=False).view(-1)
    empty_video = (~video_valid.any(dim=1)).nonzero(as_tuple=False).view(-1)
    if empty_text.numel():
        raise ValueError(
            f"WTI no valid text token at batch indices={empty_text.tolist()}"
        )
    if empty_video.numel():
        raise ValueError(
            f"WTI no valid video frame at batch indices={empty_video.tolist()}"
        )

    text_weight = self.text_weight_fc(text_token).squeeze(-1)
    text_weight = text_weight.masked_fill(~text_valid, float("-inf"))
    text_weight = torch.softmax(text_weight, dim=-1)
    video_weight = self.video_weight_fc(frame_token).squeeze(-1)
    video_weight = video_weight.masked_fill(~video_valid, float("-inf"))
    video_weight = torch.softmax(video_weight, dim=-1)

    similarities = torch.einsum("atd,bvd->abtv", text_token, frame_token)
    pair_valid = (
        text_valid[:, None, :, None]
        & video_valid[None, :, None, :]
    )
    similarities = similarities.masked_fill(
        ~pair_valid,
        torch.finfo(similarities.dtype).min,
    )
    t2v_logits = similarities.max(dim=-1).values
    t2v_logits = torch.einsum("abt,at->ab", t2v_logits, text_weight)
    v2t_logits = similarities.max(dim=-2).values
    v2t_logits = torch.einsum("abv,bv->ab", v2t_logits, video_weight)
    return (t2v_logits + v2t_logits) / 2.0
```

- [ ] **Step 4: 运行测试并提交**

Run:

```bash
pytest -q tests/test_modeling_mulit_losses.py -k "wti or final_score"
git add modules/modeling_mulit.py tests/test_modeling_mulit_losses.py
git diff --cached --check
git commit -m "fix: exclude padding from WTI max pooling"
```

Expected: tests PASS；提交成功。

---

### Task 8: 建立真正的 hygiene WTI-only 早返回路径

**Files:**
- Modify: `modules/modeling_mulit.py:390-461,503-855,1430-1472`
- Modify: `modules/backbone_adapter.py:18-33,58-84`
- Modify: `main_task_retrieval.py:1397-1438,1489-1534`
- Modify: `tests/test_modeling_mulit_losses.py`
- Modify: `tests/test_backbone_adapter.py`
- Create: `tests/test_eval_optional_features.py`

**Interfaces:**
- Consumes: `experiment_profile=hygiene`、`final_score_mode=wti`、全关闭辅助项。
- Produces:
  - `self.hygiene_wti_only: bool`
  - `get_visual_output(..., require_hidden: bool | None = None) -> tuple[Tensor, Tensor | None]`
  - `_wti_only_similarity(...)`
  - `BackboneSpec.supports_text_hidden` 与 `supports_visual_hidden`
  - eval 缓存可接受 `visual_hidden=None`。

- [ ] **Step 1: 写“不调用辅助模块”的失败测试**

Append to `tests/test_modeling_mulit_losses.py`:

```python
class _ExplodingModule(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("inactive hygiene module was called")


class _FakeClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.return_hidden_values = []

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        self.return_hidden_values.append(return_hidden)
        pooled = torch.ones(image.size(0), 2)
        if return_hidden:
            return pooled, pooled[:, None, :]
        return pooled


def test_hygiene_visual_encoder_does_not_request_patch_hidden():
    model = UATVR.__new__(UATVR)
    torch.nn.Module.__init__(model)
    model.clip = _FakeClip()
    model.hygiene_wti_only = True
    model.spatial_enhancer = _ExplodingModule()
    video = torch.zeros(2, 3, 2, 2)
    cls, hidden = model.get_visual_output(
        video,
        torch.tensor([[1, 1]]),
        shaped=True,
        video_frame=2,
    )
    assert cls.shape == (1, 2, 2)
    assert hidden is None
    assert model.clip.return_hidden_values == [False]


def test_hygiene_similarity_skips_sap_and_probability_modules():
    model = _tiny_wti_model()
    model.train()
    model.hygiene_wti_only = True
    model.task_config = Namespace(world_size=1, rank=0)
    model.clip = _FakeClip()
    model.sap = _ExplodingModule()
    model.pie_net_text = _ExplodingModule()
    model.uncertain_net_text = _ExplodingModule()
    result = model._wti_only_similarity(
        text_token=torch.tensor([[[1.0, 0.0]]]),
        visual_output=torch.tensor([[[1.0, 0.0]]]),
        attention_mask=torch.tensor([[1]]),
        video_mask=torch.tensor([[1]]),
        video_group_id=torch.tensor([5]),
    )
    assert result["retrieve_logits"].shape == (1, 1)
    assert result["video_group_id"].tolist() == [5]
```

Import `Namespace` from `argparse`.

Append to `tests/test_backbone_adapter.py`:

```python
def test_eva_backbone_spec_declares_hidden_feature_capabilities():
    spec = get_eva_clip_backbone_spec(
        "EVA02-CLIP-B-16",
        PROJECT_ROOT / "ref/EVA/EVA-CLIP/rei",
    )
    assert spec.supports_text_hidden is True
    assert spec.supports_visual_hidden is True
```

- [ ] **Step 2: 写 eval optional hidden 失败测试**

Create `tests/test_eval_optional_features.py`:

```python
import pytest
import torch

from main_task_retrieval import _cat_optional_tensors, _to_cpu_optional


def test_optional_feature_helpers_preserve_none():
    assert _cat_optional_tensors([None, None]) is None
    assert _to_cpu_optional(None) is None


def test_optional_feature_helpers_concatenate_tensors():
    result = _cat_optional_tensors([torch.ones(1, 2), torch.zeros(2, 2)])
    assert result.shape == (3, 2)


def test_optional_feature_helpers_reject_mixed_values():
    with pytest.raises(ValueError, match="mixed optional tensor"):
        _cat_optional_tensors([torch.ones(1), None])
```

- [ ] **Step 3: 运行测试并确认红灯**

Run:

```bash
pytest -q tests/test_modeling_mulit_losses.py -k "hygiene_visual or hygiene_similarity"
pytest -q tests/test_eval_optional_features.py
```

Expected: FAIL because `require_hidden`/`_wti_only_similarity`/optional helpers do not exist。

- [ ] **Step 4: 让视觉编码按需返回 hidden**

Set after loss activation resolution:

```python
self.hygiene_wti_only = (
    getattr(self.task_config, "experiment_profile", "default") == "hygiene"
    and self.final_score_mode == "wti"
    and not any(self.loss_activations.values())
)
```

Extend `BackboneSpec` with:

```python
supports_text_hidden: bool
supports_visual_hidden: bool
```

Set both values to `True` in `get_eva_clip_backbone_spec`. During EVA model initialization, fail before training if default profile requires SAP but `supports_visual_hidden` is false, or if WTI cannot obtain text hidden tokens. OpenAI CLIP is a built-in known-capable implementation.

Change `get_visual_output`:

```python
def get_visual_output(
    self,
    video,
    video_mask,
    shaped=False,
    video_frame=-1,
    require_hidden=None,
):
    if shaped is False:
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if video.dim() != 7:
            raise ValueError(
                f"Expected 7D video tensor, got shape={tuple(video.shape)}"
            )
        b, pair, num_frames, clips_per_frame, channel, height, width = video.shape
        video = video.view(
            b * pair * num_frames * clips_per_frame,
            channel,
            height,
            width,
        )
        video_frame = num_frames * clips_per_frame
    bs_pair = video_mask.size(0)
    if require_hidden is None:
        require_hidden = not self.hygiene_wti_only
    encoded = self.clip.encode_image(
        video,
        return_hidden=require_hidden,
        video_frame=video_frame,
    )
    if require_hidden:
        visual_cls, visual_hidden = encoded
        visual_hidden = visual_hidden.float().view(
            bs_pair, video_frame, visual_hidden.size(-2), visual_hidden.size(-1)
        )
        if self.spatial_enhancer is not None:
            batch, frames, token_count, dim = visual_hidden.shape
            if token_count > 1:
                spatial_tokens = visual_hidden[:, :, 1:, :]
                spatial_len = spatial_tokens.size(2)
                side = int(math.sqrt(spatial_len))
                if side * side == spatial_len:
                    if self.spatial_enhancer.rope_mode == "3d":
                        spatial_tokens = spatial_tokens.reshape(
                            batch, frames, side, side, dim
                        ).permute(0, 4, 1, 2, 3)
                        spatial_tokens = self.spatial_enhancer(spatial_tokens)
                        spatial_tokens = spatial_tokens.permute(
                            0, 2, 3, 4, 1
                        ).reshape(batch, frames, spatial_len, dim)
                    else:
                        spatial_tokens = spatial_tokens.reshape(
                            batch * frames, side, side, dim
                        ).permute(0, 3, 1, 2)
                        spatial_tokens = self.spatial_enhancer(spatial_tokens)
                        spatial_tokens = spatial_tokens.permute(
                            0, 2, 3, 1
                        ).reshape(batch, frames, spatial_len, dim)
                    visual_hidden = torch.cat(
                        [visual_hidden[:, :, :1, :], spatial_tokens],
                        dim=2,
                    )
    else:
        visual_cls = encoded
        visual_hidden = None
    visual_cls = visual_cls.float().view(bs_pair, -1, visual_cls.size(-1))
    return visual_cls, visual_hidden
```

Do not synthesize a dummy hidden tensor.

- [ ] **Step 5: 增加 hygiene 早返回**

Add:

```python
def _wti_only_similarity(
    self,
    text_token,
    visual_output,
    attention_mask,
    video_mask,
    video_group_id=None,
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
        video_group_id = allgather_no_grad(
            video_group_id.contiguous().view(-1),
            self.task_config,
        )
    text_token = F.normalize(text_token, dim=-1)
    visual_output = F.normalize(visual_output, dim=-1)
    logits = self.weighted_token_wise_intersection(
        text_token, visual_output, attention_mask, video_mask
    ) * self.clip.logit_scale.exp()
    if not self.training:
        return logits
    zero = logits.new_zeros(())
    return {
        "retrieve_logits": logits,
        "video_group_id": video_group_id,
        "MIL_loss": zero,
        "evidential_loss": zero,
        "neg_reg_loss": zero,
        "orth_loss": zero,
        "hard_negative_loss": zero,
        "uacl_intra_loss": zero,
        "uacl_kl_loss": zero,
    }
```

In `_loose_similarity`, after the shared seqTransf token/frame transformation and before any access to `visual_output_hidden`, add:

```python
if self.hygiene_wti_only:
    return self._wti_only_similarity(
        text_token,
        visual_output,
        attention_mask,
        video_mask,
        video_group_id=video_group_id,
    )
if visual_output_hidden is None:
    raise ValueError("default profile requires visual hidden tokens for SAP")
```

This branch must occur before `self.sap(...)`, `_evidential_similarity(...)` and `probabilistic_text(...)`.

- [ ] **Step 6: 让 eval 缓存支持 None**

Add to `main_task_retrieval.py`:

```python
def _to_cpu_optional(tensor):
    return None if tensor is None else tensor.cpu()


def _cat_optional_tensors(tensors):
    if all(tensor is None for tensor in tensors):
        return None
    if any(tensor is None for tensor in tensors):
        raise ValueError("mixed optional tensor values in eval cache")
    return torch.cat(tensors, dim=0)
```

Use `_to_cpu_optional(visual_output_all)` when caching. In `_run_on_single_gpu`:

```python
visual_output_all_full = _cat_optional_tensors(
    [value[1] for value in batch_visual_output_list]
)
full_chunk = (
    None
    if visual_output_all_full is None
    else visual_output_all_full[v_start:v_end].to(device)
)
```

When moving tuples to another GPU, map only tensors and preserve `None`.

- [ ] **Step 7: 运行 hygiene/default 回归并提交**

Run:

```bash
pytest -q tests/test_modeling_mulit_losses.py tests/test_eval_optional_features.py tests/test_backbone_adapter.py
git add modules/modeling_mulit.py modules/backbone_adapter.py main_task_retrieval.py tests/test_modeling_mulit_losses.py tests/test_backbone_adapter.py tests/test_eval_optional_features.py
git diff --cached --check
git commit -m "perf: make hygiene execute only the WTI path"
```

Expected: hygiene spy tests PASS；default SAP 相关既有测试仍 PASS；提交成功。

---

### Task 9: 写入实验 sidecar 与精确 batch 口径

**Files:**
- Create: `experiment_tracking.py`
- Create: `tests/test_experiment_tracking.py`
- Modify: `main_task_retrieval.py:497-596,888-1011,1715-1738`

**Interfaces:**
- Consumes: trusted split validation summary、args、dataloader step 数、loss batch telemetry。
- Produces:
  - `compute_batch_semantics(...) -> dict`
  - `build_experiment_manifest(...) -> dict`
  - `atomic_write_json(path, payload) -> None`
  - `append_batch_protocol_stats(...) -> None`
  - `output_dir/experiment_manifest.json`
  - `output_dir/batch_protocol_stats.tsv`。

- [ ] **Step 1: 写 tracking 失败测试**

Create `tests/test_experiment_tracking.py`:

```python
import json
from types import SimpleNamespace

from experiment_tracking import (
    atomic_write_json,
    build_experiment_manifest,
    compute_batch_semantics,
)


def test_batch_semantics_distinguishes_all_batch_sizes():
    result = compute_batch_semantics(
        requested_effective_batch=256,
        gradient_accumulation_steps=2,
        world_size=2,
        dataloader_steps=664,
        epochs=5,
    )
    assert result == {
        "requested_effective_batch": 256,
        "forward_global_contrastive_batch": 128,
        "per_rank_micro_batch": 64,
        "gradient_accumulation_steps": 2,
        "optimizer_effective_batch": 256,
        "forward_steps_per_epoch": 664,
        "optimizer_steps_per_epoch": 332,
        "total_optimizer_steps": 1660,
        "world_size": 2,
    }


def test_batch_semantics_rejects_non_divisible_values():
    with pytest.raises(ValueError, match="divisible"):
        compute_batch_semantics(255, 2, 2, 664, 5)


def test_manifest_contains_protocol_code_data_and_backbone(tmp_path):
    args = SimpleNamespace(
        seed=42,
        experiment_profile="hygiene",
        final_score_mode="wti",
        backbone_type="openai_clip",
        pretrained_clip_name="ViT-B/16",
        backbone_name="",
        backbone_path="",
        output_dir=str(tmp_path),
        train_csv="data/generated/msrvtt_trusted_v1/train.csv",
        val_csv="data/generated/msrvtt_trusted_v1/val.csv",
        test_csv="/data/MSRVTT_JSFUSION_test.csv",
        source_train_csv="/data/MSRVTT_train.9k.csv",
        data_path="/data/MSRVTT_v2.json",
        split_manifest="dataloaders/splits/msrvtt_trusted_v1_seed42.json",
    )
    payload = build_experiment_manifest(
        args,
        split_summary={"protocol_version": "trusted-v1", "manifest_sha256": "abc"},
        batch_semantics={"world_size": 2},
        git_state={"commit": "deadbeef", "dirty": True, "modified_paths": ["x.py"]},
    )
    assert payload["protocol_version"] == "trusted-v1"
    assert payload["git"]["dirty"] is True
    assert payload["backbone"]["type"] == "openai_clip"
    assert payload["data"]["test_csv"] == "/data/MSRVTT_JSFUSION_test.csv"
    path = tmp_path / "experiment_manifest.json"
    atomic_write_json(path, payload)
    assert json.loads(path.read_text())["split"]["manifest_sha256"] == "abc"
```

Add `import pytest`.

- [ ] **Step 2: 运行测试并确认红灯**

Run:

```bash
pytest -q tests/test_experiment_tracking.py
```

Expected: collection FAIL with `ModuleNotFoundError: experiment_tracking`。

- [ ] **Step 3: 实现 batch 与 manifest helper**

Create `experiment_tracking.py`:

```python
import csv
import json
import os
import subprocess
from pathlib import Path


def compute_batch_semantics(
    requested_effective_batch,
    gradient_accumulation_steps,
    world_size,
    dataloader_steps,
    epochs,
):
    if requested_effective_batch % gradient_accumulation_steps:
        raise ValueError("requested batch must be divisible by gradient accumulation")
    forward_global = requested_effective_batch // gradient_accumulation_steps
    if forward_global % world_size:
        raise ValueError("forward global batch must be divisible by world size")
    if dataloader_steps % gradient_accumulation_steps:
        raise ValueError("dataloader steps must be divisible by gradient accumulation")
    optimizer_steps = dataloader_steps // gradient_accumulation_steps
    return {
        "requested_effective_batch": requested_effective_batch,
        "forward_global_contrastive_batch": forward_global,
        "per_rank_micro_batch": forward_global // world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer_effective_batch": forward_global * gradient_accumulation_steps,
        "forward_steps_per_epoch": dataloader_steps,
        "optimizer_steps_per_epoch": optimizer_steps,
        "total_optimizer_steps": optimizer_steps * epochs,
        "world_size": world_size,
    }


def collect_git_state(project_root):
    root = Path(project_root)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    lines = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=root, text=True
    ).splitlines()
    return {
        "commit": commit,
        "dirty": bool(lines),
        "modified_paths": [line[3:] for line in lines],
    }


def build_experiment_manifest(args, split_summary, batch_semantics, git_state):
    return {
        "protocol_version": split_summary["protocol_version"],
        "git": git_state,
        "split": split_summary,
        "seed": args.seed,
        "profile": args.experiment_profile,
        "final_score_mode": args.final_score_mode,
        "backbone": {
            "type": args.backbone_type,
            "pretrained_clip_name": args.pretrained_clip_name,
            "name": getattr(args, "backbone_name", ""),
            "path": getattr(args, "backbone_path", ""),
        },
        "data": {
            "source_train_csv": args.source_train_csv,
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "annotation_json": args.data_path,
            "split_manifest": args.split_manifest,
        },
        "batch": batch_semantics,
        "losses": {
            name: getattr(args, name)
            for name in (
                "w_mil",
                "w_evidential",
                "w_neg_reg",
                "w_orth",
                "w_hard_negative",
                "w_uacl_intra",
                "w_uacl_kl",
            )
            if hasattr(args, name)
        },
    }


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_batch_protocol_stats(path, epoch, forward_step, global_step, stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "forward_step",
                "global_step",
                "unique_video_count",
                "duplicate_sample_count",
                "mean_positive_count",
            ],
            delimiter="\t",
        )
        if new_file:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch,
            "forward_step": forward_step,
            "global_step": global_step,
            **stats,
        })
```

- [ ] **Step 4: 保留 shell batch 并接入 sidecar**

Before mutating `args.batch_size` in `get_args()`:

```python
args.requested_effective_batch_size = args.batch_size
if args.batch_size % args.gradient_accumulation_steps:
    raise ValueError(
        "--batch_size must be divisible by --gradient_accumulation_steps"
    )
args.batch_size = args.batch_size // args.gradient_accumulation_steps
```

In `main()`, validate the committed manifest against all three canonical source files before tokenizer/model construction:

```python
split_summary = None
if args.datatype == "msrvtt":
    manifest = load_trusted_manifest(args.split_manifest)
    split_summary = validate_trusted_manifest(
        manifest,
        args.source_train_csv,
        args.data_path,
        args.test_csv,
    )
```

After building the trusted train dataloader, compute batch semantics, log every field, and atomically write:

```python
tracking_payload = build_experiment_manifest(
    args,
    split_summary=split_summary,
    batch_semantics=batch_semantics,
    git_state=collect_git_state(Path(__file__).resolve().parent),
)
atomic_write_json(
    Path(args.output_dir) / "experiment_manifest.json",
    tracking_payload,
)
```

Use `batch_semantics["total_optimizer_steps"]` for optimizer construction. Do not use the current floating-point expression at lines 1717-1719.

- [ ] **Step 5: 记录每个 forward batch 的正例分布**

In `train_epoch`, after `loss_dict` is returned and only on rank 0:

```python
batch_stats = {
    key: float(loss_dict[key])
    for key in (
        "unique_video_count",
        "duplicate_sample_count",
        "mean_positive_count",
    )
}
append_batch_protocol_stats(
    Path(args.output_dir) / "batch_protocol_stats.tsv",
    epoch=epoch,
    forward_step=step,
    global_step=global_step,
    stats=batch_stats,
)
```

Also include the three values in the periodic logger line. Remove them from gradient-bearing loss accumulation.

- [ ] **Step 6: 运行 tracking 与训练入口测试并提交**

Run:

```bash
pytest -q tests/test_experiment_tracking.py tests/test_trusted_eval_protocol.py tests/test_main_task_hard_negative_args.py tests/test_multi_positive_loss.py
git add experiment_tracking.py tests/test_experiment_tracking.py main_task_retrieval.py
git diff --cached --check
git commit -m "feat: record trusted experiment provenance"
```

Expected: tests PASS；提交成功。

---

### Task 10: 迁移脚本与科研文档到唯一可信口径

**Files:**
- Modify: `train_msrvtt.sh`
- Modify: `eval.sh`
- Modify: `AGENTS.md`
- Modify: `docs/README.md`
- Modify: `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md`
- Modify: `docs/project/STATUS.md`
- Modify: `docs/project/plan.md`
- Modify: `docs/superpowers/specs/2026-07-10-trusted-experiment-foundation-design.md`

**Interfaces:**
- Consumes: split CLI、trusted 训练/评估参数。
- Produces: 单一训练命令、显式 val/test 评估命令、roadmap 的新决策门槛。

- [ ] **Step 1: 更新训练脚本先生成派生 CSV**

Add near the top of `train_msrvtt.sh`:

```bash
SOURCE_TRAIN_CSV="${DATA_PATH}/csv/MSRVTT_train.9k.csv"
TEST_CSV="${DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv"
ANNOTATION_JSON="${DATA_PATH}/annotation/MSRVTT_v2.json"
SPLIT_MANIFEST="dataloaders/splits/msrvtt_trusted_v1_seed42.json"
GENERATED_SPLIT_DIR="data/generated/msrvtt_trusted_v1"
python3 scripts/build_msrvtt_trusted_split.py --train-csv "${SOURCE_TRAIN_CSV}" --annotation-json "${ANNOTATION_JSON}" --test-csv "${TEST_CSV}" --manifest "${SPLIT_MANIFEST}" --output-dir "${GENERATED_SPLIT_DIR}"
```

Replace train/eval CSV arguments with:

```bash
--train_csv "${GENERATED_SPLIT_DIR}/train.csv" \
--val_csv "${GENERATED_SPLIT_DIR}/val.csv" \
--source_train_csv "${SOURCE_TRAIN_CSV}" \
--test_csv "${TEST_CSV}" \
--split_manifest "${SPLIT_MANIFEST}" \
--eval_split val \
```

Keep `--expand_msrvtt_sentences`. Ensure hygiene explicitly passes `--final_score_mode wti --w_mil 0 --w_evidential 0 --w_neg_reg 0 --w_orth 0 --uncertainty_mode none`.

- [ ] **Step 2: 让评估脚本要求显式 split**

At the top of `eval.sh`:

```bash
: "${EVAL_SPLIT:?请显式设置 EVAL_SPLIT=val 或 EVAL_SPLIT=test}"
if [[ "${EVAL_SPLIT}" != "val" && "${EVAL_SPLIT}" != "test" ]]; then
  echo "EVAL_SPLIT 只能是 val 或 test" >&2
  exit 2
fi
```

In the MSRVTT branch, define one repository-local generated directory and run the split builder:

```bash
SOURCE_TRAIN_CSV="${MSRVTT_DATA_PATH}/csv/MSRVTT_train.9k.csv"
TEST_CSV="${MSRVTT_DATA_PATH}/csv/MSRVTT_JSFUSION_test.csv"
ANNOTATION_JSON="${MSRVTT_DATA_PATH}/annotation/MSRVTT_v2.json"
SPLIT_MANIFEST="dataloaders/splits/msrvtt_trusted_v1_seed42.json"
GENERATED_SPLIT_DIR="data/generated/msrvtt_trusted_v1"
python3 scripts/build_msrvtt_trusted_split.py --train-csv "${SOURCE_TRAIN_CSV}" --annotation-json "${ANNOTATION_JSON}" --test-csv "${TEST_CSV}" --manifest "${SPLIT_MANIFEST}" --output-dir "${GENERATED_SPLIT_DIR}"
```

Pass these arguments to `main_task_retrieval.py`:

```bash
--source_train_csv "${SOURCE_TRAIN_CSV}" \
--val_csv "${GENERATED_SPLIT_DIR}/val.csv" \
--test_csv "${TEST_CSV}" \
--split_manifest "${SPLIT_MANIFEST}" \
--eval_split "${EVAL_SPLIT}" \
```

For `EXPERIMENT_PROFILE=hygiene`, construct an array and append it after the base args so the trusted values win:

```bash
EXTRA_PROFILE_ARGS=()
if [[ "${EXPERIMENT_PROFILE}" == "hygiene" ]]; then
  EXTRA_PROFILE_ARGS+=(--final_score_mode wti)
  EXTRA_PROFILE_ARGS+=(--w_mil 0 --w_evidential 0 --w_neg_reg 0 --w_orth 0)
  EXTRA_PROFILE_ARGS+=(--uncertainty_mode none)
fi

# At the end of the Python command:
"${EXTRA_PROFILE_ARGS[@]}"
```

Use only the repository-local `data/generated/msrvtt_trusted_v1/` directory. Remove the eval-only `--expand_msrvtt_sentences`, because it is a train dataset contract.

- [ ] **Step 3: 把 roadmap 改为“先可信基线、后 backbone”**

Add a dated P0 decision block to `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` containing these exact claims:

```markdown
### P0：可信实验基座与新基线

- 旧结果存在 JSFusion test 逐 epoch 选模、同视频描述被当作负例、WTI padding 最大池化三项混杂，只保留为历史档案。
- 新协议固定为 trusted-v1：8500 train / 500 internal val / JSFusion 1K blind test。
- 主损失按精确 video_id 使用双向多正例 InfoNCE。
- 下一次可解释实验必须先重跑 OpenAI CLIP hygiene WTI-only；未完成该基线前，不判断 EVA adapter、SAP 或不确定性模块收益。
- OpenAI hygiene 新基线建立后，EVA02-CLIP-B/16 只能在相同 split、global contrastive batch、optimizer steps 和 checkpoint-selection 指标下比较。
```

Remove the current “直接进入 EVA backbone-only 实验”的优先级表述。

- [ ] **Step 4: 精简 STATUS 与 plan**

Replace `docs/project/STATUS.md` with:

```markdown
# UATVR 当前状态

> 更新时间：2026-07-10

## 当前判断

历史 MSRVTT 数字受到 JSFusion test 逐 epoch 选模、同视频描述错误负例和 WTI padding 三项混杂影响，只作为历史档案，不再直接用于新模型决策。

## 已终止方向

Hard negative、UACL、置信度加权和 query-conditioned SAP 不再作为训练主线；其结果保留为负结果与边界分析。

## 唯一实验协议

- `trusted-v1`：8500 train / 500 internal val / JSFusion 1K blind test；
- val 使用每视频全部 20 条官方描述；
- checkpoint 只按 val T2V R@1 选择；
- 主损失为按精确 `video_id` 定义的双向多正例 InfoNCE；
- hygiene 只执行 WTI 路径。

## 下一步

先重跑 OpenAI CLIP trusted hygiene WTI-only 基线；基线建立后，才在完全相同协议下评估 EVA02-CLIP-B/16 adapter。

完整问题、证据和停止条件见
[`RESEARCH_ISSUES_AND_ROADMAP.md`](RESEARCH_ISSUES_AND_ROADMAP.md)。
```

Replace `docs/project/plan.md` with:

```markdown
# 实验计划入口

科研问题、停止条件、证据等级和后续实验顺序已统一维护在
[`RESEARCH_ISSUES_AND_ROADMAP.md`](RESEARCH_ISSUES_AND_ROADMAP.md)。

本文件不再维护独立实验优先级，以避免与 roadmap 和 `STATUS.md` 产生重复口径。
历史实验结论请查阅 Git 历史及 `docs/logs/README.md`。
```

In the current-state section of `AGENTS.md`, insert:

```markdown
- MSRVTT 唯一有效协议为 `trusted-v1`：8500/500 内部拆分，JSFusion 1K 只作显式盲测，主损失按精确 `video_id` 使用双向多正例 InfoNCE。
- 下一步先重跑 OpenAI CLIP hygiene WTI-only 新基线；未完成前不判断 EVA adapter、SAP 或 uncertainty 收益。
- 科研决策以 `docs/project/RESEARCH_ISSUES_AND_ROADMAP.md` 为唯一主入口，`STATUS.md` 只提供摘要。
```

In `docs/README.md`, label `RESEARCH_ISSUES_AND_ROADMAP.md` as “科研决策唯一主文档” and `STATUS.md` as “当前摘要”. Mark the design spec status as `已实施`.

- [ ] **Step 5: 验证脚本和文档口径**

Run:

```bash
bash -n train_msrvtt.sh
bash -n eval.sh
rg -n -- "--val_csv.*JSFUSION" train_msrvtt.sh eval.sh
rg -n "当前下一步.*EVA|直接进入.*EVA" docs/project AGENTS.md
git diff --check
```

Expected:

- shell syntax PASS；
- 两条 `rg` 命令均无输出、退出码 1；
- diff check 无输出。

- [ ] **Step 6: 提交脚本与文档迁移**

Run:

```bash
git add train_msrvtt.sh eval.sh AGENTS.md docs/README.md docs/project/RESEARCH_ISSUES_AND_ROADMAP.md docs/project/STATUS.md docs/project/plan.md docs/superpowers/specs/2026-07-10-trusted-experiment-foundation-design.md
git diff --cached --check
git commit -m "docs: make trusted protocol the research baseline"
```

Expected: 提交成功。

---

### Task 11: 完整验证与交付命令

**Files:**
- Verify only: all files from Tasks 1-10
- Modify only if a verification failure exposes a scoped defect

**Interfaces:**
- Consumes: 全部 trusted-v1 实现。
- Produces: 可交付的 clean/known-dirty 状态、测试证据、用户手动运行命令。

- [ ] **Step 1: 校验真实 split 与派生文件**

Run:

```bash
python3 scripts/build_msrvtt_trusted_split.py --train-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv --annotation-json /data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json --test-csv /data2/hxj/data/MSRVTT/csv/MSRVTT_JSFUSION_test.csv --manifest dataloaders/splits/msrvtt_trusted_v1_seed42.json --output-dir data/generated/msrvtt_trusted_v1 --check-only
```

Expected: `trusted-v1 validated: train=8500 val=500`。

- [ ] **Step 2: 运行重点单元测试**

Run:

```bash
pytest -q tests/test_msrvtt_trusted_protocol.py tests/test_msrvtt_dataloader_contract.py tests/test_trusted_eval_protocol.py tests/test_multi_positive_loss.py tests/test_modeling_mulit_losses.py tests/test_eval_optional_features.py tests/test_experiment_tracking.py tests/test_backbone_adapter.py tests/test_main_task_hard_negative_args.py
```

Expected: all PASS。

- [ ] **Step 3: 运行全量测试和静态检查**

Run:

```bash
pytest -q
ruff check dataloaders/msrvtt_protocol.py scripts/build_msrvtt_trusted_split.py dataloaders/dataloader_msrvtt_retrieval.py dataloaders/data_dataloaders.py modules/until_module.py modules/modeling_mulit.py modules/backbone_adapter.py experiment_tracking.py main_task_retrieval.py tests/test_msrvtt_trusted_protocol.py tests/test_msrvtt_dataloader_contract.py tests/test_trusted_eval_protocol.py tests/test_multi_positive_loss.py tests/test_modeling_mulit_losses.py tests/test_eval_optional_features.py tests/test_experiment_tracking.py tests/test_backbone_adapter.py tests/test_main_task_hard_negative_args.py
bash -n train_msrvtt.sh
bash -n eval.sh
git diff --check
```

Expected: 全部退出 0。

- [ ] **Step 4: 做 CPU smoke test**

Run:

```bash
pytest -q tests/test_multi_positive_loss.py::test_unique_groups_equal_diagonal_cross_entropy tests/test_modeling_mulit_losses.py::test_wti_padding_zero_cannot_beat_negative_valid_similarity tests/test_modeling_mulit_losses.py::test_hygiene_similarity_skips_sap_and_probability_modules
```

Expected: 3 PASS；无 CUDA 初始化、无权重下载。

- [ ] **Step 5: 审计提交和工作树**

Run:

```bash
git log --oneline --decorate -12
git status --short --branch
git diff --stat
```

Expected: 能清楚看到文档、adapter、split、dataloader、评估隔离、多正例、WTI、hygiene、追踪、文档迁移的独立提交；没有被遗漏的实现修改。若存在用户原有的无关修改，逐项列出而不提交。

- [ ] **Step 6: 交付用户手动运行命令，不执行**

OpenAI CLIP trusted hygiene baseline:

```bash
EXPERIMENT_PROFILE=hygiene FINAL_SCORE_MODE=wti RUN_ID=202607xx_trusted_v1_openai_hygiene CUDA_VISIBLE_DEVICES=1,2 bash run_train_msrvtt_bg.sh
```

内部 val 复评:

```bash
EVAL_SPLIT=val EXPERIMENT_PROFILE=hygiene FINAL_SCORE_MODE=wti INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> CUDA_VISIBLE_DEVICES=4 bash eval.sh
```

JSFusion 1K 一次性盲测:

```bash
EVAL_SPLIT=test EXPERIMENT_PROFILE=hygiene FINAL_SCORE_MODE=wti INIT_MODEL=ckpts/<run>/pytorch_model.bin.<N> CUDA_VISIBLE_DEVICES=4 bash eval.sh
```

Do not run these three commands during implementation.
