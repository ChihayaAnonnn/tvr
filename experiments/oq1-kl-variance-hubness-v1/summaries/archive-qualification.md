# Stage 0 · 四卡归档资格测试 —— PASS（2026-08-29）

PC-9 要求的四卡资格测试已在本 study 自己的 registry/数据面上完成并登记。
四个场景全部按设计行为运行,证据齐全。

## 终态运行

| 场景 | Run ID | 终态 |
|---|---|---|
| success | `msrvtt-baseline-qual_success-seed000-20260829T103000Z-ae601bd295ee` | `completed`, exit 0 |
| non-owner rank failure | `msrvtt-baseline-qual_nonowner_fail-seed000-20260829T103100Z-ae601bd295ee` | `failed`, exit 1 |
| owner collective failure | `msrvtt-baseline-qual_owner_collective_fail-seed000-20260829T103200Z-ae601bd295ee` | `failed`, exit 1 |
| SIGTERM, mismatched collectives（重试） | `msrvtt-baseline-qual_sigterm_mismatch-seed000-20260829T110000Z-9af92caa39e8` | `cancelled`, exit 143 |

设备:探测选定 GPU {0,1,3,4}(各 81142 MiB 空闲,driver 560.35.05),探测记录在各
manifest 的 `selection_lineage` 与 run 目录 `provenance/gpu-probe-launch.json`。

## 关键证据

- **success**:真实四 rank CUDA/NCCL 优化步,`qualification_loss = 7.5`
  (= (1²+2²+3²+4²)/4,解析值精确);2,097,152 字节的 (524288,) float32 数组按内容
  寻址外置,协议摘要 `908251af60385de2…`、文件摘要 `647092735d0f8309…`——
  **与 contextual-risk-v2 同一夹具的历史摘要逐位一致**(交叉验证)。
- **non-owner / owner failure**:计划内异常经 torchrun 传播,exit 1,rank 打点与
  (owner 场景)`worker_failure.json` 保留。
- **SIGTERM 排水**:4 rank 停驻在错配 collective(NCCL 忙等)窗口内,SIGTERM 命中
  存活控制器;`controller.log.process-group-drain.txt` 记录 `signal:SIGTERM →
  timeout:SIGTERM → timeout:SIGKILL` 完整升级链,组完全排空,无
  `UNRESOLVED:live-process-group` 记录。

## 途中取消的预留(全部证据保留,`cancelled` 不可变)

1. **第一批 4 个**(`…-90c583c8db05`):`device_selection_probe` 放进
   `config.evaluation` 触发 effective-config 逐键核对失败。修正:探测记录移入
   `selection_lineage`(仍属 manifest,满足 study.yaml 的 `record_in_manifest`)。
2. **第二批 4 个**(`…-f51f0232e970`):manifest 写 `torchrun-4-rank`(study.yaml
   叫法),运行时推导为 `torchrun-4`。修正:环境字段用运行时词表。
3. **第三批的 sigterm 首试**(`…-20260829T103300Z-ae601bd295ee`,`failed` 143):
   悬挂窗口已确认,但控制器提前退出,SIGTERM 未命中存活控制器,排水证据缺失;
   孤儿组手动 SIGTERM 排空(GPU 全部释放),`provenance/attempt-note.json` 记录经过。

## 与 stage0_qualification 门的对账

| 门 | 状态 |
|---|---|
| `four_gpu_qualification_recorded` | ✅ 本文档 + registry 终态行 |
| `val1k_disjoint_from_test` | ✅ `data/generated/msrvtt_trusted_v1_val1k/manifest.json`(构建即断言,重建逐字节一致) |
| `source_tree_clean_and_sha_bound` | ✅ 每次 launch 前 `verify-run` 通过;run-id 内嵌精确 git sha |
| `determinism_first_200_steps_bitwise_equal_same_seed` | ⬜ **未完成**——依赖 arm 训练实现(需先落地概率头 + val1k 评测 + 注册 launcher,且须翻回 `cudnn.deterministic`,见 `main_task_retrieval.py:1240-1242`) |

**结论:Stage 0 完成 3/4;确定性契约与 arm 实现绑定,是 Stage 1 前的最后一项。**
