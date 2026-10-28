# S2 归档事务恢复（2026-10-10）

五个 matcher 科学 worker 均 exit 0 并完成 128 次 query 访问，控制器已递归验证
6128 个数组引用并将五个 run 置为 completed。阶段 finalization 随后因五份完整
output_hashes 合计超过摘要 1 MiB 上限而拒绝；并非训练失败。

在 `819f5fa` 干净预留源码上，先保存完整待提交 registry 到
`/tmp/evidence-S2-pending-registry-819f5fa.jsonl`，SHA-256
`6baa3180d7360ec0dc988ed27114c93c766425b386e62eeeb5dafc46ba7c1e4e`。
确认它严格等于 HEAD registry 追加这五个 completed 事件，没有其他变化。
临时恢复 HEAD registry 后逐 run 执行生产 `verify_run`，身份全部通过；重新计算
2238/2238/2238/2243/2243 个输出文件 SHA-256，与每个 terminal status 完全一致。
没有改写终态产物、历史事件、权重或研究 gate。

修复仅使大型阶段摘要 SHA-256 绑定既有完整 terminal status inventory；
manifest/registry/status/audit schema 及 1 MiB 限制保持不变，小摘要保持原格式。
最终恢复须回填上述精确 pending append，再用生产 finalization 比对并提交整批
terminal registry 与摘要，不重新运行 worker 或重复执行 terminal transition。

行为测试用两个真实归档目录、5000 个实际输出文件覆盖超限、摘要预验证失败无
registry 追加、每个引用及输出哈希验证、精确恢复与提交后重入。

独立审查发现 Unicode 文件名的大小预验证与 writer 默认 ASCII 转义不一致。
统一 fallback、预验证及 atomic writer 均按实际 `ensure_ascii=True` 字节计算。
新增测试先观察到未拒绝的 RED，再通过；真实双 run 用 ASCII 2500 或 Unicode
2000 个输出/运行验证所有条目。独立定向复核 3 项通过，无剩余阻塞问题。

实际验证（统一前缀 `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=1 /home/xujie/.conda/envs/tvr/bin/python -m pytest -p no:cacheprovider`）：
- `tests/test_manage_experiment_run.py tests/test_experiment_archive_e2e.py tests/test_experiment_artifact_docs.py tests/test_experiment_artifacts.py tests/test_experiment_tracking.py tests/test_legacy_artifact_migration.py -q`：176 passed，78.74s（Unicode 修复前）。
- Unicode 修复后 `tests/test_manage_experiment_run.py tests/test_experiment_archive_e2e.py tests/test_experiment_artifact_docs.py -q`：50 passed，86.65s。
- `tests/semantic_retrieval/evidence -q`：107 passed，45.10s，既有 TypedStorage 警告一条。
