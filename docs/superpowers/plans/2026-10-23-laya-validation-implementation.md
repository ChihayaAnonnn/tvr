# Laya validation implementation plan

Goal: frozen local Laya integration with CPU correctness tests and fixed independent control plane.
Architecture: raw RGB evidence → native Laya judge → stable full-gallery evaluator; immutable visual/query caches. Existing UATVR scoring remains unchanged.

- [x] Save approved design and create branch from codex/full-data-baselines.
- [x] Reuse and verify existing core, RGB decoder and native judge adapter.
- [x] Implement query-isolated resumable evaluator and source-group paired bootstrap with failing-first CPU tests.
- [x] Freeze study, baseline checkpoint identity, external runtime/weights inventory and four-GPU gate.
- [x] Add manifest-only entry and reserve/start policy dispatch.
- [x] Verify native processor/model in isolated environment, cache equivalence and batch1/8.
- [x] Complete focused regression and independent review; archive implementation report.

This task does not reserve or execute GPU experiments. A future separately reserved synthetic four-GPU inference/cache qualification must complete and be archived before scientific execution. Budget/retry accounting and the formal timing worker require qualification-stage implementation and review before enabling science_authorized. CPU tests cannot issue qualification receipts.
