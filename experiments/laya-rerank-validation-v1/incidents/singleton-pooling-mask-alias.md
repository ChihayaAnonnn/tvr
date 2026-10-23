# Diagnosed performance failure: singleton pooling mask alias

Failed immutable parent: msrvtt-efficiency-laya-performance-seed713-20261023T092517Z-ea3f763f7930. Failure class: singleton_pooling_mask_alias. Warmup Top10 assertion stopped before any timing report.

Original upstream `_mean_pooling_for_similarity_sequence` converts an already-FP32 mask without copying and zeroes BOS in-place. A 32-of34 contiguous slice aliases at batch1; at archived batch32 it is copied. WTI therefore sees different masks. Independent audit identified and confirmed this alias mechanism. Upstream file remains unchanged.

Read-only CPU causal diagnostic with strict trained epoch05 checkpoint, fixed query video9487:52690 (index9510), and hash-verified prepared video cache: unchanged singleton maximum score error vs archive4.1860151290893555; original encoding128/scoring32 error0.00001621246337890625; duplicating query to scoring2 error0.00001621246337890625; protecting pooling input with mask.clone at actual batch1 error0.00001621246337890625, identical archived Top10. Clone singleton vs diagnostic duplicated2 maximum5.7220458984375e-06. Duplicate query was diagnostic only, never the measurement implementation. CPU diagnostics are not GPU performance measurements.

Correction is a local measurement adapter which clones only the helper temporary mask, preserving original archived batch32 WTI semantics at query batch1. No upstream baseline source, checkpoint, preprocessing, frame count, query batch, score fusion, or scientific accuracy result changes. Strict archived baseline/enhanced Top10 checks remain enabled. Native synthetic qualification now additionally compares original multiquery scores with protected singleton scores at1e-5 and exact rankings. CPU mask regression covers batches1,8,32. All29 narrow CPU tests pass (two opt-in native tests skipped).

A new source-matched synthetic qualification and archive is required before one diagnosed performance successor. Accuracy is not repeated or tuned. Original failed attempt remains immutable and included in cumulative scientific budget.
