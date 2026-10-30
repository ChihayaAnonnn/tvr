# HVP-Net evaluation-only repair

The user explicitly authorized removing candidate paired-caption inputs and one reevaluation of a trained checkpoint. Epoch1 (checkpoint_epoch_0.pth) was fixed in the response before any repaired score is computed; no best.pth selection, no epoch sweep, no retraining. The running training source is untouched in the original workspace. Training retains candidate-caption weighting; only inference uses query word weights. Scores measure the existing model under repaired inference, not training under a repaired objective.

Use the same official JSFUSION1000 queries/videos, frame/text preprocessing, B32 precision and chunk settings. Verify all video/CSV/checkpoint bytes. No candidate-caption weights may be accepted by the scoring API. Store the full score matrix through an external hashed NPY reference. Preserve historical runs; no overwrite.

Two CUDA ranks extract and score, sharing GPU0/3 with our own existing training only if preflight memory permits. Each evaluation rank is limited to16GiB; no other process is stopped. User's continuing dual-card HVP authorization is scoped to this evaluation; a fresh production random/selected checkpoint inference qualifier must pass first. All outputs use SchemeA reservation and terminal archives in this independent study. A published completed epoch checkpoint from the still-running parent may be read once frozen by file SHA256 and verified epoch/step, without waiting for or modifying that parent. Final metrics must distinguish synthetic qualification from real test results.

Postprocessing correction: upstream `compute_metrics` counts every tie with the
positive score as a separate query. A CPU-only child may summarize the single
already archived 1000x1000 matrix with exactly one rank per query, using number
of strictly greater scores (optimistic ties). This fixes duplicated counting,
does not access videos/checkpoints or execute the model, and does not authorize
another inference run or selection. Preserve the original matrix and raw report.
