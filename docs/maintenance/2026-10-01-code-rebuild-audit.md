# Reconstruction A audit

Source baseline: `ff8137c`. Worktree: `codex/retrieval-rebuild-a`. Scope: A only; no GPU scientific experiments or sealed test access.

Baseline: 166 CPU tests passed (artifacts, manager, risk metrics). No data plane copied into the worktree.

## Source disposition

| File | Disposition | Reason |
|---|---|---|
| `crf_ladder/__init__.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `crf_ladder/codec.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `crf_ladder/protocol.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `crf_ladder/statistics.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/crf_pairing.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/data_dataloaders.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/dataloader_msrvtt_retrieval.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/dataloader_msvd_retrieval.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/hard_negative_mapping.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/hard_negative_sampler.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/msrvtt_protocol.py` | migrate required subset, delete original | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/preprocess/preprocess_msvd.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/rawframes_util.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/rawvideo_util.py` | migrate required subset, delete original | New-system consumption determines retention; historical SHA preserves retired source |
| `dataloaders/tqfs_util.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `eval.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `experiment_artifacts.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `experiment_tracking.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `main_task_retrieval.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `metrics.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/crf_head.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/file_utils.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/modeling.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/modeling_mulit.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/module_clip.py` | migrate required subset, delete original | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/module_cross.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/module_mamba.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/mus_util.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/optimization.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/oq1_probabilistic_scoring.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/rala.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/spatial_enhancer.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/tokenization_clip.py` | migrate required subset, delete original | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/until_config.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `modules/until_module.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/contextual_retrieval_risk.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/csd_logvar.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/csd_logvar_protocol.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/pie_model.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/probemb.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/reliability_campaign.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/retrieval_risk.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/retrieval_risk_baselines.py` | migrate required subset, delete original | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/retrieval_risk_metrics.py` | migrate required subset, delete original | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/tensor_utils.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/uncertainty_module.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `prob_models/uncertainty_protocol.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `query_models/head.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `query_models/module_query.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `run_train_bg.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `run_train_msrvtt_bg.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `run_train_msrvtt_contextual_risk_bg.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `run_train_msrvtt_crf_bg.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `run_train_msrvtt_csd_logvar_bg.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `run_train_msrvtt_risk_sidecar_bg.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `run_train_msvd_bg.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/__init__.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/build_idf_weights.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/build_msrvtt_crf_ladder.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/build_msrvtt_hard_negatives.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/build_msrvtt_model_mined_hard_negatives.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/build_msrvtt_trusted_split.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/build_msvd_crf_ladder.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/build_val1k_partition.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/diagnose_crf_training_contract.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/diagnose_msrvtt_hard_negative_runtime.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/diagnose_msrvtt_p1_baseline.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/diagnose_msrvtt_validation_errors.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/diagnose_uncertainty_retrieval_correlation.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/dump_training_args_for_existing_ckpt.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/evaluate_crf_ladder.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/inventory_legacy_artifacts.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/manage_experiment_run.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/merge_msrvtt_model_mined_hard_negatives.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/migrate_critical_artifacts.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/oq1_arm_launch.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/oq1_stage0_launch.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/oq1_stage1_p0_metrics.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/oq1_tier0_correction_envelope.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/run_reliability_campaign.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `scripts/verify_frozen_retrieval_equivalence.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `smoke_test_msrvtt.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/_bench_rawvideo_decode.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/a3_four_rank_nccl_smoke.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_build_crf_ladder.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk_final_fixes.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk_integration.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk_launcher.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk_loss.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk_metrics.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk_protocol.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_contextual_retrieval_risk_training.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_crf_codec.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_crf_model_integration.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_crf_pairing.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_crf_protocol.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_crf_statistics.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_csd_logvar.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_csd_logvar_config.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_csd_logvar_integration.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_csd_logvar_launcher.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_csd_logvar_loss.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_csd_logvar_metrics.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_csd_logvar_protocol.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_diagnose_crf_contract.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_directional_eval.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_evaluate_crf_ladder.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_experiment_archive_e2e.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_experiment_artifact_docs.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_experiment_artifacts.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_experiment_tracking.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_legacy_artifact_migration.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_manage_experiment_run.py` | retain governance / decouple | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_reliability_campaign.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_retrieval_risk_baselines.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_retrieval_risk_features.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_retrieval_risk_integration.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_retrieval_risk_launcher.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_retrieval_risk_loss.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_retrieval_risk_metrics.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_run_train_msrvtt_bg.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_run_train_msrvtt_crf_bg.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `tests/test_uncertainty_protocol.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `train_msvd.sh` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/__init__.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/aleatoric.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/epistemic.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/losses.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/tests/test_aleatoric.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/tests/test_crf_ranking.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/tests/test_epistemic.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/tests/test_losses.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `uncertainty_modules/types.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |
| `util.py` | retire after necessary functions migrate | New-system consumption determines retention; historical SHA preserves retired source |

## Main checkout changes preserved

Existing tracked documentation edits are not included in this worktree. The following untracked launchers remain untouched pending hash-matched integration:

- `run_train_msvd_june_bg.sh`: SHA-256 `a29bf55ed8e1414142a711173b44ac08f779448aad2980c6b40516a4400ef5cc`
- `run_train_msvd_risk_sidecar_bg.sh`: SHA-256 `f38a722c7c651dd5fcb70b648c37a022fa899a750ba15b1951b0825975e4befa`

## Migration and verification record

- A1: named input/feature/target contracts; independent query/video counts; unknown=-1 preserved; no legacy configuration defaults.
- Relative and lazy imports observed: SpatialEnhancer→RALA, decoder→TQFS, diagnostics→MUS. Archive tests import the old training entrypoint; A6 must remove those dependencies before A7.
- Later task evidence will be appended here; no stage is claimed complete ahead of verification.

## A2 evaluation migration

Independent ranking, calibration, paired bootstrap and score proxies migrated. Unknown rows are excluded from Recall; undefined metrics serialize as null. Ties use score descending then candidate ID ascending; singleton entropy is zero and undefined margin is omitted. The transposed tie fixture expectation was corrected from [0,0] to [0,1] by manually sorting its second row. Initial missing-function run: 15 failures.
