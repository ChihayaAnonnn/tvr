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

## A3 CLIP migration

ViT/text primitives originate from OpenAI CLIP (MIT, source https://github.com/openai/CLIP), via baseline modules/module_clip.py and modules/tokenization_clip.py. ResNet, 3D patches, downloading, checkpoint wrappers and configurable low-precision LayerNorm are removed. Supported pretrained family: ViT-B/32; small ViT fixtures exercise the same calculation. The unchanged BPE SHA-256 is `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`. Local weights must match all keys and dimensions; only three original scalar metadata keys are checked and stripped. Initial test run: 7 missing-component failures; final: 7 passed. Python 3.10 lacks hashlib.file_digest; streaming SHA-256 fixes that compatibility failure.

## A4 input migration

Split schema and algorithms copied without changing frozen metadata; builder now imports the independent module. Decoder uses OpenCV frame indices (no FPS division), PIL bicubic Resize + CenterCrop + ToTensor and exact original CLIP mean/std. Missing/invalid inputs fail; undecodable requested frames stay zero with false masks. Synthetic short-video, zero-FPS, source-hash drift checks: 4 passed; builder --help passed. Versions: torch=2.1.2+cu121, torchvision=0.16.2+cu121, Pillow=11.0.0, opencv-python=4.11.0.86.

## A5 matching and gather

Masked maxima exclude invalid pairs before reduction and zero invalid reduced positions before weighting. Negative similarity remains -1 despite padding; chunking and suffix padding preserve scores; invalid/nonfinite rows fail; valid features and both learned weights backpropagate. Gather uses PyTorch autograd collectives and verifies equal shapes/dtypes across ranks. Two-rank CPU/Gloo gradient reference, metadata alignment and unequal-batch rejection pass. Initial missing components: 7 failures; final: 7 passed.

## A6 lifecycle migration and assertion destinations

RuntimeConfig binds RunContext without Namespace or model imports. Source, reservation, command hash, canonical private inputs, environment/precision, registry/study and validation/test authorization are checked before attaching. Stage A rejects any unconsumed model/loss/optimizer settings; it is not a scientific worker. Worker completion remains validating; controller owns terminal output hashes. Metrics are exclusive, bounded JSON and verify referenced arrays. CPU qualification exercises success, tampering and actual SIGTERM under a temporary handler; failures retain evidence. This does not qualify Stage 0 GPUs.

Manager retains generic reserve/verify/start/transition/finalize and historical manifests, but no longer generates retired A3 launch argv. Worker lifecycle, deferred hashing, sealed test, changed inputs and config-mismatch assertions in test_manage_experiment_run migrated to new RuntimeConfig/API. Old parser/A3 allowlist tests were removed because those launch interfaces are intentionally retired; identity and environment assertions now live in test_runtime_artifacts and the preserved generic archive suite. Model-specific CRF/risk/continuation manifest-constructor tests were removed with their constructors; generic strict JSON/source/batch telemetry tests remain in test_experiment_tracking. CPU archive end-to-end fixture now uses the real new prepare/start/publish/complete bridge; its >1 MiB reference and readonly legacy-directory assertions remain. Initial new tests: 9 failures; migrated archive suite: 141 passed. No frozen study/schema/registry or historical outputs changed.

## A7 actual removal inventory

Every executable below was checked against the A1 disposition table and removed using explicit git rm paths. scripts/__init__.py remains because the retained governance scripts are imported as a package by tests. Historical source remains at baseline ff8137c; no compatibility entrypoint is retained. Old uncertainty_modules/README.md remains only as historical documentation.

- `crf_ladder/__init__.py`
- `crf_ladder/codec.py`
- `crf_ladder/protocol.py`
- `crf_ladder/statistics.py`
- `dataloaders/crf_pairing.py`
- `dataloaders/data_dataloaders.py`
- `dataloaders/dataloader_msrvtt_retrieval.py`
- `dataloaders/dataloader_msvd_retrieval.py`
- `dataloaders/hard_negative_mapping.py`
- `dataloaders/hard_negative_sampler.py`
- `dataloaders/msrvtt_protocol.py`
- `dataloaders/preprocess/preprocess_msvd.py`
- `dataloaders/rawframes_util.py`
- `dataloaders/rawvideo_util.py`
- `dataloaders/tqfs_util.py`
- `eval.sh`
- `main_task_retrieval.py`
- `metrics.py`
- `modules/crf_head.py`
- `modules/file_utils.py`
- `modules/modeling.py`
- `modules/modeling_mulit.py`
- `modules/module_clip.py`
- `modules/module_cross.py`
- `modules/module_mamba.py`
- `modules/mus_util.py`
- `modules/optimization.py`
- `modules/oq1_probabilistic_scoring.py`
- `modules/rala.py`
- `modules/spatial_enhancer.py`
- `modules/tokenization_clip.py`
- `modules/until_config.py`
- `modules/until_module.py`
- `prob_models/contextual_retrieval_risk.py`
- `prob_models/csd_logvar.py`
- `prob_models/csd_logvar_protocol.py`
- `prob_models/pie_model.py`
- `prob_models/probemb.py`
- `prob_models/reliability_campaign.py`
- `prob_models/retrieval_risk.py`
- `prob_models/retrieval_risk_baselines.py`
- `prob_models/retrieval_risk_metrics.py`
- `prob_models/tensor_utils.py`
- `prob_models/uncertainty_module.py`
- `prob_models/uncertainty_protocol.py`
- `query_models/head.py`
- `query_models/module_query.py`
- `run_train_bg.sh`
- `run_train_msrvtt_bg.sh`
- `run_train_msrvtt_contextual_risk_bg.sh`
- `run_train_msrvtt_crf_bg.sh`
- `run_train_msrvtt_csd_logvar_bg.sh`
- `run_train_msrvtt_risk_sidecar_bg.sh`
- `run_train_msvd_bg.sh`
- `scripts/build_idf_weights.py`
- `scripts/build_msrvtt_crf_ladder.py`
- `scripts/build_msrvtt_hard_negatives.py`
- `scripts/build_msrvtt_model_mined_hard_negatives.py`
- `scripts/build_msvd_crf_ladder.py`
- `scripts/build_val1k_partition.py`
- `scripts/diagnose_crf_training_contract.py`
- `scripts/diagnose_msrvtt_hard_negative_runtime.py`
- `scripts/diagnose_msrvtt_p1_baseline.py`
- `scripts/diagnose_msrvtt_validation_errors.py`
- `scripts/diagnose_uncertainty_retrieval_correlation.py`
- `scripts/dump_training_args_for_existing_ckpt.py`
- `scripts/evaluate_crf_ladder.py`
- `scripts/merge_msrvtt_model_mined_hard_negatives.py`
- `scripts/oq1_arm_launch.sh`
- `scripts/oq1_stage0_launch.sh`
- `scripts/oq1_stage1_p0_metrics.py`
- `scripts/oq1_tier0_correction_envelope.py`
- `scripts/run_reliability_campaign.py`
- `scripts/verify_frozen_retrieval_equivalence.py`
- `smoke_test_msrvtt.sh`
- `tests/_bench_rawvideo_decode.py`
- `tests/a3_four_rank_nccl_smoke.py`
- `tests/test_build_crf_ladder.py`
- `tests/test_contextual_retrieval_risk.py`
- `tests/test_contextual_retrieval_risk_final_fixes.py`
- `tests/test_contextual_retrieval_risk_integration.py`
- `tests/test_contextual_retrieval_risk_launcher.py`
- `tests/test_contextual_retrieval_risk_loss.py`
- `tests/test_contextual_retrieval_risk_metrics.py`
- `tests/test_contextual_retrieval_risk_protocol.py`
- `tests/test_contextual_retrieval_risk_training.py`
- `tests/test_crf_codec.py`
- `tests/test_crf_model_integration.py`
- `tests/test_crf_pairing.py`
- `tests/test_crf_protocol.py`
- `tests/test_crf_statistics.py`
- `tests/test_csd_logvar.py`
- `tests/test_csd_logvar_config.py`
- `tests/test_csd_logvar_integration.py`
- `tests/test_csd_logvar_launcher.py`
- `tests/test_csd_logvar_loss.py`
- `tests/test_csd_logvar_metrics.py`
- `tests/test_csd_logvar_protocol.py`
- `tests/test_diagnose_crf_contract.py`
- `tests/test_directional_eval.py`
- `tests/test_evaluate_crf_ladder.py`
- `tests/test_reliability_campaign.py`
- `tests/test_retrieval_risk_baselines.py`
- `tests/test_retrieval_risk_features.py`
- `tests/test_retrieval_risk_integration.py`
- `tests/test_retrieval_risk_launcher.py`
- `tests/test_retrieval_risk_loss.py`
- `tests/test_retrieval_risk_metrics.py`
- `tests/test_run_train_msrvtt_bg.py`
- `tests/test_run_train_msrvtt_crf_bg.py`
- `tests/test_uncertainty_protocol.py`
- `train_msvd.sh`
- `uncertainty_modules/__init__.py`
- `uncertainty_modules/aleatoric.py`
- `uncertainty_modules/epistemic.py`
- `uncertainty_modules/losses.py`
- `uncertainty_modules/tests/test_aleatoric.py`
- `uncertainty_modules/tests/test_crf_ranking.py`
- `uncertainty_modules/tests/test_epistemic.py`
- `uncertainty_modules/tests/test_losses.py`
- `uncertainty_modules/types.py`
- `util.py`
- `modules/bpe_simple_vocab_16e6.txt.gz`
- `modules/cross-base/cross_config.json`
- `dataloaders/VOCAB.json`

### A7 protection and validation

Removed 125 explicitly listed old files. Remaining executable closure consists of the new package plus four governance/data utilities and generic provenance. Dependencies removed: tqdm, requests, boto3, einops, scikit-learn, decord, PyAV; scipy is now explicit for evaluation. Structural negative injection and CLI/import closure: 12 passed. Full retained CPU tests: 237 passed (exit 0); Ruff and git diff --check passed. The exact test count is recorded by the task completion command below if pytest discovery changes. No GPU qualification/scientific run, dataset/test evaluation, phase reservation or real checkpoint loading occurred.

Baseline and current protected Git subtree identities are identical, and the working diff for these paths is empty:

```text
040000 tree 115c7d6253a9ffc768f1e0b570caa45c1b8ad898	data/generated
040000 tree 7969c1429e8c0e253dbeb48aad3676e03100605f	dataloaders/splits
040000 tree 76439a72dcd02b192a5c5338c1c178c9c2f19651	experiments
040000 tree 55e78c7bb49c60bc7ba36b797769bbf354a2483d	related_work
040000 tree 020167bf57e1c6d753c8f0a553f3360bcf457bf4	research
```

Original main checkout and its untracked launchers are untouched; their approved deletion is reserved for hash-checked integration. No logs, ckpts or artifacts directories were deleted, moved or rewritten. Historical array/weight metadata is preserved without re-reading large weights.
