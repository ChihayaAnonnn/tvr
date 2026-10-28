# CRF Degradation Ladder Implementation Plan

## Execution Summary

- **Execution mode:** inline sequential development; do not dispatch subagents. All work runs in the current session; no `spawn_agent`, no worktrees, no parallel task delegation.
- **Goal:** teach `u_frame` / `u_video` (aleatoric) to detect incremental x264 compression damage, and validate with in-domain unseen CRFs, a zero-parameter baseline, and MSVD zero-shot.
- **Architecture:** offline PyAV builds a per-file manifest CRF ladder; MSRVTT dataloader uses a batch-aware sampler so every local batch carries a fixed number of order pairs sharing clean frame indices and transforms; only four ranking losses update `uncertainty_head`; diagnostics use single-pass inference per CRF level.
- **Five sequential stages (must complete A → E in order):**
  - **A.** Frozen protocol + deterministic x264 codec (Tasks 1–2)
  - **B.** Ladder builders + batch-aware sampler / synchronized collate (Tasks 3–4)
  - **C.** Ranking loss + model wiring + training config/optimizer preflight (Tasks 5–7)
  - **D.** Statistics + CRF-dev selection + CRF-audit / MSVD diagnostics (Tasks 8–10)
  - **E.** MSVD subset, standalone runner, README, end-to-end acceptance (Task 11)
- **Key acceptance gates:** CRF-audit `u_video` three-pair accuracy ≥ 0.90; CRF50 vs CRF42 accuracy ≥ 0.80; challenge-pair `(u − Laplacian)` bootstrap CI low > 0; MSVD three-pair accuracy ≥ 0.80 with CI low > 0.5. Failure = falsified; no rescue.

---

## Global Constraints (single source of truth)

Referenced elsewhere as **[Gn]**. Do **not** repeat these in stage bodies; cite the number.

1. Serial execution only. One task in progress at a time. No `spawn_agent` / `subagent-driven-development` / cross-agent delegation.
2. Baseline must contain commit `1e11659`. Before starting, run `git cat-file -e 1e11659^{commit}` and `git merge-base --is-ancestor 1e11659 HEAD`. If either check fails, **stop and ask the user which branch to work on**; the plan is not authorized to switch branches, checkout, reset, or otherwise mutate git state on the user's behalf. (Do not encode any specific "current HEAD" assertion here — HEAD drifts; the check commands are the source of truth.)
3. Supervised quantities are `u_frame` / `u_video` only. `propagated_variance`, `global_variance`, `embedding.variance` are unsupervised by this plan; never claim probabilistic variance is calibrated.
4. Clean source = the training pipeline's `msrvtt_224_12fps/*.mp4`. No extra transcode on the clean path.
5. Training CRFs = `{30, 38, 46}`; in-domain evaluation CRFs = `{34, 36, 42, 44, 50}`. All degraded levels share one explicit x264 recipe (see [Appendix A](#appendix-a-frozen-x264-recipe)).
6. Trusted-v1 split: build 8500 `train_video_ids` and 500 `val_video_ids`. **Never** build / train / tune / evaluate on the 1000 test videos.
7. 500 val videos split in manifest order: first 250 = **CRF-dev**, last 250 = **CRF-audit**. CRF-audit runs **once** on the final selected checkpoint.
8. First-degradation CRF sampled from `{30,38,46}` such that within each **global** batch (`local_batch_size * world_size`) the three rank counts differ by at most 1 (near-uniform, deterministic under `(seed, epoch)`). Order-pair count per local batch = `max(1, round(B_local * p))`, `p` default `0.20`, tunable only on CRF-dev within `[0.20, 0.25]`. `drop_last=True`.
9. Clean, primary, and secondary paths share frame indices, resize/crop/normalize, `frame_order`, TQFS decisions, and `relevance_weights(clean).detach()`. Missing / corrupt / metadata-mismatched files must raise; **no fallback to clean**.
10. `base_margin = 0.05`. Four lambdas default `1.0`. Order loss is normalized by actual pair count (not `B_local`). Total is multiplied by the existing `--aleatoric_ranking_weight`.
11. All CRF arms: `--aleatoric_prior_weight=0`, `pooling_temperature=0`, `global_uncertainty_head.requires_grad_(False)` and excluded from optimizer.
12. Remove `degrade_video_pixels` and its five flags. Manual pixel degradation and CRF degradation must not coexist.
13. Standard retrieval eval keeps single-pass behavior. CRF diagnostics also run single-pass per level; no paired forward in standard eval.
14. Main gate: CRF-audit `u_video` three-pair accuracy ≥ 0.90; CRF50 vs CRF42 accuracy ≥ 0.80 (else spec's "extrapolation unreadable" wording).
15. Falsification gate: on challenge pairs `(34,36)` and `(42,44)`, video-bootstrap `acc_u − acc_Laplacian` 95% CI **low > 0**.
16. MSVD: fixed ≥300-video test subset in original resolution/fps. Zero training / zero fine-tuning. Three-pair accuracy ≥ 0.80 and 95% CI low > 0.5.
17. TDD for every task: write the exact failing test, then the minimum implementation. One commit per task.

---

## Serial Execution Protocol

1. Load this plan through `superpowers:executing-plans`; maintain checkboxes in the current session only.
2. Execute Task 1 → Task 11 strictly in order. Later tasks may only consume interfaces produced by earlier committed tasks.
3. Per task checkpoint: modified files, failing-test evidence, passing-test evidence, commit hash, residual risk.
4. On failure, stay in the current task and debug via `superpowers:systematic-debugging`. Do not skip or defer.
5. CRF-audit and MSVD evaluations are **not** ordinary implementation steps; they run only after all code in Task 11 is verified **and** the user explicitly approves running the experiment.

---

## File Responsibilities

| File | Role |
|---|---|
| `requirements.txt` | Pin PyAV runtime. |
| `crf_ladder/protocol.py` | CRF levels, x264 recipe, split, manifest schema, path convention. |
| `crf_ladder/codec.py` | PyAV preflight, explicit transcode, probe, hash, atomic write, manifest row builder. |
| `crf_ladder/statistics.py` | Paired accuracy, video bootstrap CI, Spearman, saturation, Laplacian score. |
| `crf_ladder/__init__.py` | Public stable exports. |
| `scripts/build_msrvtt_crf_ladder.py` | MSRVTT 8500/500 batch build with resume, rebuild-on-hash-mismatch, ETA. |
| `scripts/build_msvd_crf_ladder.py` | MSVD fixed subset original-geometry ladder. |
| `dataloaders/crf_pairing.py` | Distributed batch-aware level sampler + three-path shared transform. |
| `dataloaders/dataloader_msrvtt_retrieval.py` | Training sample = clean + primary degraded + compact order sub-batch. |
| `dataloaders/data_dataloaders.py` | CRF sampler/collate wiring. |
| `uncertainty_modules/aleatoric.py` | `pooling_temperature == 0` numerical short-circuit + shared-weight aggregation. |
| `uncertainty_modules/losses.py` / `types.py` | Four frozen ranking losses + named result. |
| `modules/modeling.py` | Frozen global head, paired no-grad visual forward, single-pass risk scorer, loss wiring. |
| `main_task_retrieval.py` | CRF-fixed config, batch unpacking, logging, optimizer / gradient constraints, CRF-dev checkpoint selection. |
| `scripts/evaluate_crf_ladder.py` | MSRVTT dev/audit, baseline challenge, MSVD cross-domain, JSON + Markdown report. |
| `scripts/diagnose_crf_training_contract.py` | spec §3.5 ten-item diagnostic table + throughput. |
| `run_train_msrvtt_crf_bg.sh` | Standalone CRF experiment launcher; does not modify baseline hygiene runner. |
| `tests/test_crf_*.py`, `uncertainty_modules/tests/test_crf_ranking.py`, `tests/test_experiment_tracking.py`, `tests/test_evaluate_crf_ladder.py`, `tests/test_diagnose_crf_contract.py`, `tests/test_run_train_msrvtt_crf_bg.py` | Task-scoped tests. |

---

## Stage A — Frozen Protocol + Deterministic x264 Codec

**Purpose:** freeze the CRF/x264 contract and provide an atomic PyAV transcoder with a preflight and a JSONL manifest.
**Preconditions:** [G2] satisfied.
**Applies:** [G4], [G5], [G6], [G7], [Appendix A], [Appendix B].

### Task 1 — Protocol, split, dependency

- **Modify:** `requirements.txt`
- **Create:** `crf_ladder/protocol.py`, `crf_ladder/__init__.py`, `dataloaders/splits/crf_dev_audit_seed0.json`, `tests/test_crf_protocol.py`
- **Interfaces produced:**
  - `TRAIN_CRFS = (30, 38, 46)`, `EVAL_CRFS = (34, 36, 42, 44, 50)`, `DEV_SIZE = AUDIT_SIZE = 250`.
  - `@dataclass(frozen=True) X264Recipe` with the exact fields in [Appendix A]; `X264_RECIPE = X264Recipe()`.
  - `split_crf_dev_audit(val_video_ids) -> (dev, audit)` — order-preserving slice `[:250]`, `[250:500]`. Does **not** mutate the trusted-v1 manifest schema.
  - `dataloaders/splits/crf_dev_audit_seed0.json` — **checked-in artifact** recording the frozen split for audit visibility. Schema: `{ "source_manifest": "dataloaders/splits/msrvtt_trusted_v1_seed0.json", "source_manifest_sha256": "<hex>", "crf_dev_video_ids": [...250 ids...], "crf_audit_video_ids": [...250 ids...] }`. Regenerated only when the trusted-v1 manifest itself changes.
  - `crf_output_path(root, crf, video_id) -> Path` equals `Path(root)/f"crf{crf}"/f"{video_id}.mp4"`.

Steps:

1. Write failing test `tests/test_crf_protocol.py` anchoring the four contracts below:

   ```python
   def test_frozen_crf_contract():
       assert TRAIN_CRFS == (30, 38, 46)
       assert EVAL_CRFS == (34, 36, 42, 44, 50)
       assert asdict(X264_RECIPE) == { ... exact fields from Appendix A ... }

   def test_dev_audit_preserves_manifest_order_and_is_disjoint():
       ids = [f"video{i}" for i in range(500)]
       dev, audit = split_crf_dev_audit(ids)
       assert dev == ids[:250] and audit == ids[250:]
       assert set(dev).isdisjoint(audit)

   def test_output_path_is_canonical(tmp_path):
       assert crf_output_path(tmp_path, 34, "video12") == tmp_path / "crf34" / "video12.mp4"

   def test_versioned_manifest_contains_frozen_crf_partition():
       payload = json.loads(Path("dataloaders/splits/msrvtt_trusted_v1_seed0.json").read_text())
       dev, audit = split_crf_dev_audit(payload["val_video_ids"])
       assert dev == payload["val_video_ids"][:250]
       assert audit == payload["val_video_ids"][250:]

   def test_checked_in_crf_split_matches_manifest_and_records_source_sha256():
       trusted_path = Path("dataloaders/splits/msrvtt_trusted_v1_seed0.json")
       split = json.loads(Path("dataloaders/splits/crf_dev_audit_seed0.json").read_text())
       assert split["source_manifest"] == "dataloaders/splits/msrvtt_trusted_v1_seed0.json"
       assert split["source_manifest_sha256"] == hashlib.sha256(trusted_path.read_bytes()).hexdigest()
       dev, audit = split_crf_dev_audit(json.loads(trusted_path.read_text())["val_video_ids"])
       assert split["crf_dev_video_ids"]   == dev
       assert split["crf_audit_video_ids"] == audit
   ```

2. Run `python -m pytest tests/test_crf_protocol.py -q` → **FAIL** with `ModuleNotFoundError: crf_ladder`.
3. Implement `crf_ladder/protocol.py` with the interfaces above; export `EVAL_CRFS`, `TRAIN_CRFS`, `X264_RECIPE`, `crf_output_path` from `__init__.py`. CRF-dev/audit uses live slicing of `val_video_ids`; **do not** add a new manifest schema.
4. Generate `dataloaders/splits/crf_dev_audit_seed0.json` from the trusted-v1 manifest with its `source_manifest_sha256`, check it into git, and re-verify with `test_checked_in_crf_split_matches_manifest_and_records_source_sha256`.
5. Add `av==17.1.0` to the video dependency block of `requirements.txt`.
6. Run `python -m pytest tests/test_crf_protocol.py -q` → **PASS**.
7. Commit: `git add requirements.txt crf_ladder/protocol.py crf_ladder/__init__.py dataloaders/splits/crf_dev_audit_seed0.json tests/test_crf_protocol.py && git commit -m "feat: freeze CRF ladder protocol"`.

### Task 2 — PyAV atomic transcode + preflight + manifest

- **Create:** `crf_ladder/codec.py`, `tests/test_crf_codec.py`
- **Interfaces produced:**
  - `sha256_file(path) -> str`.
  - `probe_video(path) -> VideoMetadata` (fields: `width, height, fps_num, fps_den, time_base_num, time_base_den, frame_count, duration_seconds`). `frame_count` must be counted by decoding, never estimated from container.
  - `encoder_options(crf) -> dict[str, str]` — produces `crf`, `preset`, and one `x264-params` string containing exactly the [Appendix A] key/value list.
  - `transcode_atomic(source, destination, crf) -> Path` — writes to `NamedTemporaryFile(dir=destination.parent, suffix=".tmp.mp4")`, runs `_transcode()`, calls `assert_transcode_invariants(source, tmp)`, then `os.replace(tmp, destination)`. On any exception `destination` bytes are unchanged and `.tmp.mp4` residue is unlinked in a `finally` block.
  - `assert_transcode_invariants(source, output)` — raises `RuntimeError` including source path if `width/height/fps_num/fps_den/frame_count` differ, or if `abs(duration_source − duration_output) > 0.5 * fps_den/fps_num`.
  - `@dataclass(frozen=True) ManifestRow` — full field list in [Appendix B]. `bitrate_bps = round(size_bytes * 8 / duration_seconds)` computed from `probe_video(output)` and file size; **never** substitute recipe constants.
  - `append_manifest_row(path, row)` — single-writer, single-line append using `json.dumps(asdict(row), sort_keys=True)`, `flush()` + `fsync()`. **Not** safe for concurrent processes; the builder (Task 3) is responsible for enforcing that only the parent process ever calls this.
  - `build_manifest_row(job, source_path, source_sha256, output_path, versions, success, error=None) -> ManifestRow` — pure function returning a `ManifestRow` from a completed (or failed) job. It performs no I/O beyond `probe_video(output)` + `sha256_file(output)` on `success=True`; on `success=False` it fills output-derived fields with `None`/`0`. Workers call this and return the row to the parent; workers **never** open the manifest file.
  - `should_rebuild(output, row, source_sha256) -> bool` — True when output missing, no row, `row.success is False`, `row.source_sha256 != source_sha256`, or `sha256_file(output) != row.output_sha256`.
  - `run_preflight(source, destination, frame_limit) -> PreflightResult` — encodes first `frame_limit` frames, re-decodes, returns `encoder="libx264"`, source/output geometry and fps, `decoded_frames`.

Steps:

1. Write failing tests in `tests/test_crf_codec.py`:

   ```python
   def test_manifest_append_is_one_complete_json_object(tmp_path):
       path = tmp_path / "manifest.jsonl"
       append_manifest_row(path, _row())
       assert json.loads(path.read_text().strip())["video_id"] == "video1"

   def test_resume_rebuilds_missing_failed_or_hash_mismatched_output(tmp_path):
       output = tmp_path / "video1.mp4"
       assert should_rebuild(output, None, "a")
       output.write_bytes(b"payload")
       assert should_rebuild(output, _row(success=False), "a")
       assert should_rebuild(output, _row(source_hash="wrong"), "a")

   def test_transcode_failure_never_replaces_destination(tmp_path, monkeypatch):
       source = tmp_path / "source.mp4";      source.write_bytes(b"source")
       destination = tmp_path / "out.mp4";    destination.write_bytes(b"known-good")
       monkeypatch.setattr(codec, "_transcode",
                           lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
       with pytest.raises(RuntimeError, match="boom"):
           codec.transcode_atomic(source, destination, 30)
       assert destination.read_bytes() == b"known-good"
       assert not list(tmp_path.glob("*.tmp.mp4"))

   def test_real_libx264_preflight_when_video_is_configured(tmp_path):
       source = os.environ.get("CRF_PREFLIGHT_VIDEO")
       if not source:
           pytest.skip("set CRF_PREFLIGHT_VIDEO for runtime codec capability check")
       result = run_preflight(Path(source), tmp_path / "preflight.mp4", frame_limit=16)
       assert result.encoder == "libx264"
       assert result.decoded_frames == 16
       assert (result.source_width, result.source_height, result.source_fps) == \
              (result.output_width, result.output_height, result.output_fps)
   ```
2. Run `python -m pytest tests/test_crf_codec.py -q` → **FAIL** (`No module named crf_ladder.codec`).
3. Implement `crf_ladder/codec.py` per interfaces. `_transcode` opens the source once, adds one `libx264` output stream with `encoder_options(crf)`, sets `pix_fmt`, `time_base=Fraction(rate.denominator, rate.numerator)`, `codec_context.max_b_frames = X264_RECIPE.b_frames`, encodes and muxes every decoded frame (setting `frame.pts=None`), then flushes the encoder.
4. Add the real-libx264 preflight test as above.
5. Run:
   - `python -m pytest tests/test_crf_codec.py -q` → unit **PASS**; preflight skipped without env var.
   - `CRF_PREFLIGHT_VIDEO=/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps/video0.mp4 python -m pytest tests/test_crf_codec.py::test_real_libx264_preflight_when_video_is_configured -q` → **1 passed**. If it fails, stop; do not proceed to bulk build.
6. Commit: `git add crf_ladder/codec.py tests/test_crf_codec.py && git commit -m "feat: add deterministic x264 ladder codec"`.

### Stage A review checkpoint

- CRF/x264 contract frozen; `TRAIN_CRFS`, `EVAL_CRFS`, `X264_RECIPE` immutable.
- CRF-dev/audit slicing verified against trusted-v1 manifest, no schema change.
- `transcode_atomic` guarantees atomic replacement, no partial writes.
- Real libx264 encode/decode round-trip succeeds on one video.
- Diff includes only the files listed in Tasks 1–2. Report to user and wait for approval before Stage B.

---

## Stage B — Ladder Builders + Batch-Aware Sampler / Synchronized Collate

**Purpose:** produce `data/generated/msrvtt_crf/crfNN/videoXXXX.mp4` + JSONL manifests for MSRVTT and MSVD, then deliver a distributed sampler whose every local batch carries a fixed number of order pairs sharing frame indices with clean.
**Preconditions:** Stage A committed.
**Applies:** [G6], [G8], [G9], [G13], [Appendix B].

### Task 3 — MSRVTT / MSVD ladder builders

- **Create:** `scripts/build_msrvtt_crf_ladder.py`, `scripts/build_msvd_crf_ladder.py`, `tests/test_build_crf_ladder.py`
- **Interfaces produced:**
  - `@dataclass(frozen=True) BuildJob(video_id: str, crf: int)`.
  - `build_jobs(manifest: dict) -> list[BuildJob]` — `TRAIN_CRFS` on `train_video_ids`, `EVAL_CRFS` on `val_video_ids`, **never** on `test_video_ids`.
  - MSRVTT CLI defaults: `--source-root /data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps`, `--manifest dataloaders/splits/msrvtt_trusted_v1_seed0.json`, `--output-root data/generated/msrvtt_crf`, `--workers 8`, plus `--check-only` and `--limit N` (for smoke).
  - MSVD CLI reads `test_list.txt` order; `--subset-size 300`, writes `subset.json` (id, source sha256, selection rule). On rerun the existing `subset.json` is **reused and re-validated**, never regenerated. Only `{34, 42}` are built. Encoder uses per-file `probe_video(source)` geometry/fps.
  - **Concurrency model (single-writer, multi-worker):**
    - The parent process is the **sole** writer of every per-level `manifest.jsonl`. Workers dispatched via `multiprocessing.Pool` / `ProcessPoolExecutor` receive a `BuildJob`, perform `transcode_atomic` + `build_manifest_row`, and return the `ManifestRow` (plus the source's sha256) as their result value. Workers never open, create, or lock the manifest file.
    - The parent iterates completed futures in submission order or with `imap_unordered` + a small in-memory reorder buffer keyed on `(video_id, crf)` so that the on-disk JSONL row order is deterministic and reproducible across reruns with the same worker count.
    - Failed jobs return a `ManifestRow` with `success=False`; the parent appends it and continues. Any exception raised inside a worker is re-raised in the parent after the current pool batch drains, so no half-written row can be interleaved.
    - Resume logic runs entirely in the parent: on startup it loads every existing per-level manifest into a `dict[(video_id, crf), ManifestRow]` (taking the **last** row per key), then for each job decides `should_rebuild` based on that in-memory map plus a freshly computed `sha256_file(source)`. Skipped jobs are not dispatched to workers at all.
  - On first source (parent), run Task-2 preflight before any worker is spawned; fail-close if the preflight fails.
  - Progress line every 100 completed jobs: `done/total files/s ETA`. `--check-only` runs entirely in the parent, verifies all files + hashes + metadata against the loaded manifest map, and never writes.

Steps:

1. Failing tests in `tests/test_build_crf_ladder.py`:

   ```python
   def test_msrvtt_jobs_use_only_train_and_val_ids():
       manifest = {
           "train_video_ids": ["video1", "video2"],
           "val_video_ids":   ["video3"],
           "test_video_ids":  ["video9999"],
       }
       jobs = build_jobs(manifest)
       assert {(j.video_id, j.crf) for j in jobs} == {
           ("video1", 30), ("video1", 38), ("video1", 46),
           ("video2", 30), ("video2", 38), ("video2", 46),
           ("video3", 34), ("video3", 36), ("video3", 42), ("video3", 44), ("video3", 50),
       }
       assert all(j.video_id != "video9999" for j in jobs)

   def test_workers_never_open_the_manifest_file(monkeypatch, tmp_path):
       # Fake transcoder + a spy on codec.append_manifest_row inside each worker.
       # Assert that the spy is called exactly once per row, in the parent, and never
       # from a worker PID different from the parent PID.
       parent_pid = os.getpid()
       writer_pids = set()
       def spy(path, row):
           writer_pids.add(os.getpid())
           return _real_append(path, row)
       monkeypatch.setattr(build_msrvtt_crf_ladder.codec, "append_manifest_row", spy)
       build_msrvtt_crf_ladder.main(argv=[
           "--source-root", str(_fake_source_root(tmp_path)),
           "--manifest",    str(_fake_manifest(tmp_path)),
           "--output-root", str(tmp_path / "out"),
           "--workers", "4", "--limit", "8",
       ])
       assert writer_pids == {parent_pid}

   def test_multi_worker_jsonl_has_one_row_per_completed_job_and_is_deterministic(tmp_path):
       # Run the builder twice with --workers 8 --limit 24 against a fake source root.
       # Assert per-level manifest: exact row count matches successful jobs; every line
       # is a complete JSON object; the two runs produce byte-identical manifest files.
       out_a = _run(tmp_path / "a", workers=8, limit=24)
       out_b = _run(tmp_path / "b", workers=8, limit=24)
       for crf in TRAIN_CRFS + EVAL_CRFS:
           rows_a = _load_jsonl(out_a / f"crf{crf}" / "manifest.jsonl")
           rows_b = _load_jsonl(out_b / f"crf{crf}" / "manifest.jsonl")
           assert len(rows_a) == _expected_success_count(crf, limit=24)
           assert rows_a == rows_b
           for line in (out_a / f"crf{crf}" / "manifest.jsonl").read_text().splitlines():
               json.loads(line)   # each line parses standalone

   def test_resume_after_partial_run_never_duplicates_rows(tmp_path):
       # Run once with --limit 8 --workers 4; then again with --limit 16 --workers 4.
       # The second run must (a) not re-transcode the first 8, (b) append only the
       # incremental 8 rows per level, (c) leave in-memory should_rebuild True for
       # exactly the 8 jobs that were dispatched.
       ...
   ```
2. Run `python -m pytest tests/test_build_crf_ladder.py -q` → **FAIL** (`No module named scripts.build_msrvtt_crf_ladder`).
3. Implement MSRVTT builder per interfaces.
4. Implement MSVD builder per interfaces; must not reference MSRVTT 298×224 / 12 fps constants.
5. Run:
   - `python -m pytest tests/test_build_crf_ladder.py tests/test_crf_codec.py -q` → **PASS**.
   - `python scripts/build_msrvtt_crf_ladder.py --source-root /data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps --manifest dataloaders/splits/msrvtt_trusted_v1_seed0.json --output-root data/generated/msrvtt_crf --limit 2` → 2 jobs succeed, JSONL parseable; rerun reports `skipped=2 rebuilt=0`.
6. Commit: `git add scripts/build_msrvtt_crf_ladder.py scripts/build_msvd_crf_ladder.py tests/test_build_crf_ladder.py && git commit -m "feat: build resumable CRF video ladders"`.

### Task 4 — Distributed batch-aware sampler + synchronized paired decode

- **Create:** `dataloaders/crf_pairing.py`, `tests/test_crf_pairing.py`
- **Modify:** `dataloaders/dataloader_msrvtt_retrieval.py`, `dataloaders/data_dataloaders.py`
- **Interfaces produced:**
  - `@dataclass(frozen=True) CRFBatchIndex(index: int, primary_rank: int, secondary_rank: int | None)` where rank ∈ `{1,2,3}` maps to `{30:1, 38:2, 46:3}` (ordinal, **never** raw CRF into margin).
  - `class CRFDistributedBatchSampler(dataset_size, local_batch_size, rank, world_size, seed, order_fraction)` with `set_epoch(epoch)`, `__iter__`, `__len__`. Iteration (fully deterministic given `(seed, epoch)`; all worker ranks share the same base generator):
    - Seed `torch.Generator().manual_seed(seed + epoch)`, `randperm(dataset_size)`.
    - Truncate to a multiple of `G = local_batch_size * world_size`.
    - Per global batch: build a `primary_rank` vector of length `G` where each of `{1,2,3}` appears exactly `G // 3` times, with the remaining `G % 3` slots filled by a deterministic rotation of `{1,2,3}` seeded by `(seed, epoch, batch_ordinal)`, then shuffled with the same generator. This guarantees the three rank counts in every global batch differ by at most 1.
    - Slice the local rank window `[rank*local_batch_size : (rank+1)*local_batch_size]`. For position `< n_order = max(1, round(local_batch_size * order_fraction))`, sample `secondary_rank` uniformly from `{1,2,3} \ {primary_rank}`; otherwise `secondary_rank = None`.
  - `MSRVTT_TrainDataLoader.__getitem__` accepts `int | CRFBatchIndex`. In CRF mode: decode clean, primary, and optional secondary in one pass; assert equal raw frame count; **compute all sampling indices, TQFS decisions, and one `frame_order` permutation once on clean raw frames** and index all three paths with them. Any missing / corrupt / shape-mismatch immediately raises `RuntimeError` carrying video_id, CRF, and path.
  - `CRFTrainBatch` (collate output; dataclass) with fields: `input_ids, input_mask, segment_ids, clean_video, video_mask, group_ids, degraded_video, primary_rank, order_video, order_rank, order_batch_indices`. Order samples are packed contiguously; `order_batch_indices[k]` maps order slot k back to its batch position for later `relevance_weights` gather.
  - `CRFTrainBatch.to(device, non_blocking=True) -> CRFTrainBatch` — returns a new dataclass where every tensor field is moved with `.to(device, non_blocking=non_blocking)`; non-tensor fields pass through unchanged. Required because the existing train loop cannot iterate a dataclass as a tuple; the CRF profile calls `batch = batch.to(device, non_blocking=True)` before forward.
  - `dataloaders/data_dataloaders.py` uses `batch_sampler=CRFDistributedBatchSampler(...)` only for the CRF profile; standard training paths unchanged. Do **not** simultaneously pass `batch_size / sampler / shuffle / drop_last` with `batch_sampler`.

Steps:

1. Failing tests in `tests/test_crf_pairing.py`:

   ```python
   def test_every_local_batch_has_fixed_order_count():
       sampler = CRFDistributedBatchSampler(dataset_size=128, local_batch_size=16,
           rank=0, world_size=4, seed=7, order_fraction=0.25)
       for batch in sampler:
           assert len(batch) == 16
           assert sum(item.secondary_rank is not None for item in batch) == max(1, round(16 * 0.25))
           assert all(item.primary_rank in (1, 2, 3) for item in batch)
           assert all(item.secondary_rank is None or item.secondary_rank != item.primary_rank
                      for item in batch)

   def test_sampler_is_epoch_deterministic():
       kw = dict(dataset_size=128, local_batch_size=16, rank=1, world_size=4,
                 seed=7, order_fraction=0.20)
       left, right = CRFDistributedBatchSampler(**kw), CRFDistributedBatchSampler(**kw)
       left.set_epoch(3); right.set_epoch(3)
       assert list(left) == list(right)

   def test_primary_rank_counts_are_balanced_across_ranks_within_each_global_batch():
       # Reassemble one global batch by collecting all world ranks and check
       # that the three primary_rank counts differ by at most 1.
       kw = dict(dataset_size=192, local_batch_size=16, world_size=4,
                 seed=11, order_fraction=0.25)
       for epoch in range(3):
           per_rank = []
           for r in range(4):
               s = CRFDistributedBatchSampler(rank=r, **kw); s.set_epoch(epoch)
               per_rank.append(list(s))
           num_batches = len(per_rank[0])
           for b in range(num_batches):
               global_batch = [item for r in range(4) for item in per_rank[r][b]]
               counts = Counter(item.primary_rank for item in global_batch)
               assert set(counts) == {1, 2, 3}
               assert max(counts.values()) - min(counts.values()) <= 1

   def test_primary_rank_marginal_is_near_uniform_across_epochs():
       # Aggregating many global batches, each rank should appear ~1/3 of the time.
       kw = dict(dataset_size=1_536, local_batch_size=16, world_size=4,
                 seed=13, order_fraction=0.20)
       counts = Counter()
       for epoch in range(4):
           for r in range(4):
               s = CRFDistributedBatchSampler(rank=r, **kw); s.set_epoch(epoch)
               for batch in s:
                   counts.update(item.primary_rank for item in batch)
       total = sum(counts.values())
       for rank_value in (1, 2, 3):
           assert abs(counts[rank_value] / total - 1/3) < 0.02
   ```
2. Run `python -m pytest tests/test_crf_pairing.py -q` → **FAIL** (`dataloaders.crf_pairing` missing).
3. Implement `CRFDistributedBatchSampler` per interfaces.
4. Wire dataloader + collate. Add identity / shared-index / corruption tests using a monkeypatched fake decoder that returns frame-numbered arrays; assert clean/primary/secondary select identical watermarks, and removing the fake degraded path raises without silently falling back to clean.
5. Run `python -m pytest tests/test_crf_pairing.py -q` → **PASS**.
6. Commit: `git add dataloaders/crf_pairing.py dataloaders/dataloader_msrvtt_retrieval.py dataloaders/data_dataloaders.py tests/test_crf_pairing.py && git commit -m "feat: load synchronized CRF training pairs"`.

### Stage B review checkpoint

- Ladder builders idempotent; `--check-only` reports zero rebuilds after a full build.
- Sampler proves fixed order count per local batch and epoch determinism.
- Clean/primary/secondary provably share frame indices; failure paths do not fall back to clean.
- Report to user and wait before Stage C.

---

## Stage C — Ranking Loss + Model Wiring + Training Preflight

**Purpose:** land the four frozen ranking losses; wire aleatoric short-circuit and paired forward with strict gradient isolation; freeze the training config, optimizer, and logging.
**Preconditions:** Stage B committed.
**Applies:** [G3], [G9], [G10], [G11], [G12], [G13].

### Task 5 — Four frozen ranking losses

- **Modify:** `uncertainty_modules/types.py`, `uncertainty_modules/losses.py`, `uncertainty_modules/__init__.py`
- **Create:** `uncertainty_modules/tests/test_crf_ranking.py`
- **Interfaces produced:**
  - `@dataclass(frozen=True) CRFRankingLoss(total, clean_frame, clean_video, order_frame, order_video)` (all `torch.Tensor`, scalars).
  - `crf_ranking_loss(u_clean_frame, u_primary_frame, primary_rank, video_mask, clean_weights, order_low_frame=None, order_high_frame=None, order_low_rank=None, order_high_rank=None, order_mask=None, order_weights=None, base_margin=0.05, lambda_cf=1.0, lambda_cv=1.0, lambda_of=1.0, lambda_ov=1.0) -> CRFRankingLoss`.
  - Frozen formulas:
    - `primary_margin = base_margin * primary_rank` (broadcast over T for frame term).
    - `clean_frame = mean_over_batch( masked_mean_over_T( relu(primary_margin + u_clean − u_primary), video_mask ) )`.
    - `clean_video = mean( relu(primary_margin + Σ_t clean_weights * u_clean − Σ_t clean_weights * u_primary) )`.
    - Order terms mirror the above with `order_margin = base_margin * (order_high_rank − order_low_rank)`, but the mean is taken over the **actual order pair count**, not `B_local`.
    - `total = λ_cf·clean_frame + λ_cv·clean_video + λ_of·order_frame + λ_ov·order_video`. When no order pairs, order terms are `u_clean_frame.sum() * 0.0` (preserves dtype/device, no NaN).
  - Masked-mean helper divides by `mask.sum(dim=1).clamp_min(1)`, so padded positions with arbitrary values never affect the result.

Steps:

1. Failing tests in `uncertainty_modules/tests/test_crf_ranking.py`:

   ```python
   def test_crf_ranking_components_match_frozen_formula():
       result = crf_ranking_loss(
           u_clean_frame=torch.tensor([[0.10,0.20],[0.20,0.30]]),
           u_primary_frame=torch.tensor([[0.30,0.40],[0.25,0.35]]),
           primary_rank=torch.tensor([1,3]),
           video_mask=torch.ones(2,2, dtype=torch.bool),
           clean_weights=torch.full((2,2), 0.5),
           order_low_frame=torch.tensor([[0.30,0.40]]),
           order_high_frame=torch.tensor([[0.50,0.60]]),
           order_low_rank=torch.tensor([1]),
           order_high_rank=torch.tensor([3]),
           order_mask=torch.ones(1,2, dtype=torch.bool),
           order_weights=torch.full((1,2), 0.5),
           base_margin=0.05,
       )
       expected = result.clean_frame + result.clean_video + result.order_frame + result.order_video
       assert torch.allclose(result.total, expected)
       assert result.order_frame.ndim == result.order_video.ndim == 0

   def test_masked_padding_never_changes_frame_loss():
       # values at masked positions must not affect the loss
       mask = torch.tensor([[True, False]])
       kw = dict(primary_rank=torch.tensor([1]), video_mask=mask,
                 clean_weights=torch.tensor([[1.0, 0.0]]))
       left  = crf_ranking_loss(torch.tensor([[0.1, 999.0]]), torch.tensor([[0.2, -999.0]]), **kw)
       right = crf_ranking_loss(torch.tensor([[0.1, 0.0]]),   torch.tensor([[0.2, 0.0]]),   **kw)
       assert torch.allclose(left.total, right.total)

   def test_order_loss_is_normalized_by_pair_count():
       # constant per-pair inputs → order loss identical whether n_pairs=1 or n_pairs=4
       def compute(n):
           low  = torch.full((n, 2), 0.20); high = torch.full((n, 2), 0.21)
           return crf_ranking_loss(
               torch.zeros(1,2), torch.ones(1,2), torch.tensor([1]),
               torch.ones(1,2, dtype=torch.bool), torch.full((1,2), 0.5),
               order_low_frame=low, order_high_frame=high,
               order_low_rank=torch.ones(n, dtype=torch.long),
               order_high_rank=torch.full((n,), 2, dtype=torch.long),
               order_mask=torch.ones_like(low, dtype=torch.bool),
               order_weights=torch.full_like(low, 0.5),
           )
       one, four = compute(1), compute(4)
       assert torch.allclose(one.order_frame, four.order_frame)
       assert torch.allclose(one.order_video, four.order_video)
   ```
2. Run `python -m pytest uncertainty_modules/tests/test_crf_ranking.py -q` → **FAIL** (`crf_ranking_loss` missing).
3. Implement loss and named result. Consume Task-4 tensor contract directly.
4. Run `python -m pytest uncertainty_modules/tests/test_losses.py uncertainty_modules/tests/test_crf_ranking.py -q` → **PASS**; existing `variance_prior_loss` tests still pass.
5. Commit: `git add uncertainty_modules/types.py uncertainty_modules/losses.py uncertainty_modules/__init__.py uncertainty_modules/tests/test_crf_ranking.py && git commit -m "feat: add ordinal CRF ranking objective"`.

### Task 6 — Aleatoric short-circuit, freeze global head, paired model forward

- **Modify:** `uncertainty_modules/aleatoric.py`, `uncertainty_modules/tests/test_aleatoric.py`, `modules/modeling.py`, `main_task_retrieval.py`
- **Create:** `tests/test_crf_model_integration.py`
- **Interfaces produced:**
  - `AleatoricUncertaintyModule(..., pooling_temperature=0.0)` uses `aggregation_logits = relevance_logits` when `pooling_temperature == 0.0` (real short-circuit; **never** `relevance_logits - 0 * local_uncertainty` since `0 * -inf → NaN`).
  - The aleatoric module on the model is `self.video_aleatoric` (existing attribute in `modules/modeling.py`). Do **not** rename it; do **not** introduce `self.aleatoric` — a rename would break `state_dict` compatibility.
  - Model constructor calls `self.video_aleatoric.global_uncertainty_head.requires_grad_(False)`.
  - Training forward order (fixed): clean visual forward → retrieval loss branch; `.detach()` clean visual features into `self.video_aleatoric`; degraded + order sub-batch traverse the **same** visual encoder under `self.training` inside one `torch.no_grad()` block (never a temporary `.eval()`); all resulting visual features `.detach()`; take `relevance_weights = clean_output.relevance_weights.detach()` (the dataclass field, not the dataclass itself); gather order weights via `order_batch_indices`; swap primary/secondary by rank into low/high; call `crf_ranking_loss(...)`. Return total loss and a named-tensor dict of the four detached components.
  - Public inference API:
    ```python
    def score_video_risk(self, video, video_mask):
        visual = self.get_visual_output(video, video_mask)
        output = self.video_aleatoric(visual, video_mask.bool())
        u_frame = output.local_uncertainty
        u_video = torch.sum(output.relevance_weights * u_frame, dim=1)
        return u_frame, u_video
    ```
  - Standard retrieval eval must not receive paired arguments — return contract unchanged.
  - `degrade_video_pixels` + its five CLI flags removed; verify with `rg -n "degrade_video_pixels|gaussian_noise|random_black|downsample" modules main_task_retrieval.py` returns no hits (historical prose in tests may remain).

Steps:

1. Failing tests:

   ```python
   # uncertainty_modules/tests/test_aleatoric.py
   def test_zero_pooling_temperature_short_circuits_uncertainty_values():
       module = AleatoricUncertaintyModule(4, 8, pooling_temperature=0.0)
       features = torch.randn(2, 3, 4)
       before = module(features).embedding.mean
       with torch.no_grad():
           for p in module.uncertainty_head.parameters():
               p.add_(torch.randn_like(p) * 1000)
       after = module(features).embedding.mean
       assert torch.equal(before, after)

   # tests/test_crf_model_integration.py
   def test_global_head_is_frozen_and_absent_from_optimizer(model, optimizer):
       assert all(not p.requires_grad for p in model.video_aleatoric.global_uncertainty_head.parameters())
       optimized = {id(p) for g in optimizer.param_groups for p in g["params"]}
       assert all(id(p) not in optimized for p in model.video_aleatoric.global_uncertainty_head.parameters())

   def test_same_visual_input_is_deterministic_in_training_dtype(model, video, mask):
       with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
           first  = model.get_visual_output(video, mask)
           second = model.get_visual_output(video, mask)
       assert torch.allclose(first, second, rtol=1e-4, atol=1e-5)

   def test_identity_pair_delta_is_below_registered_bound(model, video, mask):
       _, clean    = model.score_video_risk(video, mask)
       _, identity = model.score_video_risk(video.clone(), mask.clone())
       assert torch.all(torch.abs(identity - clean) < 5e-4)
   ```

   Also add two backward tests: retrieval-only → uncertainty/global grads are None or strict-zero; rank-only → relevance head + retrieval backbone grads are None or strict-zero.
2. Run `python -m pytest uncertainty_modules/tests/test_aleatoric.py::test_zero_pooling_temperature_short_circuits_uncertainty_values -q` → **FAIL**.
3. Implement short-circuit in `aleatoric.py`.
4. Implement paired forward + `score_video_risk` in `modeling.py`; wire loss dict return.
5. Remove `degrade_video_pixels` and its five flags; run the ripgrep above (must be empty).
6. Run `python -m pytest uncertainty_modules/tests/test_aleatoric.py uncertainty_modules/tests/test_crf_ranking.py tests/test_crf_model_integration.py -q` → **PASS**.
7. Commit: `git add uncertainty_modules/aleatoric.py uncertainty_modules/tests/test_aleatoric.py modules/modeling.py tests/test_crf_model_integration.py main_task_retrieval.py && git commit -m "feat: train aleatoric risk on CRF pairs"`.

### Task 7 — Frozen training config, optimizer, logging, experiment manifest

- **Modify:** `main_task_retrieval.py`, `dataloaders/data_dataloaders.py`, `experiment_tracking.py`, `tests/test_crf_model_integration.py`
- **Create:** `tests/test_experiment_tracking.py`
- **Interfaces produced:**
  - New CLI flags (defaults exactly as listed):
    ```
    --aleatoric_crf_training
    --crf_ladder_root                  (default data/generated/msrvtt_crf)
    --crf_order_fraction float         (default 0.20)
    --aleatoric_base_margin float      (default 0.05)
    --aleatoric_lambda_cf float        (default 1.0)
    --aleatoric_lambda_cv float        (default 1.0)
    --aleatoric_lambda_of float        (default 1.0)
    --aleatoric_lambda_ov float        (default 1.0)
    ```
    Reuse existing `--aleatoric_ranking_weight`, `--aleatoric_prior_weight`, and `pooling_temperature` — **do not** add duplicate flags.
  - `prep_optimizer()` still consumes only `_trainable_named_parameters(model)`, so the frozen global head naturally never enters any of the four param groups. `_unpack_train_batch()` calls `batch.to(device, non_blocking=True)` for the CRF profile (using `CRFTrainBatch.to`, not per-tensor iteration) and returns the dataclass; existing 5/6/8/9-tuple batches keep their per-tensor `.to()` code path unchanged.
  - `train_epoch()` unpacks model return `(total, loss_terms)`; gradient accumulation applies to `total` only. Log window means for `rank / cf / cv / of / ov` alongside retrieval loss (retrieval loss definition unchanged).
  - `experiment_tracking.py` records: recipe, `TRAIN_CRFS`, `base_margin`, `p`, four lambdas, prior weight, pooling temperature, ladder manifest sha256. Snapshot only (no second-pass verification). Standard hygiene runs do **not** gain enabled CRF fields.

Steps:

1. Failing test: config-parse test asserts CRF runner produces `prior=0`, `temperature=0`, `ranking_weight=1`, `base_margin=0.05`, four lambdas = `1.0`, `p=0.20`, `datatype=msrvtt`, `eval_split=val`, `run_final_test=False`. Manifest snapshot test asserts the recorded fields exactly.
2. Implement flag wiring + batch unpacking + logging.
3. Update `experiment_tracking.py` snapshot payload.
4. Run `python -m pytest tests/test_crf_model_integration.py tests/test_experiment_tracking.py -q` → **PASS**; hygiene namespace snapshot unchanged.
5. Commit: `git add main_task_retrieval.py dataloaders/data_dataloaders.py experiment_tracking.py tests/test_crf_model_integration.py tests/test_experiment_tracking.py && git commit -m "feat: enforce CRF training contract"`.

### Stage C review checkpoint

- Ranking loss components sum, mask, and per-pair normalization proven.
- Global head frozen and excluded from optimizer; paired forward deterministic; identity delta below 5e-4.
- CRF config fully frozen; experiment manifest snapshot verified.
- Report to user and wait before Stage D.

---

## Stage D — Statistics, CRF-dev Selection, Audit + MSVD Diagnostics

**Purpose:** implement the pre-registered statistics, the single-pass evaluator, checkpoint selection on CRF-dev, and the §3.5 diagnostic table + throughput.
**Preconditions:** Stage C committed.
**Applies:** [G7], [G13], [G14], [G15], [Appendix C].

### Task 8 — Statistics, Laplacian baseline, pre-registered criteria

- **Create:** `crf_ladder/statistics.py`, `tests/test_crf_statistics.py`
- **Interfaces produced:**
  - `paired_accuracy(a, b) -> float` — ties count as failures.
  - `laplacian_video_score(frames_uint8_rgb) -> float`:
    - `Y = 0.299 R + 0.587 G + 0.114 B`
    - per-frame `var = cv2.Laplacian(y, CV_64F, ksize=3).var()`
    - `score = -log( median_over_frames(var) + 1e-8 )`
  - `bootstrap_video_accuracy(correct_by_video: np.ndarray[V, K], seed, samples) -> BootstrapResult(point, low, high)` — sampled indices are **video-row indices only**, never flattened pairs. For paired `delta_acc`, the same resample indices are reused for both `u` and baseline within each replicate.
  - `saturation_rate(u: np.ndarray) -> float` — fraction of samples whose distance to `u.min()` or `u.max()` is `< 0.01 * (u.max() − u.min())`.
  - `extrapolation_readable(u34, u42, u50, sat_rate) -> bool`:
    ```
    readable = median(u50 − u42) >= 0.2 * median(u42 − u34) and sat_rate <= 0.10
    ```
  - Spearman correlation is exploratory only — not part of pass/fail.

Steps:

1. Failing tests in `tests/test_crf_statistics.py`:

   ```python
   def test_laplacian_score_uses_bt601_median_and_negative_log():
       frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
       frames[1, 2:6, 2:6] = (255, 255, 255)
       score = laplacian_video_score(frames)
       assert np.isfinite(score)
       assert laplacian_video_score(np.zeros_like(frames)) > score   # flatter → higher score

   def test_accuracy_counts_ties_as_failures():
       assert paired_accuracy(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5

   def test_bootstrap_resamples_videos_not_flat_pairs():
       correct_by_video = np.array([[1,1,1], [0,0,0]], dtype=float)
       result = bootstrap_video_accuracy(correct_by_video, seed=0, samples=10000)
       assert result.point == 0.5
       assert 0.0 <= result.low <= result.high <= 1.0
   ```
2. Run `python -m pytest tests/test_crf_statistics.py -q` → **FAIL**.
3. Implement per interfaces.
4. Run `python -m pytest tests/test_crf_statistics.py -q` → **PASS**; fixed-seed CI snapshot stable.
5. Commit: `git add crf_ladder/statistics.py tests/test_crf_statistics.py && git commit -m "feat: add CRF ladder registered statistics"`.

### Task 9 — Single-pass evaluator, CRF-dev selection, audit / MSVD reports

- **Create:** `scripts/evaluate_crf_ladder.py`, `tests/test_evaluate_crf_ladder.py`
- **Modify:** `main_task_retrieval.py`
- **Interfaces produced:**
  - `choose_checkpoint(candidates) -> dict` — sort by `u_video` three-pair `{clean,34,42}` accuracy descending; on tie prefer earlier `epoch`; writes `best_crf_checkpoint.json`. `>` (strict) is used for updates in the training loop.
  - Per video/level: same decode + `model.score_video_risk()` as retrieval eval (single-pass; clean, 34, 36, 42, 44, 50 never packed as paired forward). Laplacian consumes the same decoded RGB frames.
  - Per-video TSV columns: `dataset, split, video_id, level, u_video, laplacian, u_frame_mean, u_frame_min, u_frame_max, saturated_low, saturated_high`.
  - CLI:
    - `--dataset msrvtt --split {dev,audit}` — MSRVTT dev selection and one-shot audit.
    - `--dataset msvd  --split msvd_test`    — MSVD fixed test subset (its own split token; do **not** reuse the `audit` label, which would collide with the MSRVTT audit exclusive-write rule).
    - Common: `--checkpoints`, `--selection-json`, `--clean-root`, `--crf-root`, `--output-dir`.
  - `write_audit_report(path, payload)` uses **exclusive create** (`open(path, "x")`); raises `FileExistsError("already exists")` if the target JSON already exists. This rule applies **only** to MSRVTT `--split audit`.
  - MSVD (`--split msvd_test`) does **not** default to silent overwrite. Instead:
    - The output filename embeds the selected checkpoint's SHA-256 (12-hex prefix), e.g. `msvd_report_<ckptsha12>.json` and matching `.tsv` / `.md`. Different checkpoints therefore never collide.
    - If a file with the exact same name already exists (same checkpoint), the evaluator refuses unless the caller passes `--overwrite`. `--overwrite` also stamps `overwritten_at` / `overwrote_previous_sha256` fields into the new report so the previous artifact is traceable.
    - No `--overwrite` flag is available for MSRVTT `--split audit`; that gate remains exclusive-create forever.
  - Report JSON (audit) must include booleans:
    ```json
    {
      "in_domain_accuracy_pass": true,
      "extrapolation_accuracy_pass": true,
      "extrapolation_readable": true,
      "baseline_gate_pass": true,
      "falsified": false
    }
    ```
    with `falsified = (audit_u_video_accuracy < 0.90) or (challenge_delta_ci_low <= 0)`. Markdown conclusion says *"incremental compression damage under the fixed x264 recipe"*; baseline-failure wording uses the spec's narrowed phrase; never *"blur detector"* or *"true image quality"*.
  - `--dataset msvd --split msvd_test` uses Task 3's fixed subset and manifest's original geometry/fps; only clean/34/42 three-pair. Report `accuracy`, video-bootstrap CI, and boolean `cross_domain_pass = accuracy >= 0.80 and ci_low > 0.5`. Failure sets only `cross_domain_pass=false`, never `falsified=true`. Additionally report `Spearman(u, -bits_per_pixel)` tagged `exploratory: true`.

Steps:

1. Failing tests in `tests/test_evaluate_crf_ladder.py`:

   ```python
   def test_checkpoint_tie_break_chooses_earlier_epoch():
       candidates = [
           {"epoch": 0, "accuracy": 0.91, "path": "epoch0.bin"},
           {"epoch": 1, "accuracy": 0.91, "path": "epoch1.bin"},
           {"epoch": 2, "accuracy": 0.90, "path": "epoch2.bin"},
       ]
       assert choose_checkpoint(candidates)["path"] == "epoch0.bin"

   def test_audit_refuses_to_overwrite_an_existing_report(tmp_path):
       report = tmp_path / "audit.json"
       report.write_text("{}")
       with pytest.raises(FileExistsError, match="already exists"):
           write_audit_report(report, {"checkpoint": "selected.bin"})

   def test_msvd_output_filename_embeds_checkpoint_sha_and_requires_overwrite_on_collision(tmp_path):
       ckpt = tmp_path / "epoch3.bin"; ckpt.write_bytes(b"weights")
       sha12 = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
       out = tmp_path / "msvd"
       first = write_msvd_report(output_dir=out, checkpoint_path=ckpt, payload={"accuracy": 0.83})
       assert first.name == f"msvd_report_{sha12}.json"
       with pytest.raises(FileExistsError, match="already exists.*--overwrite"):
           write_msvd_report(output_dir=out, checkpoint_path=ckpt, payload={"accuracy": 0.84})
       second = write_msvd_report(output_dir=out, checkpoint_path=ckpt,
                                  payload={"accuracy": 0.84}, overwrite=True)
       body = json.loads(second.read_text())
       assert body["overwrote_previous_sha256"] == hashlib.sha256(b'{"accuracy": 0.83}').hexdigest()
       assert "overwritten_at" in body
   ```
2. Run `python -m pytest tests/test_evaluate_crf_ladder.py -q` → **FAIL**.
3. Implement single-pass scoring, dev selection, audit gates, MSVD zero-shot, exploratory Spearman.
4. Wire dev selection into `main_task_retrieval.py`: each CRF epoch runs the evaluator's Python API on CRF-dev; standard retrieval R@1 continues to report but does **not** update CRF best. Writes `best_crf_checkpoint.json`. CRF profile must not enter `run_final_test`; audit is a separate explicit command.
5. Run `python -m pytest tests/test_crf_statistics.py tests/test_evaluate_crf_ladder.py -q` → **PASS**.
6. Commit: `git add scripts/evaluate_crf_ladder.py tests/test_evaluate_crf_ladder.py main_task_retrieval.py && git commit -m "feat: evaluate registered CRF ladder gates"`.

### Task 10 — Full §3.5 diagnostic table + throughput smoke

- **Create:** `scripts/diagnose_crf_training_contract.py`, `tests/test_diagnose_crf_contract.py`
- **Modify:** `tests/test_crf_model_integration.py`
- **Interfaces produced:**
  - Diagnostic JSON schema (fixed keys, see [Appendix C]): `retrieval_gradient_isolation`, `ranking_gradient_isolation`, `global_head_frozen`, `all_trainable_parameters_have_grad`, `double_forward_allclose`, `retrieval_invariance`, `identity_delta_max`, `risk_distributions`, `frame_accuracy`, `video_accuracy`, `saturation_rate`, `hygiene_r1_delta`, `throughput_seconds_per_step`.
  - Assertions:
    - Sequentially zero grads and backward retrieval / ranking / total. Per-parameter (not per-module) check: `.grad is None` or `torch.count_nonzero(grad) == 0` for the isolated group; after total backward every `requires_grad=True` parameter has grad.
    - Double visual forward under real bf16 autocast, `rtol=1e-4`, `atol=1e-5`.
    - Retrieval invariance: `torch.equal(logits_before, logits_after)` after random perturbation of uncertainty head weights.
    - Identity: strict `< 5e-4`.
    - Distribution rows: `count / mean / std / p5 / p50 / p95` for clean and each of u30/u38/u46.
    - Frame and video accuracy reported separately; saturation uses 1% range.
    - Hygiene R@1: `abs(delta) < 1.1` — informational, never a selection criterion.
  - Throughput: same hardware + local batch; warm up 20 steps, time 100 steps for baseline and CRF; report median seconds/step, decode-wait ratio, and speed ratio. No CI, no automatic p change; changing p requires rerunning the full tuning/selection protocol on CRF-dev.

Steps:

1. Failing test locking the JSON schema keys above.
2. Implement gradient / determinism / identity / distribution / hygiene / throughput sections.
3. Run `python -m pytest tests/test_crf_model_integration.py tests/test_diagnose_crf_contract.py -q` → **PASS**.
4. Commit: `git add scripts/diagnose_crf_training_contract.py tests/test_crf_model_integration.py tests/test_diagnose_crf_contract.py && git commit -m "test: register CRF training diagnostics"`.

### Stage D review checkpoint

- Bootstrap resamples on video row indices; ties fail accuracy; extrapolation readability formula matches spec.
- Evaluator is single-pass, refuses to overwrite audit reports, and produces the exact boolean payload.
- Dev selection uses strict `>`; ties preserve earlier epoch.
- Diagnostic JSON keys match [Appendix C]; throughput ratio recorded but does not gate anything.
- Report to user and wait before Stage E.

---

## Stage E — MSVD, Standalone Runner, README, Acceptance

**Purpose:** ship the standalone experiment runner, update the README, and complete a narrow-to-wide verification loop (unit tests → 1-batch GPU smoke). No experiment run happens automatically.
**Preconditions:** Stage D committed.
**Applies:** [G6], [G7], [G11], [G16], [Appendix A], [Appendix C].

### Task 11 — Standalone runner + end-to-end acceptance

- **Create:** `run_train_msrvtt_crf_bg.sh`, `tests/test_run_train_msrvtt_crf_bg.py`
- **Modify:** `uncertainty_modules/README.md`
- **Interfaces produced:**
  - `run_train_msrvtt_crf_bg.sh` reuses the baseline runner's process management, GPU / batch / hygiene guards, and logging; but its command only references train/val files.
  - Preflight sequence before training: (a) `--check-only` trusted-v1 split, (b) `--check-only` ladder builder, (c) real-codec preflight (`CRF_PREFLIGHT_VIDEO`).
  - Fixed flags in the command line:
    ```
    --aleatoric_crf_training
    --aleatoric_prior_weight 0
    --aleatoric_pooling_temperature 0
    --aleatoric_ranking_weight 1.0
    --aleatoric_base_margin 0.05
    --aleatoric_lambda_cf 1.0
    --aleatoric_lambda_cv 1.0
    --aleatoric_lambda_of 1.0
    --aleatoric_lambda_ov 1.0
    --crf_order_fraction 0.20
    ```
    Tail-arg override list must reject overriding any of these. **Never** references `--run_final_test` or `MSRVTT_JSFUSION_test.csv`.
  - After training, the runner **prints** (does not execute) two explicit commands:
    ```bash
    python scripts/evaluate_crf_ladder.py --dataset msrvtt --split audit     --selection-json "${OUTPUT_DIR}/best_crf_checkpoint.json" --output-dir "${OUTPUT_DIR}/crf_audit"
    python scripts/evaluate_crf_ladder.py --dataset msvd   --split msvd_test --selection-json "${OUTPUT_DIR}/best_crf_checkpoint.json" --output-dir "${OUTPUT_DIR}/crf_msvd"
    ```
    Note the MSVD command uses `--split msvd_test`, not `--split audit` — MSVD is not subject to the MSRVTT audit exclusive-write rule. `--selection-json` avoids a `jq` dependency.
  - README updates: `local_uncertainty ≡ u_frame`, weighted value `≡ u_video`. Keep variance's probabilistic meaning but state this experiment does **not** supervise it. Note `base_margin` is a heuristic monotone gap (not a claim that CRF damage is linear). Scope only to current encoder + x264 recipe.

Steps:

1. Failing tests in `tests/test_run_train_msrvtt_crf_bg.py`:

   ```python
   ROOT   = Path(__file__).resolve().parents[1]
   SCRIPT = ROOT / "run_train_msrvtt_crf_bg.sh"

   def test_crf_runner_has_frozen_flags_and_never_requests_final_test():
       text = SCRIPT.read_text()
       assert "--aleatoric_crf_training" in text
       assert "--aleatoric_prior_weight 0" in text
       assert "--crf_order_fraction 0.20" in text
       assert "--aleatoric_base_margin 0.05" in text
       assert "--run_final_test" not in text
       assert "MSRVTT_JSFUSION_test.csv" not in text
       # MSVD helper command uses its own split token, never the MSRVTT audit label.
       assert "--dataset msvd --split msvd_test" in text
       assert "--dataset msvd --split audit"     not in text

   def test_crf_runner_parses_as_shell():
       subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
   ```
2. Run `python -m pytest tests/test_run_train_msrvtt_crf_bg.py -q` → **FAIL** (script missing).
3. Implement runner + preflight sequence + printed audit commands.
4. Update README per interfaces.
5. Narrow-to-wide verification:
   - `python -m pytest tests/test_run_train_msrvtt_crf_bg.py tests/test_crf_protocol.py tests/test_crf_codec.py tests/test_build_crf_ladder.py tests/test_crf_pairing.py uncertainty_modules/tests/test_crf_ranking.py tests/test_crf_model_integration.py tests/test_crf_statistics.py tests/test_evaluate_crf_ladder.py tests/test_diagnose_crf_contract.py -q` → **PASS** (only real-codec preflight may skip when `CRF_PREFLIGHT_VIDEO` unset).
   - `python -m pytest -q` → full suite **PASS**, no new failures.
   - `python -m ruff check crf_ladder dataloaders uncertainty_modules scripts tests main_task_retrieval.py modules/modeling.py experiment_tracking.py` → `All checks passed!`.
6. 1-batch GPU smoke:
   - `CRF_PREFLIGHT_VIDEO=/data2/hxj/data/MSRVTT/videos/compressed_videos/msrvtt_224_12fps/video0.mp4 SMOKE_STEPS=1 bash run_train_msrvtt_crf_bg.sh --epochs=1`
   - Expected: every rank has order pairs; `cf/cv/of/ov` finite; no unused-parameter / DDP / static-graph errors; diagnostic JSON reports frozen global head, double-forward allclose, identity + retrieval invariance pass.
7. Commit: `git add run_train_msrvtt_crf_bg.sh tests/test_run_train_msrvtt_crf_bg.py uncertainty_modules/README.md && git commit -m "feat: add reproducible CRF ladder experiment"`.

### Stage E review checkpoint

- Runner refuses to run final-test; preflights all green.
- Full narrow-to-wide test matrix + ruff pass; 1-batch smoke green.
- README wording adheres to [G3] and [G14] scope.
- Report to user and hand off — do not launch the full experiment without explicit user approval.

---

## Post-Code Experiment Order (only after Stage E approval)

1. Full build + `--check-only` verify MSRVTT train/val ladder; record all manifest hashes.
2. CRF-dev tuning limited to `p` and the four lambdas / diagnostic thresholds; each change → its own experiment manifest.
3. Choose checkpoint by CRF-dev `u_video` accuracy with strict `>` and earlier-epoch tie-break. Retrieval R@1 does not participate.
4. Freeze final checkpoint + statistics code, then run CRF-audit **once**; save command, git hash, manifest hash, per-video TSV.
5. If either in-domain falsification condition fires, close the route; do not proceed to `temperature > 0`.
6. If in-domain passes, run MSVD zero-shot once on the fixed subset; failure narrows to "in-domain holds; cross-domain unverified".
7. Only when extrapolation readability holds (median ratio + saturation) may CRF50 extrapolation be reported; otherwise report interpolation only.

---

## Appendix A — Frozen x264 Recipe

`X264Recipe` field values (source of truth; referenced by Task 1 test and Task 2 encoder options):

| field | value |
|---|---|
| `preset` | `veryfast` |
| `pixel_format` | `yuv420p` |
| `color_range` | `mpeg` |
| `color_primaries` | `smpte170m` |
| `color_trc` | `smpte170m` |
| `colorspace` | `smpte170m` |
| `time_base_policy` | `inverse_input_fps` |
| `keyint` | `48` |
| `min_keyint` | `48` |
| `scenecut` | `0` |
| `b_frames` | `2` |

`encoder_options(crf)` returns exactly `{"crf": str(crf), "preset": "veryfast", "x264-params": "keyint=48:min-keyint=48:scenecut=0:bframes=2:force-cfr=1:colorprim=smpte170m:transfer=smpte170m:colormatrix=smpte170m:range=limited"}`.

## Appendix B — `ManifestRow` Fields

Referenced by Task 2 (`crf_ladder/codec.py`), Task 3 (builders), and Task 7 (experiment manifest hash). All fields required, sorted keys when serialized:

`video_id, source_path, output_path, source_sha256, output_sha256, crf, preset, encoder, pyav_version, av_library_versions, pixel_format, color_range, color_primaries, color_trc, colorspace, fps_num, fps_den, time_base_num, time_base_den, frame_count, width, height, keyint, min_keyint, scenecut, b_frames, size_bytes, bitrate_bps, success`.

Notes:
- `frame_count` must be counted by decoding; do not use container estimate.
- `bitrate_bps = round(size_bytes * 8 / duration_seconds)` from measured metadata; **never** from recipe constants.
- `encoder` is the literal string `"libx264"` (matches the PyAV stream codec name; no version implied).
- `pyav_version = av.__version__`.
- `av_library_versions` is the full `av.library_versions` dict (linked FFmpeg component versions such as `libavcodec`, `libavformat`, `libavutil`, `libswscale`), serialized as a sorted-key JSON string. This is the manifest's version fingerprint; downstream tooling distinguishes builds by comparing this string together with the frozen recipe (Appendix A) and the output SHA-256.
- **Precise `x264_version` is intentionally NOT a manifest field.** The reviewer confirmed no reliable source exists: `av.codec.Codec.properties` is not a stable PyAV contract, and a system-`PATH` `x264 --version` is not guaranteed to equal the libx264 that PyAV/FFmpeg was linked against. Recording an unverifiable version would falsely imply audit-grade provenance.
- **Optional fail-closed audit mode:** builders accept `--require-x264-provenance {none,strict}` (default `none`). In `strict`, before spawning workers the parent runs a preflight that (a) links libx264 through PyAV and (b) reads its version through whichever mechanism the local PyAV build exposes; if the mechanism is absent OR the read fails OR the returned value is empty, the builder exits non-zero with `x264 provenance unavailable — refusing to build audit-grade ladder`. When this mode succeeds, the resolved version is recorded once in the run-level `experiment_manifest.json` (not in every `ManifestRow`) as `{"x264_version": "<hex>", "x264_version_source": "<mechanism>"}`. CRF-audit runs (Task 11 helper command) require `--require-x264-provenance strict` at build time; MSRVTT/MSVD smoke and CRF-dev tuning may use `none`.

## Appendix C — Diagnostic JSON Schema (Task 10)

Fixed key set (any additional keys are additive but must not remove these):

```
retrieval_gradient_isolation          bool   # per-parameter check on ranking-branch backward
ranking_gradient_isolation            bool   # per-parameter check on retrieval-branch backward
global_head_frozen                    bool
all_trainable_parameters_have_grad    bool
double_forward_allclose               bool   # bf16 autocast, rtol=1e-4, atol=1e-5
retrieval_invariance                  bool   # torch.equal after uncertainty-head perturbation
identity_delta_max                    float  # must be < 5e-4
risk_distributions                    { "clean|u30|u38|u46": {count,mean,std,p5,p50,p95} }
frame_accuracy                        { "clean_vs_30": …, "30_vs_38": …, "38_vs_46": … }
video_accuracy                        { same three keys as frame_accuracy }
saturation_rate                       float  # 1% of full range
hygiene_r1_delta                      float  # |delta| < 1.1, informational only
throughput_seconds_per_step           { baseline_median, crf_median, decode_wait_ratio, ratio }
```

## Appendix D — Frozen Loss Contract (Task 5 reference)

- Input tensor shapes: `u_clean_frame[B,T]`, `u_primary_frame[B,T]`, `primary_rank[B]`, `video_mask[B,T]` (bool), `clean_weights[B,T]`. Order tensors: `order_low_frame[n_order,T]`, `order_high_frame[n_order,T]`, `order_low_rank[n_order]`, `order_high_rank[n_order]`, `order_mask[n_order,T]`, `order_weights[n_order,T]`.
- Ranks are ordinal `{1,2,3}` mapping `{30:1, 38:2, 46:3}`.
- Masked mean divisor: `mask.sum(dim=1).clamp_min(1)`.
- When `order_low_frame is None`: `order_frame = order_video = u_clean_frame.sum() * 0.0` (dtype/device preserving; no NaN).

---

## Compression Changelog

- **Original length:** 1382 lines (`docs/superpowers/plans/2026-08-09-crf-degradation-ladder.md`, as of the file this rewrite replaces). The request cited "~3277 lines"; the pointed-to attachment was not on disk, so the rewrite operated on the actual 1382-line file. Some values (see Open Decisions) may differ from the 3277-line source and now require confirmation.
- **New length:** see `wc -l` after write. Below the 800-line soft floor before the reviewer round; grew again with restored contracts (checked-in split JSON, sampler balance rule + tests, `CRFTrainBatch.to`, x264 version sourcing, MSVD split split, corrected `self.video_aleatoric` interface). Not padded — every added line is a real contract.
- **Reviewer-round #1 fixes applied:**
  - **[G8]** `crf_order_fraction` default reverted to `0.20`; near-uniform primary-rank balance rule added (max count difference ≤ 1 within each global batch).
  - **[G2]** Outdated "current `7de0994` does not contain `1e11659`" sentence removed; only the check commands remain.
  - **Task 1** now creates and checks in `dataloaders/splits/crf_dev_audit_seed0.json` with `source_manifest_sha256` for audit visibility; new test asserts the checked-in JSON matches the manifest hash.
  - **Task 4** sampler documents primary-rank balancing across ranks within each global batch; added `test_primary_rank_counts_are_balanced_across_ranks_within_each_global_batch` and `test_primary_rank_marginal_is_near_uniform_across_epochs`.
  - **Task 4/7** `CRFTrainBatch.to(device, non_blocking=True)` added; `_unpack_train_batch` uses it (dataclass, not tuple iteration).
  - **Task 6** attribute name corrected to `self.video_aleatoric` throughout; `relevance_weights = clean_output.relevance_weights.detach()` (dataclass field, not the dataclass itself); state-dict compatibility preserved.
  - **Task 9/11** MSVD CLI decoupled from MSRVTT audit — `--dataset msvd --split msvd_test`; the exclusive-create write rule applies only to MSRVTT `--split audit`; runner test asserts MSVD does not use `--split audit`.
- **Reviewer-round #2 fixes applied:**
  - **[G2]** clarified: on ancestor-check failure, stop and ask the user; no autonomous branch switching.
  - **Task 2** interface: split into `append_manifest_row` (single-writer, parent-only) and `build_manifest_row` (pure, worker-callable). Typo "mules all frames" → "encodes and muxes every decoded frame".
  - **Task 3** now specifies a parent-sole-writer concurrency model for `--workers N`. Added tests: `test_workers_never_open_the_manifest_file`, `test_multi_worker_jsonl_has_one_row_per_completed_job_and_is_deterministic`, `test_resume_after_partial_run_never_duplicates_rows`.
  - **Task 9** MSVD reports now embed a 12-hex checkpoint SHA in the output filename; collisions require explicit `--overwrite`, which stamps `overwrote_previous_sha256` + `overwritten_at`. New test `test_msvd_output_filename_embeds_checkpoint_sha_and_requires_overwrite_on_collision`. MSRVTT audit exclusive-create is unchanged.
  - **Appendix B** removed the unreliable `x264_version` field (`Codec.properties` and system `x264` were not trustworthy sources). Manifest now records `encoder="libx264"` + full `av_library_versions` dict. Precise x264 provenance is gated behind opt-in `--require-x264-provenance strict` at build time; CRF-audit builds must use `strict`.
- **Structural changes:**
  - Replaced the `superpowers:executing-plans` / subagent-driven-development preamble with a single **Execution mode** line stating inline sequential development.
  - Added a top **Execution Summary** (≤15 lines) covering the five stages, execution mode, and key acceptance gates.
  - Consolidated the Global Constraints as the **single source of truth** and numbered them **[G1]–[G17]**. Stage bodies now reference constraint numbers instead of re-quoting the constraints.
  - Reorganized Tasks 1–11 into **five serial stages (A–E)**, keeping the original task numbers as sub-items within each stage as requested.
  - **Removed duplicated content:** repeated prose blocks about clean-source rules, drop_last, prior=0/pooling_temperature=0, frame-index sharing, no-clean-fallback, and audit-only-once (now only in the constraints). Removed multi-page code skeletons that duplicated existing project code (e.g. full PyAV `_transcode` body, full `CRFDistributedBatchSampler` body, full `crf_ranking_loss` implementation, full `AleatoricUncertaintyModule` snippet). Each is now expressed as an interface signature + acceptance behavior, with formulas and hard-coded constants kept.
  - **Moved to appendices:** (A) x264 recipe field table + full `x264-params` string; (B) full `ManifestRow` field list; (C) fixed diagnostic JSON schema; (D) frozen loss tensor-shape contract. Stage bodies reference these instead of repeating them.
  - Retained: every acceptance threshold, CLI flag, path, JSON boolean, `>=`/`<=` operator, tie-break rule, exclusive-create audit write, real-codec preflight gating, ripgrep sanity check, narrow-to-wide test list, ruff invocation, 1-batch smoke command, and the seven-step post-code experiment order.
  - Test snippets kept only where they anchor a non-obvious contract (order-count formula, tie-break, exclusive-create audit). Boilerplate imports and duplicate assertion scaffolding removed.

## Open decisions requiring approval

Reviewer round #2 accepted `p=0.20`, `msvd_test`, and requested x264 fail-closed. Nothing below is currently blocking Stage A — all four are explicit ratifications of positions the plan already encodes. Recorded here so the ratifications are auditable and any future contributor can find the reasoning.

1. **`crf_order_fraction=0.20`** — **RATIFIED** by reviewer. Encoded at [G8], Task 7 flag default, Task 11 runner literal, and runner shell test.
2. **MSVD split token `msvd_test`** — **RATIFIED** by reviewer. Encoded at Task 9 CLI block, `write_audit_report` scope note, Task 11 helper-command block, and runner shell test (both positive and negative assertions).
3. **x264 provenance policy** — **RATIFIED as fail-closed.** Appendix B removes precise `x264_version` from every `ManifestRow`; instead the manifest records `encoder="libx264"` and the full `av_library_versions` dict. Builders accept `--require-x264-provenance {none,strict}` (default `none`); `strict` runs a preflight that must produce a non-empty version through PyAV — otherwise the run fails. CRF-audit at Task 11 requires `strict`; smoke/dev tuning may use `none`.
4. **[G2] no auto-branch-switch** — **CONFIRMED**. If the `1e11659` ancestor check fails, the plan requires stopping and asking the user which branch to work on; it does not authorize checkout/reset/merge on its own.

Non-blocking items surfaced by reviewer round #2 and now encoded in the plan (no approval required, listed for traceability):

5. **Single-writer manifest under `--workers 8`** — Task 3 concurrency model now names the parent as the sole `append_manifest_row` caller; workers only return `ManifestRow` values. Verified by `test_workers_never_open_the_manifest_file`, `test_multi_worker_jsonl_has_one_row_per_completed_job_and_is_deterministic`, and `test_resume_after_partial_run_never_duplicates_rows`.
6. **MSVD overwrite semantics** — output filename embeds a 12-hex checkpoint SHA; collision requires explicit `--overwrite`, which stamps `overwritten_at` and `overwrote_previous_sha256` into the new report. MSRVTT audit remains exclusive-create-forever.
7. **Task 2 wording fix** — "mules all frames" was a typo; corrected to "encodes and muxes every decoded frame".

If any additional detail was lost in compression relative to the 3277-line source (which was not accessible during rewriting), please point to it and it will be added here rather than merged silently.
