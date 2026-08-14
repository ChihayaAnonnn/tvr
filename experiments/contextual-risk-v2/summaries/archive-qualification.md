# Experiment artifact archive qualification

## Decision

**Stage 0: PASS.** The run-centric archive is qualified for the validation-only
scientific campaign. This decision authorizes the campaign controller and does
not authorize test access or make the fixture run a scientific candidate.

Qualification was completed on node `node199` with four NVIDIA A800 80 GB PCIe
GPUs (`CUDA_VISIBLE_DEVICES=0,2,3,4`). The successful worker used one real NCCL
step in FP32, four ranks, and no dataset loader or model checkpoint.

## Fresh verification

- Focused E2E command:
  `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests/test_experiment_archive_e2e.py`
  — `2 passed in 8.74s`.
- Full command:
  `/home/xujie/.conda/envs/tvr/bin/python -m pytest -q tests`
  — `606 passed, 1 skipped, 1 warning in 79.37s`.
- The warning is the existing PyTorch `TypedStorage` deprecation warning.
- `git diff --check`, the tracked worktree diff, and the staged diff were clean
  after terminal phase publication. The only untracked control-plane file is
  the administrative `registry.jsonl.lock` created by the registry API.

The CPU E2E fixture independently exercised phase reservation and commit,
exact reservation verification, `reserved -> running -> validating ->
completed`, one optimization step, a 2 MiB array reference, scalar validation
metrics, terminal phase publication, and immutable root `logs/`/`ckpts/`
sentinels.

## Registered run lineage

The first attempt is preserved and was never modified or retried in place:

- Run ID:
  `msrvtt-baseline-archive_qualification_4gpu-seed000-20260814T090705Z-e6fddd50c4ec`
- Manifest SHA-256:
  `cd8c3049d861ee659266b3bc22326d75e539d8a7e5b7a9f3624e214ba9e402f1`
- Terminal state: `cancelled`, exit code `1`.
- Failure: all ranks exited before model construction because the directly
  executed fixture entrypoint could not import the repository-root
  `experiment_artifacts` module.
- Evidence: run-local `logs/train.log`, `provenance/failure.json`, atomic
  status, and audit chain. The failure terminal publication is commit
  `d98e426`.

The directly executable CLI and worker paths were fixed with regression tests
in commits `e6fddd5` and `41ff854`. After explicit human authorization, a new
phase reserved this unique successor rather than reusing the failed identity:

- Run ID:
  `msrvtt-baseline-archive_qualification_4gpu_retry1-seed000-20260814T164421Z-82830a3ab21d`
- Parent run ID: the cancelled run above.
- Source SHA: `82830a3ab21d8f7fba8bb6dc2553811ef566b0a9`.
- Reservation commit: `63726c6`.
- Manifest contract SHA-256:
  `9eedfc3eb1336de9d83ac5de78d6176fa2e3eb65dcbb2e93838eedce3317a7b3`.
- Manifest file SHA-256:
  `15d52810798d1ed6f044096ac770c90b0d121f05133c8f8986dc6a6b130b7c5a`.
- Terminal state: `completed`, exit code `0`.
- Terminal phase commit: `fb491ee`.

The successor manifest's `parent_run_ids` and `selection_lineage` record the
predecessor, import-path defect, fixing commit, and human authorization reason.
Both reserved and terminal records carry the same manifest digest.

## Successful four-rank evidence

- Four rank markers exist for ranks `0,1,2,3`, each reporting world size 4 and
  an A800 CUDA device.
- Exactly one lifecycle-owner marker exists, owned by rank 0.
- Mean fixture loss is finite: `4.1768999099731445`.
- The externalized array contains `2,097,152` numeric bytes. Its reference
  digest is
  `908251af60385de2b211f1fd4281d4e69cc3ece66815e1ccd2e019d24cace430`;
  its immutable NPY file digest is
  `647092735d0f83096419a07a203198e8caba708bd1effbd1b0ff903c75fb910a`.
- `metrics/selected_validation.json` is 500 bytes and has SHA-256
  `61f65c79af6c88280be5ca9862f2db97d9d3027b7696163fb9b13c14eb4e4831`.
- The run contains no `test.log`, `final_test.json`, JSFUSION path, test CSV,
  or `run_final_test` argument. Its manifest has `test_authorized=false` and
  the metric split is `validation`.
- Every qualification JSON file is below 1 MiB. The 2 MiB payload appears only
  in content-addressed NPY storage.

The successful output tree consists of the sealed command, train log, manifest,
status/audit chain, four rank markers, one owner marker, one scalar metric JSON,
and one content-addressed NPY. Terminal hashing covered every non-administrative
output.

## Four-rank failure and signal probes

Three independent ignored probes were intentionally non-registered and are not
scientific candidates. Each used a unique artifact path and retained its log,
four rank markers, and `provenance/result.json`:

1. `archive_probe_rank_failure`: rank 2 raised intentionally before a barrier;
   torchrun returned nonzero, propagated SIGTERM to ranks 0, 1, and 3, and left
   no worker process.
2. `archive_probe_collective_fault`: rank 3 exited with code 17 while the other
   ranks entered an NCCL all-reduce; torchrun returned nonzero, terminated the
   remaining ranks, and left no worker process.
3. `archive_probe_sigterm_drain`: SIGTERM was sent to the isolated torchrun
   process group after all four rank markers existed; the bounded drain found
   no remaining process in the PGID.

No probe accessed a dataset or test split. A final `nvidia-smi`/process audit
found no qualification GPU process or worker PGID.

## Archive and legacy integrity

- Registry SHA-256 after terminal publication:
  `4291e811c2a5bd9b70ca241c47a34a61b8b8ab188b0dcceec021bb08fa638e46`.
- Successor phase SHA-256:
  `f48bfb04a59c9b6dce2fad32ecdba47bfbee56f621bfc3c78487f955ef24e998`.
- Successor phase-summary SHA-256:
  `8bff15961ba37ede8ca914b8504bec96754c566bb1531987d3fd80f1d2012e7a`.
- The real legacy `logs/` and `ckpts/` roots contained 340 files before and
  after qualification. Their path/size/nanosecond-mtime fingerprint remained
  `441ae072e07c4a4683342ae2249c511143e5d2f5cf43f560c6f9ca0f964d7866`.
- All new outputs are below `artifacts/contextual-risk-v2/`; no legacy file was
  deleted, moved, or overwritten.

Stage 1 may now be prepared under the committed-reservation and validation-only
rules. Held-out MSRVTT/MSVD test access remains sealed.
