# Scientific gate status, 2026-10-10

New-chain production four-CUDA-rank qualification passed and is committed in `a0bb35a`; see `../summaries/Q0.json`. This is software evidence, not a scientific result. No scientific stage is authorized by this report. MSRVTT/MSVD test remains sealed.

The approved design §5.4 requires near-duplicate sources to stay in one role, but neither an audited relation table nor a defined near-duplicate criterion was found in the inspected source confirmation/assets. Canonical video identity and exact content SHA alone do not prove that constraint. The agent asked the user for the relation table or scientific definition; no answer had arrived when this report was written. Do not silently substitute a new definition, generate a supposedly audited five-role split, or open a training gate.

Available source evidence (read-only inspection; not a new scientific experiment):

- `dataloaders/splits/msrvtt_trusted_v1_seed0.json`: train IDs 8,500; validation IDs 500; SHA-256 `a0e5a82dddec34fa76753c2323cf560c2afc36f3683b231ccd24db2c268be882`.
- `docs/maintenance/2026-10-01-msrvtt-source-confirmation.json` records the existing official-source confirmation. It does not supply an audited near-duplicate graph.
- Annotations `/data2/hxj/data/MSRVTT/annotation/MSRVTT_v2.json`, train CSV `/data2/hxj/data/MSRVTT/csv/MSRVTT_train.9k.csv`, and compressed training/validation video source directory are available. No test evaluation was performed.
- The inspected local CLIP cache has B/16 weights, while this design's implemented capacity and Q0 use B/32. A pretrained B/32 input still needs preparation and verification; this is an outstanding prerequisite, not evidence that obtaining it is impossible.

After the scientific source definition is supplied, the remaining work is to prove five source-disjoint roles with trusted validation membership/test exclusion, bind the pretrained B/32 input and initial index, and preregister the small falsification pilot with all matched-budget arms. Only then may the corresponding stage gate be evaluated. Do not use this status report itself as a passed training gate.

Scientific metrics, scientific GPU-hours, and hypothesis verdict: **not available; 0 scientific GPU-hours; not tested**. Software Q0 allocation: **0.07994 GPU-hours**. No favorable or unfavorable scientific arm was omitted because none was run.

## 2026-10-10 source audit update

The user approved youtube-id-content-sha256-v1 and exclusion of 254 training clips sharing source components with trusted validation. The pinned read-only audit found 202 crossing original sources, 203 affected validation clips and 31 temporally overlapping cross-boundary pairs; no exact-byte duplicates among 9000 allowed videos. All 9000 source metadata records match the hash-pinned published release. See source-audit-2026-10-10.md. This resolves the operational definition question, but the actual five-role manifests, pretrained input preparation and scientific preregistration/gates remain pending. No scientific gate changed. The earlier missing-definition statement records the previous state.
