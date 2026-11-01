#!/usr/bin/env bash
# Launch only an already reserved Gaussian KL run. This script never reserves a run.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode=''
run_dir=''
dry_run=0
while (($#)); do
  case "$1" in
    --mode) mode="${2:?missing mode}"; shift 2 ;;
    --run-dir) run_dir="${2:?missing run directory}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "Usage: $0 --mode qualify|train --run-dir /reserved/run [--dry-run]" >&2; exit 2 ;;
  esac
done
[[ "$mode" == qualify || "$mode" == train ]] || { echo 'mode must be qualify or train' >&2; exit 2; }
[[ "$run_dir" == /* ]] || { echo 'run-dir must be absolute' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#devices[@]} == 4 ]] || { echo 'exactly four CUDA devices are required' >&2; exit 2; }
declare -A seen=()
for device in "${devices[@]}"; do
  [[ "$device" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ && ! -v "seen[$device]" ]] || { echo 'CUDA devices must be four distinct indices or UUIDs' >&2; exit 2; }
  seen[$device]=1
done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${UATVR_KL_PYTHON:-/data2/hxj/env/uatvr-kl-check/bin/python}"
command=("$python_bin" -m torch.distributed.run --standalone --nproc_per_node=4 --module semantic_retrieval.aqs.fusion_launch --run-dir "$run_dir" --mode "$mode")
if ((dry_run)); then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
[[ -x "$python_bin" ]] || { echo 'set UATVR_KL_PYTHON to the pinned Python executable' >&2; exit 2; }
[[ -f "$run_dir/manifest.json" ]] || { echo 'reserved manifest.json is required' >&2; exit 2; }
cd "$repo_dir"
exec "${command[@]}"
