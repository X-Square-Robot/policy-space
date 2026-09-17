#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$POLICY_DIR/../../.." && pwd)"
PYTHON_BIN="${POLICY_SPACE_PYTHON:-python}"
: "${CUDA_VISIBLE_DEVICES:?Select an available GPU explicitly after nvidia-smi}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/packages/protocol/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -n "${TMPDIR:-}" ]]; then
  mkdir -p "$TMPDIR"
fi
if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  mkdir -p "$XDG_CACHE_HOME"
fi
if [[ -n "${JAX_COMPILATION_CACHE_DIR:-}" ]]; then
  mkdir -p "$JAX_COMPILATION_CACHE_DIR"
fi
cd "$PROJECT_ROOT"
if [[ "${1:-}" == "--check" ]]; then
  shift
  exec "$PYTHON_BIN" -m policy_space.cli check "$POLICY_DIR/deploy.yaml" "$@"
fi
exec "$PYTHON_BIN" -m policy_space.cli serve "$POLICY_DIR/deploy.yaml" "$@"
