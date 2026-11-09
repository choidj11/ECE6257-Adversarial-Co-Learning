#!/usr/bin/env bash
# Source before running Python, uv, Hugging Face, training, or evaluation.
export PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export XDG_DATA_HOME="$PROJECT_ROOT/.local/share"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
export XDG_STATE_HOME="$PROJECT_ROOT/.local/state"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_TOKEN_PATH="$HF_HOME/token"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.tools/python"
export UV_TOOL_DIR="$PROJECT_ROOT/.tools/uv-tools"
export UV_TOOL_BIN_DIR="$PROJECT_ROOT/.tools/bin"
export UV_LINK_MODE=hardlink
export UV_CONCURRENT_DOWNLOADS=2
export UV_CONCURRENT_INSTALLS=1
export UV_CONCURRENT_BUILDS=1
export PYTHONUSERBASE="$PROJECT_ROOT/.local"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PROJECT_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$PROJECT_ROOT/.cache/cuda"
export VLLM_CACHE_ROOT="$PROJECT_ROOT/.cache/vllm"
export NUMBA_CACHE_DIR="$PROJECT_ROOT/.cache/numba"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export NLTK_DATA="$PROJECT_ROOT/data/nltk"
export WANDB_DIR="$PROJECT_ROOT/artifacts/wandb"
export WANDB_CACHE_DIR="$PROJECT_ROOT/.cache/wandb"
export WANDB_CONFIG_DIR="$PROJECT_ROOT/.cache/wandb-config"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_TMPDIR="$PROJECT_ROOT/.cache/ray"
export APPTAINER_CACHEDIR="$PROJECT_ROOT/.cache/apptainer"
export APPTAINER_TMPDIR="$PROJECT_ROOT/.cache/apptainer-tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export PATH="$PROJECT_ROOT/.tools/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export USE_FLAX=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MAX_JOBS="${MAX_JOBS:-4}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_ASSETS_CACHE" \
    "$HF_XET_CACHE" "$HF_MODULES_CACHE" "$TMPDIR" "$UV_CACHE_DIR" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" \
    "$PROJECT_ROOT/models" "$PROJECT_ROOT/data" "$PROJECT_ROOT/artifacts/setup"
