#!/usr/bin/env bash
# One-shot dev environment setup for MineIA.
#
# Tested on Ubuntu 22.04 with NVIDIA driver >= 550 (CUDA 12.1 runtime).
# Run from the repo root: `bash scripts/setup_env.sh`

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

echo ">> Checking Python..."
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. MineRL 1.0 requires Python 3.10."
  exit 1
fi

echo ">> Checking Java 8 (required by MineRL)..."
if ! java -version 2>&1 | grep -q '"1.8'; then
  echo "WARNING: Java 8 not detected as default JVM."
  echo "         On Ubuntu: sudo apt install openjdk-8-jdk && sudo update-alternatives --config java"
fi

echo ">> Checking NVIDIA driver / CUDA..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. CUDA training will not work without it."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo ">> Creating virtualenv at .venv ..."
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

echo ">> Installing PyTorch (CUDA 12.1) and project deps..."
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo ">> Installing project in editable mode..."
pip install -e .

echo ">> Cloning OpenAI VPT repo into vendor/vpt (for the policy / model defs)..."
mkdir -p vendor
if [ ! -d vendor/vpt ]; then
  git clone --depth 1 https://github.com/openai/Video-Pre-Training.git vendor/vpt
fi

echo ">> Verifying CUDA visible to PyTorch..."
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name(0)}")
PY

echo ">> Done. Next: python scripts/download_vpt.py --size 2x"
