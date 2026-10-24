#!/usr/bin/env bash
# Install the tested HO3D inference/evaluation dependencies into a Conda environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash setup_env.sh [ENV_NAME]

Create or reuse a Python 3.10 Conda environment (default: 4dhof), then install
CUDA 12.4, PyTorch, and the inference/evaluation dependencies in build order.
Requires Conda, Git, a C++ compiler, and internet access.
HELP
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# > 1 )); then
    usage >&2
    exit 1
fi
environment_name="${1:-4dhof}"
if [[ ! "$environment_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ || "$environment_name" == "base" ]]; then
    echo "Choose a named Conda environment other than base (letters, digits, underscores, hyphens)." >&2
    exit 1
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda_executable="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_executable" ]]; then
    echo "Install Conda and make its conda command available before running this script." >&2
    exit 1
fi
command -v git >/dev/null || { echo "Git is required to install Chumpy." >&2; exit 1; }
command -v "${CXX:-c++}" >/dev/null || { echo "A C++ compiler is required to build CUDA extensions." >&2; exit 1; }
conda_base="$("$conda_executable" info --base)"
# shellcheck source=/dev/null
source "$conda_base/etc/profile.d/conda.sh"
if ! conda activate "$environment_name" 2>/dev/null; then
    conda create --name "$environment_name" python=3.10 pip -y
    conda activate "$environment_name"
fi
python - <<'PY'
import os
import sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve():
    raise SystemExit("The selected environment has no usable Python. Choose a new environment name.")
if sys.version_info[:2] != (3, 10):
    raise SystemExit("This setup requires Python 3.10. Choose a new environment name.")
PY

printf '\nInstalling CUDA toolkit in %s\n' "$environment_name"
conda install --name "$environment_name" -y -c nvidia/label/cuda-12.4.0 cuda-toolkit=12.4.0
# Save CUDA_HOME for future activations; Conda already adds the environment's bin to PATH.
conda env config vars set --name "$environment_name" CUDA_HOME="$CONDA_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version

printf '\nInstalling PyTorch and base dependencies\n'
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$repo_root/requirements.txt"

printf '\nInstalling CUDA extensions and MANO pickle support\n'
python -m pip install xformers==0.0.29.post3 spconv-cu124==2.3.8
python -m pip install torch-scatter==2.1.2+pt26cu124 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da --no-build-isolation

printf '\nChecking installed dependencies\n'
python -m pip check
python - <<'PY'
import torch
import flash_attn
import spconv.pytorch
import torch_scatter
import xformers.ops
import open3d
import rtree
import smplx
import chumpy
import huggingface_hub
import hf_xet

if torch.__version__.split('+')[0] != '2.6.0' or torch.version.cuda != '12.4':
    raise SystemExit(f"Expected PyTorch 2.6.0 / CUDA 12.4, got {torch.__version__} / {torch.version.cuda}")
print('Hugging Face Hub:', huggingface_hub.__version__)
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('Before inference, run on a GPU machine with a compatible NVIDIA driver and a visible GPU.')
PY
hf --help >/dev/null
printf '\nSetup complete. Activate the environment in your shell:\n  conda activate %s\n' "$environment_name"
