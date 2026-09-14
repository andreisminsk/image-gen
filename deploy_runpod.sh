#!/usr/bin/env bash
# Image Generation - RunPod.io Deployment Script
# Run this on a fresh RunPod PyTorch pod (A100 40GB or better)
#
# Quick start:
#   1. Go to https://runpod.io → Pods → Deploy
#   2. Select "PyTorch" template, GPU: A100 40GB (or RTX 4090)
#   3. Set Container Disk to 50GB+
#   4. Open the pod's Jupyter Lab or Terminal
#   5. Clone the repo, cd into it, and run: bash deploy_runpod.sh

set -euo pipefail

echo "=========================================="
echo "  Image Generation - RunPod Setup"
echo "=========================================="

# --- Config ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-${SCRIPT_DIR}}"
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"
CONDA_ENV="${CONDA_ENV:-image-gen}"

# --- Step 1: System deps ---
echo ""
echo "[1/5] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq ffmpeg > /dev/null 2>&1

# --- Step 2: Find and activate conda, or fall back to venv ---
echo ""
echo "[2/5] Setting up Python environment..."
cd "${INSTALL_DIR}"

CONDA_SH=""
for candidate in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh /root/anaconda3/etc/profile.d/conda.sh /home/*/miniconda3/etc/profile.d/conda.sh /home/*/anaconda3/etc/profile.d/conda.sh; do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

USE_CONDA=false
if [ -n "${CONDA_SH}" ]; then
    echo "  Found conda at: ${CONDA_SH}"
    source "${CONDA_SH}"
    if conda env list | grep -q "^${CONDA_ENV} "; then
        echo "  Environment ${CONDA_ENV} already exists, skipping."
    else
        conda create -n "${CONDA_ENV}" python=3.10 -y -q
    fi
    conda activate "${CONDA_ENV}"
    USE_CONDA=true
else
    echo "  Conda not found, using venv."
    ${PYTHON} -m venv "${INSTALL_DIR}/.venv"
    source "${INSTALL_DIR}/.venv/bin/activate"
    PYTHON="$(which python)"
    PIP="$(which pip)"
fi

# --- Step 3: Install project (torch already in base image) ---
echo ""
echo "[3/5] Installing image-gen package..."
cd "${INSTALL_DIR}"
${PIP} install -e .

# HuggingFace token for faster downloads and higher rate limits
if [ -n "${HF_TOKEN:-}" ]; then
    echo "  Logging in to HuggingFace with HF_TOKEN..."
    huggingface-cli login --token "${HF_TOKEN}" 2>/dev/null || true
fi

# --- Step 4: Pre-download model weights ---
echo ""
echo "[4/5] Pre-downloading model weights (this takes a while)..."

echo "  Downloading Z-Image-Turbo FP8 transformer (~6.2GB)..."
${PYTHON} -c "
from huggingface_hub import hf_hub_download
hf_hub_download('T5B/Z-Image-Turbo-FP8', 'z-image-turbo-fp8-e4m3fn.safetensors')
print('FP8 transformer downloaded.')
"

echo "  Downloading Z-Image-Turbo components (tokenizer, text encoder, VAE, scheduler ~8.3GB)..."
${PYTHON} -c "
from transformers import Qwen2Tokenizer, Qwen3Model
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
Qwen2Tokenizer.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='tokenizer')
Qwen3Model.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='text_encoder')
AutoencoderKL.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='vae')
FlowMatchEulerDiscreteScheduler.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='scheduler')
print('Z-Image-Turbo components downloaded.')
"

echo "  Downloading Qwen-Image-Edit-2511 FP8 transformer (~20GB)..."
${PYTHON} -c "
from huggingface_hub import hf_hub_download
hf_hub_download('drbaph/Qwen-Image-Edit-2511-FP8', 'qwen_image_edit_2511_fp8_e4m3fn.safetensors')
print('FP8 transformer downloaded.')
"

echo "  Downloading Qwen-Image-Edit-2511 components (text encoder, VAE, tokenizer ~17GB)..."
${PYTHON} -c "
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, AutoProcessor
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
Qwen2Tokenizer.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='tokenizer')
AutoProcessor.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='processor')
Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='text_encoder')
AutoencoderKLQwenImage.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='vae')
FlowMatchEulerDiscreteScheduler.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='scheduler')
print('Qwen-Image-Edit-2511 components downloaded.')
"

# --- Step 5: Done ---
echo ""
echo "=========================================="
echo "  ✅ Setup complete!"
echo "=========================================="
echo ""
echo "  Activate the environment:"
if [ "${USE_CONDA}" = true ]; then
    echo "    conda activate ${CONDA_ENV}"
else
    echo "    source ${INSTALL_DIR}/.venv/bin/activate"
fi
echo ""
echo "  Text-to-image (FP8, ~14.5GB):"
echo "    image-gen 'A cat astronaut on the moon' --seed 42"
echo ""
echo "  Text-to-image with animation:"
echo "    image-gen-anim 'A cat astronaut on the moon' --seed 42 --output output.png --animation animation.mp4"
echo ""
echo "  Image-to-image restyling (FP8, ~37GB):"
echo "    i2i-gen -i photo.jpg -p 'turn it into a watercolor painting' --seed 42"
echo ""
echo "  Image-to-image with animation:"
echo "    i2i-gen-anim -i photo.jpg -p 'turn it into a watercolor painting' --seed 42"
echo ""
echo "  Object removal:"
echo "    remove-object -i photo.jpg -p 'person' --inpaint --inpainter lama"
echo ""
echo "  Output files will be in: ${INSTALL_DIR}/output/"
echo ""
echo "  Installed models:"
echo "    Z-Image-Turbo:  T5B/Z-Image-Turbo-FP8 + Tongyi-MAI/Z-Image-Turbo"
echo "    Qwen-Edit:      Qwen/Qwen-Image-Edit-2511"
echo "=========================================="
