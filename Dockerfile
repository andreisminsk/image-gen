# ---- Image Generation Docker Image ----
# Build:  docker build -t ghcr.io/andreisminsk/image-gen:1.0.0-cu128 .
# Run:    docker run --gpus all -v ./output:/app/output ghcr.io/andreisminsk/image-gen:1.0.0-cu128 image-gen "A cat astronaut on the moon"
# GPU required (CUDA). Use --gpus all or --gpus device=0.
#
# Base image: RunPod PyTorch with CUDA 12.8 + torch 2.8.0
# No torch upgrade needed — image-gen works with the base image's torch 2.8.0.

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /run/sshd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir -e .

# Pre-download models (optional — adds ~35GB but avoids first-run delay)
# Uncomment to include models in the image:
# RUN python -c "from huggingface_hub import hf_hub_download; \
#     hf_hub_download('T5B/Z-Image-Turbo-FP8', 'z-image-turbo-fp8-e4m3fn.safetensors')" && \
#     python -c "from transformers import Qwen2Tokenizer, Qwen3Model; from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler; \
#     Qwen2Tokenizer.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='tokenizer'); \
#     Qwen3Model.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='text_encoder'); \
#     AutoencoderKL.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='vae'); \
#     FlowMatchEulerDiscreteScheduler.from_pretrained('Tongyi-MAI/Z-Image-Turbo', subfolder='scheduler')" && \
#     python -c "from huggingface_hub import hf_hub_download; \
#     hf_hub_download('drbaph/Qwen-Image-Edit-2511-FP8', 'qwen_image_edit_2511_fp8_e4m3fn.safetensors')" && \
#     python -c "from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler; \
#     Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='text_encoder'); \
#     AutoencoderKLQwenImage.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='vae'); \
#     AutoProcessor.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='processor'); \
#     FlowMatchEulerDiscreteScheduler.from_pretrained('Qwen/Qwen-Image-Edit-2511', subfolder='scheduler')"

# Default output directory
RUN mkdir -p /app/output
VOLUME /app/output

# HF cache can be mounted for persistence
ENV HF_HOME=/root/.cache/huggingface

# Start SSH, then keep container alive
# RunPod users SSH in and run: image-gen "prompt" ...
# SCP is enabled for file transfer: scp root@<pod>:/app/output/image.png ./
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

# Start SSH server
/usr/sbin/sshd

# Keep container alive
exec "$@"
EOF
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sleep", "infinity"]
