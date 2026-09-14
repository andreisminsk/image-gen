# Image Generation Toolkit

Three diffusion model pipelines for text-to-image, image-to-image, and object removal on Apple Silicon (MPS), CUDA, or CPU.

## 1. Z-Image-Turbo — Text-to-Image (FP8, ~14.5GB)

Generate high-quality images using the [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) model with an optimized download strategy — **~14.5GB** instead of ~33GB.

## 2. Qwen-Image-Edit-2511 — Image-to-Image (~20GB)

Restyle or transform an existing image using [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). Provide a source image + a text instruction (e.g. *"turn it into a watercolor painting"*) and the model regenerates the image accordingly. Supports multiple input images for compositing.

## How It Works

### Theory: How Diffusion Image Generation Works

Diffusion models generate images by starting from pure noise and gradually refining it into a coherent image. Think of it like sculpting — you start with a rough block and iteratively chip away until the form emerges.

**The forward process** (training only — not used during generation) gradually adds Gaussian noise to a clean image over many steps until it becomes pure random static. The model learns to predict the noise that was added at each step.

**The reverse process** (what the script does during generation) starts from pure random noise — controlled by the `--seed` parameter — and at each step, the transformer predicts "what noise is left" and the scheduler removes it. After 8 steps (for Turbo), the latents are clean enough to decode into a final image via the VAE.

```
Random noise → Step 1 (remove some noise) → Step 2 → ... → Step 8 → Clean latents → VAE → Final image
```

Z-Image-Turbo uses a **flow-matching** variant of diffusion, which learns a direct path (vector field) from noise to data in fewer steps. Combined with **distillation** (Decoupled-DMD), it achieves high quality in just 8 steps instead of the typical 50+. The key insight of distillation is that the model learned to take **shortcuts** — instead of 50+ tiny steps, it makes 8 large jumps that land directly on a good image.

**Key concepts:**

- **Latent space**: Instead of working directly with pixels (e.g., 1024×1024×3 = 3M values), the VAE compresses images into a much smaller latent representation (e.g., 128×128×16 = 262K values). The diffusion process operates in this compressed space, making it far more efficient.

- **Timestep conditioning**: At each step, the model knows "how noisy" the current image is (the timestep). This tells it how aggressively to denoise — early steps make large structural changes, later steps refine details.

- **CFG (Classifier-Free Guidance)**: A technique that amplifies the model's adherence to the prompt by running it with and without the prompt text, then scaling the difference. Z-Image-Turbo **does not use CFG** (`guidance_scale=0.0`) because the distillation process already bakes in prompt-following capability.

### Component Roles

The pipeline has four components, each with a distinct role:

#### 1. Tokenizer (`Qwen2Tokenizer`)
Converts the text prompt into a sequence of integer tokens. This is the same tokenizer used by the Qwen3 language model, supporting both English and Chinese text.

```
"A cat on the moon" → [3847, 278, 8976, 389, 276, 18542, 382]
```

#### 2. Text Encoder (`Qwen3Model`)
A 3.4B-parameter language model that converts token IDs into rich semantic embeddings. These embeddings capture the *meaning* of the prompt — not just individual words, but their relationships, attributes, and context.

```
Token IDs → Text Encoder → [sequence_length × hidden_dim] embeddings
```

This is the largest component after the transformer. It's a full Qwen3 model (similar to GPT-style transformers) that understands language deeply, giving the image generator a rich understanding of what to draw.

#### 3. Transformer (`ZImageTransformer2DModel`)
The core image generator — a 6B-parameter **Single-Stream DiT** (S3-DiT). This is where the magic happens:

- Takes two inputs: **noisy latent image** and **text embeddings**
- In a single-stream architecture, text tokens and image tokens are **concatenated into one sequence** — unlike dual-stream models (e.g., Flux) that process them separately
- At each timestep, predicts the "clean" image given the current noise level and the text prompt
- Runs 8 times (8 DiT forward passes), progressively refining the image

```
Noisy latents + Text embeddings → Transformer × 8 steps → Denoised latents
```

This is the most computationally expensive component and the primary target for quantization (FP8 reduces it from ~12GB to ~6GB).

#### 4. VAE (`AutoencoderKL`)
The **Variational Autoencoder** bridges between pixel space and latent space:

- **Encoder** (not used during generation): Compresses an image into a compact latent representation (~8× spatial compression)
- **Decoder** (used during generation): Converts the final denoised latents back into a full-resolution pixel image

```
Denoised latents [128×128×16] → VAE Decoder → Output image [1024×1024×3]
```

The VAE is small (~168MB) and runs only once at the end. It's responsible for the fine pixel-level details and color accuracy in the final output.

#### 5. Scheduler (`FlowMatchEulerDiscreteScheduler`)
Orchestrates the denoising process — determines the noise schedule and how to combine the transformer's predictions at each step. For Z-Image-Turbo, this is a flow-matching Euler scheduler that defines the 8-step trajectory from noise to image.

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Text Prompt                          │
│              "A cat astronaut on the moon"               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │    Tokenizer     │  Text → Token IDs
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Text Encoder    │  Token IDs → Embeddings (~8GB)
              │   (Qwen3 3.4B)  │
              └────────┬────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│           Random Noise (latent space)                    │
│           [1 × 16 × 128 × 128]                          │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌─────────────▼──────────────┐
          │   Transformer × 8 steps    │  Denoising loop
          │   (ZImageTransformer 6B)   │  Each step refines
          │   + Text embeddings         │  the latent image
          └─────────────┬──────────────┘
                       ▼
              ┌─────────────────┐
              │   VAE Decoder    │  Latents → Pixels
              │   (AutoencoderKL)│
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │   Output Image   │  1024 × 1024 × 3
              └─────────────────┘
```

### Reduced Download Strategy

Instead of downloading the full BF16 model (~33GB), the script loads components separately:

| Component | Source | Download Size |
|-----------|--------|---------------|
| Transformer (FP8) | `T5B/Z-Image-Turbo-FP8` | ~6.2 GB |
| Text encoder (Qwen3) | `Tongyi-MAI/Z-Image-Turbo` | ~8.1 GB |
| VAE | `Tongyi-MAI/Z-Image-Turbo` | ~0.17 GB |
| Tokenizer + Scheduler | `Tongyi-MAI/Z-Image-Turbo` | <1 MB |
| **Total** | | **~14.5 GB** |

The FP8 transformer is downloaded via `huggingface_hub.hf_hub_download` and loaded with `from_single_file`. The remaining components are loaded from the original repo using `from_pretrained` with subfolders.

On MPS (Apple Silicon), FP8 weights are cast to float32 for numerical stability. Quality matches the full model while saving ~18GB in download size.

### Model: Z-Image-Turbo

Z-Image-Turbo is a 6B-parameter diffusion transformer that generates images in just **8 inference steps**. It was distilled from Z-Image using [Decoupled-DMD](https://arxiv.org/abs/2511.22677) and refined with [DMDR](https://arxiv.org/abs/2511.13649).

Key properties:
- **No CFG needed** — always use `guidance_scale=0.0`
- **8 steps** — set `num_inference_steps=9` (9 = 8 DiT forward passes + 1 scheduler step)
- **Bilingual** — handles English and Chinese text rendering well

### Architecture

```
Prompt → Qwen3 (text encoder) → text embeddings
                                      ↓
                              ZImageTransformer2DModel (DiT)
                                      ↓
                              AutoencoderKL (VAE decoder)
                                      ↓
                                   Output image
```

The model uses a **single-stream DiT** (S3-DiT) architecture where text and image tokens are concatenated into one sequence.

### Memory Management

- **CUDA**: Full pipeline on GPU. Use `--fp8` for FP8 quantization via torchao (requires compute capability ≥8.9).
- **MPS/CPU**: Full pipeline loaded to device in float32 (~28GB). FP8 weights are cast to float32, so memory usage is similar to full model but download is ~14.5GB instead of ~33GB.

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** (CUDA) or **Apple Silicon** (MPS) or **CPU** (very slow)
- **16GB+ RAM** (MPS/CPU) or **10GB+ VRAM** (CUDA with FP8)
- **Git** (for installing diffusers from source)

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate    # Windows

# Install PyTorch (adjust for your platform)
# macOS:
pip install torch
# Linux/CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install diffusers from source (includes ZImagePipeline)
pip install git+https://github.com/huggingface/diffusers

# Install remaining dependencies
pip install transformers accelerate sentencepiece
```

> **Why install diffusers from Git?** The `ZImagePipeline` class was recently merged but may not be in the latest PyPI release. Once it is, you can simply `pip install diffusers --upgrade`.

## Usage

### Basic (FP8, smaller download)

```bash
image-gen "A cat astronaut on the moon"
```

### Full BF16 Model (33GB download, best quality)

```bash
image-gen "A cat astronaut on the moon" --full-model
```

### With Options

```bash
image-gen "A sunset over mountains" \
  --seed 123 \
  --output sunset.png \
  --width 768 \
  --height 768 \
  --steps 9
```

### Animation (step-by-step denoising video)

```bash
pip install imageio imageio-ffmpeg  # extra dependency for video

image-gen-anim "A cat astronaut on the moon"
# Produces: output.png + animation.mp4

# Customize animation:
image-gen-anim "A sunset over mountains" \
  --output sunset.png \
  --animation sunset_anim.mp4 \
  --fps 2
```

Each denoising step is shown for 1 second (adjustable with `--fps`). The animation reveals how the model progressively refines noise into a coherent image.

### All Options — image-gen

| Flag | Default | Description |
|------|---------|-------------|
| `prompt` | *(required)* | Text prompt for image generation |
| `--negative-prompt` | `""` | Negative prompt (unused for Turbo) |
| `--seed` | auto (datetime) | Random seed for reproducibility. Auto-generated from date/time if not specified. Pass the printed seed value to reproduce an image. |
| `--height` | `1024` | Image height in pixels |
| `--width` | `1024` | Image width in pixels |
| `--steps` | `9` | Inference steps (9 = 8 DiT forwards) |

#### How `--steps` Works

The `--steps` parameter controls how many **denoising iterations** the diffusion process runs. The model starts from pure random noise and progressively removes noise over multiple steps — each step runs the transformer once to predict a cleaner version of the image. More steps = more refinement, but diminishing returns.

**For Z-Image-Turbo specifically:**

- The model was **distilled** to work optimally at **8 DiT forward passes**
- The scheduler uses `num_inference_steps=9` internally, which equals **8 transformer calls + 1 initial noise step**
- So `--steps 9` is the sweet spot — that's what the model was trained for

| `--steps` | Result |
|-----------|--------|
| 4-5 | Undercooked — blurry, missing details, may have artifacts |
| **9** | **Optimal** — what the model was distilled for |
| 15-20 | Slightly more refined, but marginal improvement |
| 30+ | Wasted compute — no real quality gain, the model converges by step 9 |

Fewer steps skip parts of the learned noise-to-image trajectory, producing incomplete images. More steps add diminishing refinement beyond what the model was trained for. Leave it at `9` unless experimenting.
| `--output` | `output.png` | Output file path |
| `--device` | auto | Force device: `cuda`, `mps`, or `cpu` |
| `--full-model` | `False` | Download full BF16 model (~33GB) instead of FP8 (~14.5GB) |

### All Options — image-gen-anim

Same as `image-gen`, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--animation` | `animation.mp4` | Output animation file path |
| `--fps` | `1` | Frames per step (1 = 1 second per step, 2 = 0.5s per step) |
| `--smooth N` | `0` | Blend N intermediate frames between each step for smooth transitions (pixel-space) |

### CUDA-only Options

| Flag | Description |
|------|-------------|
| `--fp8` | Enable torchao FP8 quantization at runtime (CUDA with compute ≥8.9 only) |

## Image-to-Image: Qwen-Image-Edit-2511

### Usage

```bash
# Basic restyle
i2i-gen -i photo.png -p "turn it into a watercolor painting"

# Multiple input images (compositing)
i2i-gen -i bear1.png bear2.png -p "both bears facing each other in a park" -o merged.png

# With animation
pip install imageio imageio-ffmpeg
i2i-gen-anim -i photo.png -p "make it anime style" --animation anim.mp4
```

### Options — i2i-gen

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--image` | *(required)* | One or more input image paths |
| `-p` / `--prompt` | *(required)* | Editing/restyling instruction |
| `-o` / `--output` | `output.png` | Output image path |
| `--negative-prompt` | `" "` | Negative prompt — concepts to steer away from (see below) |
| `--steps` | `20` | Inference steps (fewer = faster) |
| `--cfg` | `4.0` | True CFG scale |
| `--guidance-scale` | `1.0` | Guidance scale |
| `--seed` | `0` | Random seed |
| `--num-images` | `1` | Number of images to generate |
| `--device` | auto | Force `cuda`, `mps`, or `cpu` |

### Options — i2i-gen-anim

Same as `i2i-gen`, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--animation` | `animation.mp4` | Output animation path |
| `--fps` | `1` | Seconds per step (1 = 1s/step) |
| `--smooth N` | `0` | Blend N intermediate frames between steps |

### How `--negative-prompt` Works

With `--cfg` above 1 (default 4.0), the model runs each denoising step twice — once with your prompt, once with the negative prompt — then extrapolates *away* from the negative direction:

```
prediction = negative + cfg × (positive − negative)
```

The default `" "` (empty) just enables CFG without steering. Putting concepts in the negative prompt actively suppresses them:

```bash
i2i-gen -i photo.png -p "a pirate on a harbor dock" \
  --negative-prompt "ships, boats, disproportionate ship, weapon, sword, gun, pistol, knife, holding weapon"
```

**Tips:**
- List concrete nouns — "weapon" alone is vague; naming sword/gun/pistol works better
- Higher `--cfg` (5–6) strengthens suppression but can add artifacts; lower (2–3) weakens it
- Phrase the positive prompt affirmatively too: *"a pirate standing on a harbor dock, empty hands, calm sea in the background, no ships"*
- Negative prompts steer rather than guarantee — combine both approaches for best results

### How the Animation Works

The script captures the **packed latent tensor** after each denoising step via the pipeline's `callback_on_step_end` hook. To turn each latent into a viewable frame, three conversions are needed:

1. **Unpack** — the transformer operates on packed latents `[1, num_patches, channels×4]` (2×2 patches flattened into the sequence). `pipe._unpack_latents()` restores them to spatial form `[1, 16, 1, h, w]`.
2. **Decode** — the VAE decoder converts latents to pixels. The Qwen-Image VAE is a 3D video-style VAE that expects 5D input and outputs 5D `(batch, channels, frames, h, w)` — the frames dimension is squeezed out.
3. **Post-process** — normalize to `[0, 1]`, convert to uint8, and collect as a frame.

**Important:** the pipeline resizes the input image to ~1024×1024 area internally (e.g. a 3904×5184 photo becomes 1184×896 latents). The unpack step must use these *pipeline* dimensions, not the original image dimensions, or the reshape will fail.

The final decoded image is appended as the last frame, then all frames are written to MP4 with `imageio`. With `--smooth N`, pixel-space blend frames are interpolated between steps for a gradual transition effect.

## Object Removal: Segment → Select → Inpaint

### Overview

`remove-object` removes unwanted objects from photos in three stages:

1. **Segmentation** — detect object masks using Mask R-CNN (default, detects people) or SAM (point/grid prompts)
2. **Mask selection** — pick the best mask via CLIP semantic matching, manual `--mask-index`, or area/IoU fallback
3. **Inpainting** — fill the masked region using SDXL (context-aware, slower) or LaMa (fast, no hallucination)

### Usage

```bash
# Dry run — detect and save the mask only (no inpainting)
remove-object -i photo.jpg -p "man in blue"

# Full removal with LaMa (fast, no hallucination)
remove-object -i photo.jpg -p "man in blue" --mask-index 0 --inpaint --inpainter lama

# Full removal with SDXL (context-aware, slower)
remove-object -i photo.jpg -p "man in blue" --mask-index 0 --inpaint --inpainter sdxl

# SAM with point prompt
remove-object -i photo.jpg -p "person" --point 700,512 --inpaint

# SDXL with custom fill prompt (describe what should replace the object)
remove-object -i photo.jpg -p "man in blue" --inpaint --inpainter sdxl --inpaint-prompt "clear blue sky"
```

### How It Works

**Segmentation** produces candidate masks:

- **Mask R-CNN** (`--maskrcnn`, default) — detects COCO person instances. For small/distant figures where the mask misses limbs, it automatically fills the bounding box if mask coverage < 60%.
- **SAM** (`--sam`) — Segment Anything Model. With `--point x,y`, segments the object near that point. Without a point, uses a 3×3 grid to auto-segment the scene.

**Mask selection** picks the best candidate:

- **CLIP** (default) — scores each masked region against the text prompt and picks the best semantic match. This ensures the mask corresponds to what you described, not just the largest object.
- **`--mask-index N`** — manual override, pick the Nth mask directly.
- **`--no-clip --prefer area|iou`** — skip CLIP and fall back to largest area or highest IoU.

**Inpainting** fills the hole:

- **SDXL** (`--inpainter sdxl`, default) — Stable Diffusion XL inpainting. Context-aware, can hallucinate realistic content. Use `--inpaint-prompt` to describe what should fill the hole (leave empty to infer from surroundings). ⚠️ Do **not** pass the removal prompt here — that would regenerate the object you're trying to remove.
- **LaMa** (`--inpainter lama`) — Large Mask Inpainting. Fast, no hallucination, good for clean removals. Supports `--dilate` to expand the mask for better coverage.

**Mask post-processing** — holes in the mask interior are filled via border flood-fill, and a red overlay preview is saved alongside the binary mask.

### Options — remove-object

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--image` | *(required)* | Input image path |
| `-p` / `--prompt` | *(required)* | Text description of object to remove |
| `-o` / `--output` | *(input dir)* | Output directory |
| `--point` | None | `x,y` point prompt for SAM |
| `--mask-index` | None | Force a specific mask (0-based) |
| `--max-dim` | `1024` | Downscale longest side |
| `--no-clip` | `False` | Skip CLIP, use area/IoU fallback |
| `--prefer` | `area` | Fallback mode: `area` or `iou` |
| `--sam` | `False` | Use SAM instead of Mask R-CNN |
| `--all-classes` | `False` | Mask R-CNN: detect all COCO classes |
| `--inpaint` | `False` | Run inpainting after segmentation |
| `--inpainter` | `sdxl` | Inpainting model: `sdxl` or `lama` |
| `--inpaint-prompt` | `""` | What to fill the hole with (SDXL only) |
| `--inpaint-model` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | SDXL model ID |
| `--steps` | `30` | SDXL inference steps |
| `--guidance-scale` | `3.0` | SDXL guidance scale |
| `--feather` | `8` | Mask feather radius (px) |
| `--dilate` | `10` | LaMa mask dilation (px) |
| `--seed` | `42` | Random seed for SDXL |
| `--device` | auto | Force `cuda`, `mps`, or `cpu` |

## Installation

```bash
pip install -e .
```

This installs five console commands: `image-gen`, `image-gen-anim`, `i2i-gen`, `i2i-gen-anim`, `remove-object`.

You can also run modules directly without installing:

```bash
python -m image_gen.gen "A cat on the moon"
python -m image_gen.remove -i photo.jpg -p "person" --inpaint
```

### Proxy Scripts (optional)

After `pip install -e .`, you can create shell proxy scripts in `~/.local/bin` so the commands are available without activating the venv:

```bash
# macOS / Linux
./install.sh

# Windows PowerShell
./install.ps1
```

This creates wrapper scripts for all five commands that point to the venv executables. To remove them:

```bash
# macOS / Linux
./uninstall.sh

# Windows PowerShell
./uninstall.ps1
```

## Docker

A Docker image is available for CUDA GPU environments (RunPod, cloud instances, etc.):

```bash
# Build
docker build -t ghcr.io/andreisminsk/image-gen:1.0.0-cu128 .

# Run (GPU required)
docker run --gpus all -v ./output:/app/output ghcr.io/andreisminsk/image-gen:1.0.0-cu128 \
    image-gen "A cat astronaut on the moon" --seed 42

# With HuggingFace token (optional — for higher download rate limits)
docker run --gpus all -e HF_TOKEN=hf_xxx -v ./output:/app/output \
    ghcr.io/andreisminsk/image-gen:1.0.0-cu128 image-gen "A cat on the moon"
```

Or with `docker-compose.yml`:

```bash
# Pass HF_TOKEN from your environment (optional)
export HF_TOKEN=hf_xxx
docker compose up -d
docker compose exec image-gen image-gen "A cat on the moon" --seed 42
```

The image is based on RunPod's PyTorch base (CUDA 12.8, torch 2.8.0) and includes SSH for RunPod access. Models are downloaded on first run (~14.5GB for Z-Image-Turbo, ~20GB for Qwen-Image-Edit). To pre-bake models into the image (~35GB larger), uncomment the pre-download section in the Dockerfile.

CI builds and pushes to `ghcr.io/andreisminsk/image-gen` on every push to `main`.

## RunPod Deployment

### Option A: Docker Image

Use the pre-built Docker image on a RunPod PyTorch pod:

1. Deploy a RunPod pod with the PyTorch template (A100 40GB+ recommended)
2. Pull and run the image:

```bash
docker run --gpus all -d \
    -e HF_TOKEN=hf_xxx \
    -v /app/output:/app/output \
    -v hf-cache:/root/.cache/huggingface \
    ghcr.io/andreisminsk/image-gen:1.0.0-cu128
```

3. SSH in and run commands:

```bash
image-gen "A cat astronaut on the moon" --seed 42
scp root@<pod>:/app/output/output.png ./
```

### Option B: Manual Setup (no Docker)

Run `deploy_runpod.sh` on a fresh RunPod PyTorch pod to install everything from scratch:

```bash
git clone <repo-url> && cd image-gen
bash deploy_runpod.sh
```

The script:
1. Installs system dependencies (ffmpeg)
2. Sets up a conda env or venv
3. Installs the `image-gen` package
4. Pre-downloads all model weights (~35GB: Z-Image-Turbo FP8 + Qwen-Image-Edit-2511)

Set `HF_TOKEN` before running for faster downloads:

```bash
export HF_TOKEN=hf_xxx
bash deploy_runpod.sh
```

After setup, activate the environment and run:

```bash
conda activate image-gen   # or: source .venv/bin/activate
image-gen "A cat astronaut on the moon" --seed 42
```

## Files

| File | Command | Description |
|------|---------|-------------|
| `src/image_gen/gen.py` | `image-gen` | Text-to-image generation (Z-Image-Turbo) |
| `src/image_gen/gen_anim.py` | `image-gen-anim` | Text-to-image + denoising animation (Z-Image-Turbo) |
| `src/image_gen/i2i.py` | `i2i-gen` | Image-to-image restyling (Qwen-Image-Edit-2511) |
| `src/image_gen/i2i_anim.py` | `i2i-gen-anim` | Image-to-image + denoising animation (Qwen-Image-Edit-2511) |
| `src/image_gen/remove.py` | `remove-object` | Prompt-driven object removal (segment → select → inpaint) |
| `pyproject.toml` | | Package config and entry points |
| `requirements.txt` | | Python dependencies (for pip install without package) |

## Scaling Up

| Goal | Change |
|------|--------|
| Higher quality, more VRAM | Use `--full-model` for full BF16 |
| More creative/diverse images | Switch to `Tongyi-MAI/Z-Image`, set `guidance_scale=3.5`, `steps=50` |
| Faster inference (CUDA) | Uncomment `pipe.transformer.set_attention_backend("flash")` or `pipe.transformer.compile()` |

## Troubleshooting

- **`ZImagePipeline not found`** — Install diffusers from source: `pip install git+https://github.com/huggingface/diffusers`
- **`ImportError: float8_weight_only`** — FP8 via torchao requires CUDA with compute capability ≥8.9. Remove `--fp8` flag.
- **CUDA OOM** — Use `--fp8`, reduce resolution to `768x768`, or add `pipe.enable_model_cpu_offload()` in the script
- **MPS OOM** — Reduce resolution to `768x768`. The full pipeline loads in float32 on MPS (~28GB). If still OOM, try `--device cpu` (very slow but works).
- **Slow first run** — Normal; model weights are downloaded and cached in `~/.cache/huggingface/`
- **`SimpleLama not found`** — Install: `pip install simple-lama-inpainting` (only needed for `--inpainter lama`)
- **`No module named 'torchvision'`** — Install: `pip install torchvision` (needed for Mask R-CNN segmentation)
- **torchao warnings** — Harmless; torchao is not used on MPS/CPU

## License

This script is provided as-is. The Z-Image-Turbo model is licensed under [Apache 2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
