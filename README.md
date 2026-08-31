# Image Generation Toolkit

Two diffusion model pipelines for text-to-image and image-to-image generation on Apple Silicon (MPS), CUDA, or CPU.

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
python image-gen.py "A cat astronaut on the moon"
```

### Full BF16 Model (33GB download, best quality)

```bash
python image-gen.py "A cat astronaut on the moon" --full-model
```

### With Options

```bash
python image-gen.py "A sunset over mountains" \
  --seed 123 \
  --output sunset.png \
  --width 768 \
  --height 768 \
  --steps 9
```

### Animation (step-by-step denoising video)

```bash
pip install imageio imageio-ffmpeg  # extra dependency for video

python image-gen-anim.py "A cat astronaut on the moon"
# Produces: output.png + animation.mp4

# Customize animation:
python image-gen-anim.py "A sunset over mountains" \
  --output sunset.png \
  --animation sunset_anim.mp4 \
  --fps 2
```

Each denoising step is shown for 1 second (adjustable with `--fps`). The animation reveals how the model progressively refines noise into a coherent image.

### All Options — image-gen.py

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

### All Options — image-gen-anim.py

Same as `image-gen.py`, plus:

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
python i2i-gen.py -i photo.png -p "turn it into a watercolor painting"

# Multiple input images (compositing)
python i2i-gen.py -i bear1.png bear2.png -p "both bears facing each other in a park" -o merged.png

# With animation
pip install imageio imageio-ffmpeg
python i2i-gen-anim.py -i photo.png -p "make it anime style" --animation anim.mp4
```

### Options — i2i-gen.py

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

### Options — i2i-gen-anim.py

Same as `i2i-gen.py`, plus:

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
python i2i-gen.py -i photo.png -p "a pirate on a harbor dock" \
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

## Files

| File | Description |
|------|-------------|
| `image-gen.py` | Text-to-image generation (Z-Image-Turbo) |
| `image-gen-anim.py` | Text-to-image + denoising animation (Z-Image-Turbo) |
| `i2i-gen.py` | Image-to-image restyling (Qwen-Image-Edit-2511) |
| `i2i-gen-anim.py` | Image-to-image + denoising animation (Qwen-Image-Edit-2511) |
| `requirements.txt` | Python dependencies |

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
- **torchao warnings** — Harmless; torchao is not used on MPS/CPU

## License

This script is provided as-is. The Z-Image-Turbo model is licensed under [Apache 2.0](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
