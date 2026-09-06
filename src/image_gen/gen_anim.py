"""Z-Image-Turbo animation generator — text-to-image with step-by-step MP4."""

import argparse
from datetime import datetime

import numpy as np
import torch
from diffusers import (
    ZImagePipeline,
    ZImageTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen3Model, Qwen2Tokenizer
from PIL import Image

ORIG_REPO = "Tongyi-MAI/Z-Image-Turbo"
FP8_REPO = "T5B/Z-Image-Turbo-FP8"


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device):
    if device.type in ("mps", "cpu"):
        return torch.float32
    return torch.bfloat16


def load_fp8_pipeline(dtype, device):
    """Load pipeline with FP8 transformer (~14.5GB download)."""
    print("Loading tokenizer...")
    tokenizer = Qwen2Tokenizer.from_pretrained(ORIG_REPO, subfolder="tokenizer")

    print("Loading text encoder (~8GB)...")
    text_encoder = Qwen3Model.from_pretrained(
        ORIG_REPO, subfolder="text_encoder", torch_dtype=dtype,
    )

    print("Loading VAE (~0.17GB)...")
    vae = AutoencoderKL.from_pretrained(
        ORIG_REPO, subfolder="vae", torch_dtype=dtype,
    )

    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        ORIG_REPO, subfolder="scheduler",
    )

    print("Loading FP8 transformer (~6.2GB)...")
    from huggingface_hub import hf_hub_download
    fp8_path = hf_hub_download(
        repo_id=FP8_REPO,
        filename="z-image-turbo-fp8-e4m3fn.safetensors",
    )
    transformer = ZImageTransformer2DModel.from_single_file(
        fp8_path,
        torch_dtype=dtype,
    )

    pipe = ZImagePipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    return pipe


def load_full_pipeline(dtype, device):
    """Load full BF16 pipeline (~33GB download)."""
    print(f"Loading full model from {ORIG_REPO}...")
    pipe = ZImagePipeline.from_pretrained(
        ORIG_REPO,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return pipe


def latent_to_pil(latents, vae, device):
    """Decode a latent tensor to a PIL image."""
    latents = latents.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype(np.uint8)
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo animation generator")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt (unused for Turbo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: auto from datetime)")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps (9 = 8 DiT forwards)")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--animation", default="animation.mp4", help="Output animation path")
    parser.add_argument("--fps", type=int, default=1, help="Seconds per step in animation (1 = 1 fps)")
    parser.add_argument("--smooth", type=int, default=0, metavar="N",
                        help="Blend N intermediate frames between each step for smooth transitions (pixel-space)")
    parser.add_argument("--device", default=None, help="Force device: cuda, mps, or cpu")
    parser.add_argument("--full-model", action="store_true",
                        help="Download full BF16 model (~33GB) instead of FP8 (~14.5GB)")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device)

    if args.full_model:
        pipe = load_full_pipeline(dtype, device)
    else:
        pipe = load_fp8_pipeline(dtype, device)

    pipe.to(device)

    if args.seed is None:
        args.seed = int(datetime.now().strftime("%Y%m%d%H%M%S"))
        print(f"Auto seed: {args.seed}")

    print(f"Generating image: {args.steps} steps, seed={args.seed}, device={device}")
    print(f"  Prompt: {args.prompt[:80]}...")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Collect intermediate latents via callback
    intermediate_latents = []

    def save_intermediate(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        intermediate_latents.append(latents.detach().clone())
        return callback_kwargs

    print("Running transformer denoising steps...")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=generator,
        callback_on_step_end=save_intermediate,
    ).images[0]

    image.save(args.output)
    print(f"✅ Saved image to {args.output}")

    # Build animation
    print(f"Building animation from {len(intermediate_latents)} steps...")
    vae_device = next(pipe.vae.parameters()).device

    frames = []
    for i, latents in enumerate(intermediate_latents):
        print(f"  Decoding step {i + 1}/{len(intermediate_latents)}...")
        pil_image = latent_to_pil(latents, pipe.vae, vae_device)
        frames.append(np.array(pil_image))

    print("  Adding final image...")
    frames.append(np.array(image))

    # Smooth interpolation (pixel-space blending)
    if args.smooth > 0 and len(frames) > 1:
        n = args.smooth
        print(f"  Interpolating {n} blend frames between each step...")
        smooth_frames = []
        for i in range(len(frames) - 1):
            smooth_frames.append(frames[i])
            for t in range(1, n + 1):
                alpha = t / (n + 1)
                blended = ((1 - alpha) * frames[i].astype(np.float32) +
                           alpha * frames[i + 1].astype(np.float32))
                smooth_frames.append(blended.round().astype(np.uint8))
        smooth_frames.append(frames[-1])
        frames = smooth_frames
        print(f"  Total frames: {len(frames)} ({len(frames) // args.fps}s at {args.fps} fps)")

    # Write MP4
    import imageio

    fps = args.fps

    if args.smooth > 0:
        video_fps = fps * 10
        imageio.mimwrite(
            args.animation,
            frames,
            fps=video_fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],
        )
        total_seconds = len(frames) / video_fps
    else:
        repeated_frames = []
        for frame in frames:
            repeated_frames.extend([frame] * fps)
        imageio.mimwrite(
            args.animation,
            repeated_frames,
            fps=fps,
            codec="libx264",
            output_params=["-pix_fmt", "yuv420p"],
        )
        total_seconds = len(repeated_frames) / fps

    print(f"✅ Saved animation to {args.animation} ({len(frames)} frames, {total_seconds:.1f}s)")

    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {vram_gb:.1f} GB")


if __name__ == "__main__":
    main()
