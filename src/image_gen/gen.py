"""Z-Image-Turbo text-to-image generation (FP8, ~14.5GB download)."""

import argparse
from datetime import datetime

import torch
from diffusers import (
    ZImagePipeline,
    ZImageTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen3Model, Qwen2Tokenizer

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


def main():
    parser = argparse.ArgumentParser(description="Z-Image-Turbo image generation (reduced download)")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt (unused for Turbo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: auto from datetime)")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--steps", type=int, default=9, help="Inference steps (9 = 8 DiT forwards)")
    parser.add_argument("--output", default="output.png", help="Output file path")
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

    print("Running transformer denoising steps...")
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt or None,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]

    image.save(args.output)
    print(f"✅ Saved to {args.output}")

    if device.type == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {vram_gb:.1f} GB")


if __name__ == "__main__":
    main()
