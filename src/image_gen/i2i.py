"""Image-to-image restyling using Qwen-Image-Edit-2511 (FP8 transformer, ~37GB download)."""

import argparse
import os
import sys

import torch
from PIL import Image
from diffusers import (
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, AutoProcessor

ORIG_REPO = "Qwen/Qwen-Image-Edit-2511"
FP8_REPO = "drbaph/Qwen-Image-Edit-2511-FP8"


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device):
    if device in ("mps", "cpu"):
        return torch.float32
    return torch.bfloat16


def load_images(paths):
    images = []
    for path in paths:
        if not os.path.isfile(path):
            sys.exit(f"Error: input image not found: {path}")
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            sys.exit(f"Error opening {path}: {e}")
        images.append(img)
        print(f"  loaded {path}  ({img.size[0]}×{img.size[1]})")
    return images


def load_fp8_pipeline(dtype, device):
    """Load pipeline with FP8 transformer (~37GB download)."""
    print("Loading tokenizer...")
    tokenizer = Qwen2Tokenizer.from_pretrained(ORIG_REPO, subfolder="tokenizer")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(ORIG_REPO, subfolder="processor")

    print("Loading text encoder (~16.6GB)...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        ORIG_REPO, subfolder="text_encoder", torch_dtype=dtype,
    )

    print("Loading VAE (~1GB)...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        ORIG_REPO, subfolder="vae", torch_dtype=dtype,
    )

    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        ORIG_REPO, subfolder="scheduler",
    )

    print("Loading FP8 transformer (~20GB)...")
    from huggingface_hub import hf_hub_download
    fp8_path = hf_hub_download(
        repo_id=FP8_REPO,
        filename="qwen_image_edit_2511_fp8_e4m3fn.safetensors",
    )
    transformer = QwenImageTransformer2DModel.from_single_file(
        fp8_path,
        torch_dtype=dtype,
    )

    pipe = QwenImageEditPlusPipeline(
        tokenizer=tokenizer,
        processor=processor,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    return pipe


def load_full_pipeline(dtype, device):
    """Load full BF16 pipeline (~57GB download)."""
    print(f"Loading full model from {ORIG_REPO}...")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        ORIG_REPO,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return pipe


def main():
    parser = argparse.ArgumentParser(
        description="Restyle/regenerate images with Qwen-Image-Edit-2511"
    )
    parser.add_argument("-i", "--image", nargs="+", required=True,
                        help="One or more input image paths")
    parser.add_argument("-p", "--prompt", required=True,
                        help="Editing/restyling instruction")
    parser.add_argument("-o", "--output", default="output.png",
                        help="Output image path (default: output.png)")
    parser.add_argument("--negative-prompt", default=" ",
                        help="Negative prompt (default: single space)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of inference steps (default: 20)")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="True CFG scale (default: 4.0)")
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="Guidance scale (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--num-images", type=int, default=1,
                        help="Number of images to generate (default: 1)")
    parser.add_argument("--device", default=None,
                        help="Force device: cuda, mps, or cpu (auto-detected if omitted)")
    parser.add_argument("--full-model", action="store_true",
                        help="Download full BF16 model (~57GB) instead of FP8 (~37GB)")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device)
    print(f"Device: {device}")

    print("Loading images...")
    images = load_images(args.image)

    if args.full_model:
        pipe = load_full_pipeline(dtype, device)
    else:
        pipe = load_fp8_pipeline(dtype, device)

    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    print("Pipeline ready.\n")

    print(f"Prompt: {args.prompt}")
    print(f"Steps={args.steps}  CFG={args.cfg}  Guidance={args.guidance_scale}  Seed={args.seed}\n")

    gen = torch.manual_seed(args.seed)
    with torch.inference_mode():
        output = pipe(
            image=images,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            true_cfg_scale=args.cfg,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images,
            generator=gen,
        )

    if args.num_images == 1:
        output.images[0].save(args.output)
        print(f"Saved: {os.path.abspath(args.output)}")
    else:
        base, ext = os.path.splitext(args.output)
        for idx, img in enumerate(output.images):
            path = f"{base}_{idx}{ext}"
            img.save(path)
            print(f"Saved: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
