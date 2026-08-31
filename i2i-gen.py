#!/usr/bin/env python3
"""
i2i-gen.py — Image-to-image generation/restyling using Qwen-Image-Edit-2511.

Usage:
    python i2i-gen.py --image photo.png --prompt "turn it into a watercolor painting"
    python i2i-gen.py -i img1.png img2.png -p "merge both bears into a park scene" -o result.png
    python i2i-gen.py -i input.jpg -p "make it anime style" --steps 30 --cfg 4.0 --seed 42
"""
import argparse
import os
import sys

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Restyle/regenerate images with Qwen-Image-Edit-2511"
    )
    p.add_argument("-i", "--image", nargs="+", required=True,
                   help="One or more input image paths")
    p.add_argument("-p", "--prompt", required=True,
                   help="Editing/restyling instruction")
    p.add_argument("-o", "--output", default="output.png",
                   help="Output image path (default: output.png)")
    p.add_argument("--negative-prompt", default=" ",
                   help="Negative prompt (default: single space)")
    p.add_argument("--steps", type=int, default=20,
                   help="Number of inference steps (default: 20)")
    p.add_argument("--cfg", type=float, default=4.0,
                   help="True CFG scale (default: 4.0)")
    p.add_argument("--guidance-scale", type=float, default=1.0,
                   help="Guidance scale (default: 1.0)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--num-images", type=int, default=1,
                   help="Number of images to generate (default: 1)")
    p.add_argument("--device", default=None,
                   help="Force device: cuda, mps, or cpu (auto-detected if omitted)")
    return p.parse_args()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_images(paths: list[str]) -> list[Image.Image]:
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


def main():
    args = parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    print("Loading images...")
    images = load_images(args.image)

    print("Loading Qwen-Image-Edit-2511 pipeline (first run downloads ~20 GB)...")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=dtype,
    )
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
