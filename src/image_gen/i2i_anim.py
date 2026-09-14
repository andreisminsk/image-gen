"""Image-to-image restyling with Qwen-Image-Edit-2511 (GGUF) + MP4 animation."""

import argparse
import math
import os
import sys

import numpy as np
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
GGUF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"


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


def load_gguf_pipeline(dtype, device, quant="Q8_0"):
    """Load pipeline with GGUF-quantized transformer (~38GB download)."""
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

    print(f"Loading GGUF transformer ({quant})...")
    from huggingface_hub import hf_hub_download
    from diffusers import GGUFQuantizationConfig
    gguf_path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=f"qwen-image-edit-2511-{quant}.gguf",
    )
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        config=ORIG_REPO,
        subfolder="transformer",
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )

    # Components stay on CPU — main() uses enable_model_cpu_offload().
    # GGUF transformer stays quantized in RAM, dequantized per-layer on GPU.
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


def latent_to_pil(latents, pipe, height, width, device):
    """Decode a packed latent tensor to a PIL image."""
    latents = latents.to(device=device, dtype=pipe.vae.dtype)
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(device=device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    if image.dim() == 5:
        image = image.squeeze(2)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype(np.uint8)
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(
        description="Restyle images with Qwen-Image-Edit-2511 + animation output"
    )
    parser.add_argument("-i", "--image", nargs="+", required=True,
                        help="One or more input image paths")
    parser.add_argument("-p", "--prompt", required=True,
                        help="Editing/restyling instruction")
    parser.add_argument("-o", "--output", default="output.png",
                        help="Output image path (default: output.png)")
    parser.add_argument("--animation", default="animation.mp4",
                        help="Output animation path (default: animation.mp4)")
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
    parser.add_argument("--fps", type=int, default=1,
                        help="Seconds per step in animation (1 = 1 fps, default: 1)")
    parser.add_argument("--intro", type=int, default=10, metavar="N",
                        help="Cross-fade N frames from a ghosted source image into the first denoising step (0 = disable, default: 10)")
    parser.add_argument("--smooth", type=int, default=0, metavar="N",
                        help="Blend N intermediate frames between steps for smooth transitions")
    parser.add_argument("--device", default=None,
                        help="Force device: cuda, mps, or cpu (auto-detected if omitted)")
    parser.add_argument("--quant", default="Q8_0",
                        choices=["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"],
                        help="GGUF quantization (default: Q8_0 ~20GB; Q4_0 ~11GB)")
    parser.add_argument("--full-model", action="store_true",
                        help="Download full BF16 model (~57GB) instead of GGUF (~38GB)")
    args = parser.parse_args()

    device = pick_device(args.device)
    dtype = pick_dtype(device)
    print(f"Device: {device}")

    print("Loading images...")
    images = load_images(args.image)

    if args.full_model:
        pipe = load_full_pipeline(dtype, device)
        pipe.to(device)
    else:
        pipe = load_gguf_pipeline(dtype, device, args.quant)
        print("Enabling CPU offload (components swap in/out of VRAM)...")
        pipe.enable_model_cpu_offload()

    pipe.set_progress_bar_config(disable=None)
    print("Pipeline ready.\n")

    print(f"Prompt: {args.prompt}")
    print(f"Steps={args.steps}  CFG={args.cfg}  Guidance={args.guidance_scale}  Seed={args.seed}\n")

    gen = torch.manual_seed(args.seed)

    intermediate_latents = []

    def save_intermediate(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        intermediate_latents.append(latents.detach().clone())
        return callback_kwargs

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
            callback_on_step_end=save_intermediate,
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

    # Build animation
    print(f"\nBuilding animation from {len(intermediate_latents)} steps...")
    vae_device = next(pipe.vae.parameters()).device

    img_w, img_h = images[0].size
    ratio = img_w / img_h
    calc_w = math.sqrt(1024 * 1024 * ratio)
    calc_h = calc_w / ratio
    calc_w = round(calc_w / 32) * 32
    calc_h = round(calc_h / 32) * 32
    multiple_of = pipe.vae_scale_factor * 2
    img_h = calc_h // multiple_of * multiple_of
    img_w = calc_w // multiple_of * multiple_of
    print(f"  pipeline dimensions: {img_h}×{img_w}")

    frames = []
    for i, latents in enumerate(intermediate_latents):
        print(f"  Decoding step {i + 1}/{len(intermediate_latents)}...")
        pil_image = latent_to_pil(latents, pipe, img_h, img_w, vae_device)
        frames.append(np.array(pil_image))

    print("  Adding final image...")
    frames.append(np.array(output.images[0]))

    # Ghosted source intro with cross-fade
    if args.intro > 0 and len(frames) > 0:
        print(f"  Building ghost intro ({args.intro} fade frames)...")
        source_resized = images[0].resize((img_w, img_h), Image.LANCZOS)
        source_arr = np.array(source_resized)
        gray = np.dot(source_arr[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        ghost = (gray * 0.3).round().astype(np.uint8)
        ghost_rgb = np.stack([ghost, ghost, ghost], axis=-1)
        first_frame = frames[0].astype(np.float32)
        ghost_arr = ghost_rgb.astype(np.float32)
        intro_frames = []
        for t in range(args.intro):
            alpha = t / args.intro
            blended = ((1 - alpha) * ghost_arr + alpha * first_frame).round().astype(np.uint8)
            intro_frames.append(blended)
        frames = intro_frames + frames

    # Smooth interpolation
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
        print(f"  Total frames: {len(frames)}")

    # Write MP4
    import imageio

    fps = args.fps
    if args.smooth > 0:
        video_fps = fps * 10
        imageio.mimwrite(
            args.animation, frames, fps=video_fps,
            codec="libx264", output_params=["-pix_fmt", "yuv420p"],
        )
        total_seconds = len(frames) / video_fps
    else:
        repeated_frames = []
        for frame in frames:
            repeated_frames.extend([frame] * fps)
        imageio.mimwrite(
            args.animation, repeated_frames, fps=fps,
            codec="libx264", output_params=["-pix_fmt", "yuv420p"],
        )
        total_seconds = len(repeated_frames) / fps

    print(f"✅ Saved animation to {args.animation} ({len(frames)} frames, {total_seconds:.1f}s)")


if __name__ == "__main__":
    main()
