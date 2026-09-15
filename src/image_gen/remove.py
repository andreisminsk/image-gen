"""Prompt-driven object removal: segment → select → inpaint."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


# ── Utilities ──

def downscale(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resize image so its longest side is at most max_dim."""
    w, h = image.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask via border flood-fill (pure numpy)."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not m[y, x]:
                visited[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not m[y, x]:
                visited[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and not m[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    holes = (~visited) & (m == 0)
    m[holes] = 1
    return m.astype(bool)


def dedup_masks(masks: list, ious: list, threshold: float = 0.8):
    """Drop masks that overlap heavily (Jaccard > threshold) with a kept mask."""
    ious = list(ious)
    kept_masks, kept_ious = [], []
    for m, io in zip(masks, ious):
        if int(m.sum()) == 0:
            continue
        dup = False
        for km in kept_masks:
            inter = int((m & km).sum())
            union = int((m | km).sum())
            if union > 0 and inter / union > threshold:
                dup = True
                break
        if not dup:
            kept_masks.append(m)
            kept_ious.append(io)
    return kept_masks, kept_ious


def select_mask(masks, ious, image, prompt, device,
                use_clip: bool = True, mask_index=None, prefer: str = "area"):
    """Pick the best mask via manual override, CLIP semantic match, or area/IoU fallback."""
    ious = np.asarray(ious).ravel()
    n = len(masks)
    if n == 0:
        return None

    if mask_index is not None:
        idx = mask_index % n
        m = masks[idx]
        print(f"  Manual mask {idx} (iou={ious[idx]:.3f}, {int(m.sum())} px)")
        return m

    if use_clip:
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_id = "openai/clip-vit-base-patch32"
            cproc = CLIPProcessor.from_pretrained(clip_id, local_files_only=True)
            clip = CLIPModel.from_pretrained(clip_id, local_files_only=True).to(device)
            text = cproc(text=[prompt], return_tensors="pt", padding=True).to(device)
            tfeat = clip.get_text_features(**text)
            tfeat = tfeat.pooler_output if hasattr(tfeat, "pooler_output") else tfeat
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            best_score, best_mask = -1.0, None
            for i, m in enumerate(masks):
                if int(m.sum()) == 0:
                    continue
                arr = np.asarray(image).copy()
                arr[~m] = 0
                masked = Image.fromarray(arr)
                iin = cproc(images=masked, return_tensors="pt").to(device)
                ifeat = clip.get_image_features(**iin)
                ifeat = ifeat.pooler_output if hasattr(ifeat, "pooler_output") else ifeat
                ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
                score = (tfeat @ ifeat.T).item()
                print(f"    mask {i}: {int(m.sum())} px, iou={ious[i]:.3f}, clip={score:.3f}")
                if score > best_score:
                    best_score, best_mask = score, m
            if best_mask is not None:
                print(f"  CLIP best mask: {int(best_mask.sum())} px, score={best_score:.3f}")
                return best_mask
            print("  CLIP found no match; falling back to area")
        except Exception as e:
            print(f"  CLIP unavailable ({e}); falling back to {prefer}")

    if prefer == "area":
        best_idx = int(np.argmax([int(m.sum()) for m in masks]))
    else:
        best_idx = int(np.argmax(ious))
    best_mask = masks[best_idx]
    print(f"  Selected mask {best_idx} by {prefer} (iou={ious[best_idx]:.3f}, {int(best_mask.sum())} px)")
    return best_mask


def save_mask_and_preview(mask: np.ndarray, image: Image.Image, out_dir: Path,
                          name: str = "mask") -> np.ndarray:
    """Fill holes, save binary mask + red overlay preview. Returns filled mask."""
    mask = fill_holes(mask)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_path = out_dir / f"{name}.mask.png"
    Image.fromarray(mask_u8).save(mask_path)
    print(f"  Mask saved -> {mask_path}")

    overlay = image.copy().convert("RGBA")
    mask_layer = Image.new("RGBA", image.size, (255, 0, 0, 0))
    mask_layer.putalpha(Image.fromarray(mask_u8, mode="L"))
    overlay = Image.alpha_composite(overlay, mask_layer).convert("RGB")
    preview_path = out_dir / f"{name}.preview.png"
    overlay.save(preview_path)
    print(f"  Preview saved -> {preview_path}")
    return mask


# ── Segmentation ──

def segment_maskrcnn(image: Image.Image, device: str = "cpu"):
    """Mask R-CNN instance segmentation (detects COCO person class)."""
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision import transforms as T
    import os

    print(f"  Mask R-CNN segmenting on {device} ...")
    cache = os.path.expanduser("~/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
    if os.path.exists(cache):
        model = maskrcnn_resnet50_fpn(weights=None)
        model.load_state_dict(torch.load(cache, map_location=device))
    else:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    model = model.to(device).eval()
    img_t = T.Compose([T.ToTensor()])(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)[0]
    h, w = image.size[1], image.size[0]
    results = []
    for i in range(len(out["masks"])):
        score = float(out["scores"][i])
        cls = int(out["labels"][i])
        if cls != 1:
            continue
        m = out["masks"][i, 0].cpu().numpy() > 0.5
        bbox = out["boxes"][i].cpu().tolist()
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bbox_area = (x2 - x1) * (y2 - y1)
        coverage = int(m.sum()) / max(bbox_area, 1)
        if coverage < 0.6 and bbox_area < 50000:
            m_bbox = np.zeros((h, w), dtype=bool)
            m_bbox[y1:y2, x1:x2] = True
            print(f"    person {i}: {int(m.sum())} px, score={score:.3f} -> FILLED bbox ({coverage:.0%})")
            m = m_bbox
        else:
            print(f"    person {i}: {int(m.sum())} px, score={score:.3f}")
        results.append((m, score, bbox))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def segment_sam(image: Image.Image, device: str = "cpu", point=None):
    """SAM segmentation — point-prompted or grid auto-segment."""
    import torch
    from transformers import SamProcessor, SamModel
    import torch.nn.functional as F

    model_id = "facebook/sam-vit-huge"
    print(f"  SAM segmenting on {device} ...")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id).to(device)
    inputs = processor(image, return_tensors="pt").to(device)
    h, w = image.size[1], image.size[0]

    def segment_at(x, y):
        pts = torch.tensor([[[[x, y]]]], device=device)
        labels = torch.tensor([[[1]]], device=device)
        with torch.no_grad():
            o = model(**inputs, input_points=pts, input_labels=labels,
                      multimask_output=True)
        ious = o.iou_scores.cpu().numpy().reshape(-1)
        # Use processor's post_process_masks to correctly handle padding
        # and resize masks back to the original image resolution.
        masks = processor.image_processor.post_process_masks(
            o.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        out = []
        for m in masks[0]:
            arr = m.numpy()
            # post_process_masks may return [1, H, W] or [H, W]
            while arr.ndim > 2:
                arr = arr[0]
            out.append(arr > 0.5)
        return out, ious

    if point is not None:
        x, y = point
        masks, ious = segment_at(x, y)
        print(f"    point ({x},{y}) -> {len(masks)} masks, iou={ious}")
    else:
        grid = [(int(w * fx), int(h * fy))
                for fx in (0.25, 0.5, 0.75) for fy in (0.3, 0.5, 0.7)]
        masks, ious = [], []
        for gx, gy in grid:
            ms, io = segment_at(gx, gy)
            masks.extend(ms)
            ious.extend(io)
        print(f"    grid -> {len(masks)} candidate masks")

    masks, ious = dedup_masks(masks, ious, threshold=0.8)
    print(f"    {len(masks)} distinct masks after dedup")
    return masks, ious


# ── Inpainting ──

def inpaint_sdxl(image: Image.Image, mask: np.ndarray, out_dir: Path,
                 device: str = "mps",
                 model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                 inpaint_prompt: str = "", steps: int = 30,
                 guidance_scale: float = 3.0, feather: int = 8,
                 seed: int = 42) -> None:
    """SDXL inpainting — context-aware but slower."""
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline

    print(f"  SDXL inpainting on {device} ({steps} steps) ...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)

    orig_w, orig_h = image.size
    new_w = (orig_w // 64) * 64
    new_h = (orig_h // 64) * 64
    image = image.resize((new_w, new_h), Image.LANCZOS)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_u8, mode="L").resize((new_w, new_h), Image.NEAREST)
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    out_img = pipe(
        prompt=inpaint_prompt,
        image=image,
        mask_image=mask_img,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    if out_img.size != (orig_w, orig_h):
        out_img = out_img.resize((orig_w, orig_h), Image.LANCZOS)

    out_path = out_dir / "inpainted.png"
    out_img.save(out_path)
    print(f"  Inpainted -> {out_path} ({out_img.size[0]}x{out_img.size[1]})")


def inpaint_lama(image: Image.Image, mask: np.ndarray, out_dir: Path,
                 feather: int = 16, dilate: int = 10) -> None:
    """LaMa inpainting — fast, no hallucination."""
    import torch
    from simple_lama_inpainting import SimpleLama
    from scipy import ndimage

    print(f"  LaMa inpainting (dilate={dilate}px, feather={feather}px) ...")
    # The LaMa TorchScript model ships with CUDA tensors embedded.
    # Force CPU mapping at load time so it works on MPS-only machines.
    _orig_jit_load = torch.jit.load
    torch.jit.load = lambda *a, **kw: _orig_jit_load(*a, **{**kw, "map_location": "cpu"})
    try:
        lama = SimpleLama(device=torch.device("cpu"))
    finally:
        torch.jit.load = _orig_jit_load

    if dilate > 0:
        mask_dilated = ndimage.binary_dilation(mask, iterations=dilate)
    else:
        mask_dilated = mask.copy()

    mask_pil = Image.fromarray((mask_dilated.astype(np.uint8) * 255), mode="L")
    result = lama(image, mask_pil)

    out_path = out_dir / "inpainted.png"
    result.save(out_path)
    print(f"  Inpainted -> {out_path}")


# ── Main ──

def pick_device(requested=None):
    if requested:
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Prompt-driven object removal: segment → select → inpaint")
    parser.add_argument("-i", "--image", required=True, help="Input image path")
    parser.add_argument("-p", "--prompt", required=True, help="Text description of object to remove")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: same as input)")
    parser.add_argument("--point", default=None, help="x,y point prompt for SAM (e.g. 700,512)")
    parser.add_argument("--mask-index", type=int, default=None, help="Force a specific mask index (0-based)")
    parser.add_argument("--exclude-mask", type=int, default=None, help="Keep this mask index, combine all others for removal")
    parser.add_argument("--max-dim", type=int, default=1024, help="Downscale longest side to this (default: 1024)")
    parser.add_argument("--no-clip", action="store_true", help="Skip CLIP semantic selection, use area fallback")
    parser.add_argument("--prefer", choices=["area", "iou"], default="area", help="Fallback selection mode")

    seg = parser.add_argument_group("Segmentation")
    seg.add_argument("--sam", action="store_true", help="Use SAM instead of Mask R-CNN")
    seg.add_argument("--all-classes", action="store_true", help="Mask R-CNN: detect all COCO classes, not just person")

    inp = parser.add_argument_group("Inpainting")
    inp.add_argument("--inpaint", action="store_true", help="Run inpainting after segmentation")
    inp.add_argument("--inpainter", choices=["sdxl", "lama"], default="sdxl", help="Inpainting model (default: sdxl)")
    inp.add_argument("--inpaint-prompt", default="", help="What to fill the hole with (empty = context, NOT the removal prompt)")
    inp.add_argument("--inpaint-model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", help="SDXL model ID")
    inp.add_argument("--steps", type=int, default=30, help="SDXL inference steps (default: 30)")
    inp.add_argument("--guidance-scale", type=float, default=3.0, help="SDXL guidance scale (default: 3.0)")
    inp.add_argument("--feather", type=int, default=8, help="Mask feather radius in px (default: 8)")
    inp.add_argument("--dilate", type=int, default=10, help="LaMa mask dilation in px (default: 10)")
    inp.add_argument("--seed", type=int, default=42, help="Random seed for SDXL (default: 42)")

    parser.add_argument("--device", default=None, help="Force device: cuda, mps, or cpu")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")

    image = Image.open(args.image).convert("RGB")
    print(f"Loaded {args.image} ({image.size[0]}x{image.size[1]})")
    if max(image.size) > args.max_dim:
        image = downscale(image, args.max_dim)
        print(f"  Downscaled to {image.size[0]}x{image.size[1]}")

    out_dir = Path(args.output) if args.output else Path(args.image).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # Segmentation
    print("\n[1/3] Segmenting ...")
    if args.sam:
        point = tuple(map(int, args.point.split(","))) if args.point else None
        masks, ious = segment_sam(image, device=device, point=point)
    else:
        results = segment_maskrcnn(image, device=device)
        if not args.all_classes:
            results = [(m, s, b) for m, s, b in results]
        masks = [r[0] for r in results]
        ious = [r[1] for r in results]

    if not masks:
        print("No masks found. Aborting.")
        sys.exit(1)

    # Mask selection
    print("\n[2/3] Selecting mask ...")
    if args.exclude_mask is not None:
        keep_idx = args.exclude_mask % len(masks)
        print(f"  Keeping mask {keep_idx}, combining {len(masks) - 1} others for removal")
        combined = np.zeros_like(masks[0], dtype=bool)
        for i, m in enumerate(masks):
            if i != keep_idx:
                combined |= m
                print(f"    adding mask {i} ({int(m.sum())} px)")
        if combined.sum() == 0:
            print("  No masks to remove. Aborting.")
            sys.exit(1)
        print(f"  Combined removal mask: {int(combined.sum())} px")
        best_mask = combined
    else:
        best_mask = select_mask(masks, ious, image, args.prompt, device,
                                use_clip=not args.no_clip,
                                mask_index=args.mask_index,
                                prefer=args.prefer)
    if best_mask is None:
        print("No mask selected. Aborting.")
        sys.exit(1)

    best_mask = save_mask_and_preview(best_mask, image, out_dir, name=stem)

    # Inpainting
    if args.inpaint:
        print("\n[3/3] Inpainting ...")
        if args.inpainter == "lama":
            inpaint_lama(image, best_mask, out_dir, feather=args.feather, dilate=args.dilate)
        else:
            inpaint_sdxl(image, best_mask, out_dir, device=device,
                         model_id=args.inpaint_model, inpaint_prompt=args.inpaint_prompt,
                         steps=args.steps, guidance_scale=args.guidance_scale,
                         feather=args.feather, seed=args.seed)
    else:
        print("\n[3/3] Skipping inpainting (--inpaint not set). Mask saved.")


if __name__ == "__main__":
    main()
