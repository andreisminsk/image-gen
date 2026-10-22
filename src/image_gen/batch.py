"""Batch processing: many image generation jobs, one pipeline load.

Input is a zip archive (or a directory) containing ``job.json`` — a versioned
list of generation parameter sets — plus any referenced input images. The
diffusion pipeline is loaded once and reused across all jobs.

Design mirrors music-gen batch (BATCH-ARCH.md):
- validate everything before model load (exit 2)
- per-job error isolation, batch continues (exit 1 if any failed)
- batch_report.json written even on Ctrl+C
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BATCH_VERSION = 1
JOB_JSON = "job.json"
MAX_UNCOMPRESSED = 2 * 1024 * 1024 * 1024  # 2 GB (input photos can be large)
MAX_ENTRIES = 2000
MAX_CONSECUTIVE_OOM = 3

OUTPUT_FILE_RE = re.compile(r"^[A-Za-z0-9_-]+\.png$")
COMMANDS = ("image-gen", "i2i-gen")
QUANTS = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0")
DEVICES = ("cuda", "mps", "cpu")

DEFAULT_FIELDS = {
    "command", "output_dir", "device", "full_model", "quant",
    "steps", "cfg", "guidance_scale", "negative_prompt",
    "width", "height", "num_images",
}
DEFAULTS_ONLY_FIELDS = {"command", "output_dir", "device", "full_model", "quant"}
JOB_FIELDS = {
    "prompt", "negative_prompt", "seed", "width", "height", "steps",
    "output_file", "image", "cfg", "guidance_scale", "num_images",
}

_BUILTIN_DEFAULTS: dict[str, Any] = {
    "command": "image-gen",
    "output_dir": "output",
    "device": None,
    "full_model": False,
    "quant": "Q8_0",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "num_images": 1,
}
# Per-command fallbacks for fields the two pipelines disagree on.
_COMMAND_DEFAULTS: dict[str, dict[str, Any]] = {
    "image-gen": {"steps": 9, "cfg": None, "guidance_scale": 0.0, "negative_prompt": ""},
    "i2i-gen": {"steps": 20, "cfg": 4.0, "guidance_scale": 1.0, "negative_prompt": " "},
}


class BatchValidationError(Exception):
    """Batch source is invalid; carries ALL collected errors, not just the first."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} validation error(s)")


@dataclass
class BatchSpec:
    """A validated batch: source dir (job.json + input files) and job list."""

    root: Path
    defaults: dict[str, Any]
    jobs: list[dict[str, Any]]
    source: str
    _temp_dir: Optional[tempfile.TemporaryDirectory] = None

    def cleanup(self) -> None:
        """Remove the extraction dir if the source was a zip."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


def load_batch(source: str) -> BatchSpec:
    """Load and validate a batch from a zip archive or a directory.

    Raises BatchValidationError (with all errors) on any problem.
    """
    path = Path(source)
    if not path.exists():
        raise BatchValidationError([f"Batch source not found: {source}"])

    if path.is_dir():
        return _load_from_dir(path, source=str(path))

    temp = tempfile.TemporaryDirectory(prefix="image-gen-batch-")
    try:
        _extract_zip(path, Path(temp.name))
        spec = _load_from_dir(Path(temp.name), source=str(path))
    except Exception:
        temp.cleanup()
        raise
    spec._temp_dir = temp
    return spec


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip safely: size/entry caps, no path traversal, no absolute paths."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ENTRIES:
                raise BatchValidationError(
                    [f"Zip has {len(infos)} entries (max {MAX_ENTRIES})"])
            total = sum(i.file_size for i in infos)
            if total > MAX_UNCOMPRESSED:
                raise BatchValidationError(
                    [f"Zip expands to {total / (1 << 30):.1f} GB "
                     f"(max {MAX_UNCOMPRESSED // (1 << 30)} GB)"])
            dest_root = dest.resolve()
            for info in infos:
                parts = [p for p in info.filename.replace("\\", "/").split("/")
                         if p not in ("", ".")]
                if not parts:
                    continue
                if info.filename.startswith(("/", "\\")) or parts[0].endswith(":"):
                    raise BatchValidationError(
                        [f"Absolute path in zip entry: {info.filename!r}"])
                if ".." in parts:
                    raise BatchValidationError(
                        [f"Path traversal in zip entry: {info.filename!r}"])
                target = dest_root.joinpath(*parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    except zipfile.BadZipFile as e:
        raise BatchValidationError([f"Invalid zip archive {zip_path}: {e}"]) from e


def _find_job_json(base: Path) -> Path:
    direct = base / JOB_JSON
    if direct.is_file():
        return direct
    # Convenience: allow a zip that wraps everything in a single top-level folder
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and (subdirs[0] / JOB_JSON).is_file():
        return subdirs[0] / JOB_JSON
    raise BatchValidationError([f"{JOB_JSON} not found at the root of the batch source"])


def _load_from_dir(base: Path, source: str) -> BatchSpec:
    job_json_path = _find_job_json(base)
    root = job_json_path.parent
    errors: list[str] = []

    try:
        data = json.loads(job_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise BatchValidationError([f"Cannot read {JOB_JSON}: {e}"]) from e

    if not isinstance(data, dict):
        raise BatchValidationError([f"{JOB_JSON} must contain a JSON object"])

    version = data.get("version", BATCH_VERSION)
    if version != BATCH_VERSION:
        raise BatchValidationError(
            [f"Unsupported {JOB_JSON} version: {version!r} (expected {BATCH_VERSION})"])

    defaults_raw = data.get("defaults", {})
    if not isinstance(defaults_raw, dict):
        errors.append("'defaults' must be a JSON object")
        defaults_raw = {}
    unknown = set(defaults_raw) - DEFAULT_FIELDS
    if unknown:
        errors.append(
            f"defaults: unknown fields {sorted(unknown)} (allowed: {sorted(DEFAULT_FIELDS)})")

    command = defaults_raw.get("command", _BUILTIN_DEFAULTS["command"])
    if command not in COMMANDS:
        errors.append(
            f"defaults: 'command' must be one of " + "|".join(COMMANDS) + f" (got {command!r})")
        command = "image-gen"  # keep validating the rest

    if "quant" in defaults_raw and defaults_raw["quant"] not in QUANTS:
        errors.append("defaults: 'quant' must be one of " + "|".join(QUANTS))
    if "full_model" in defaults_raw and not isinstance(defaults_raw["full_model"], bool):
        errors.append("defaults: 'full_model' must be a boolean")
    if "device" in defaults_raw and defaults_raw["device"] not in (None,) + DEVICES:
        errors.append("defaults: 'device' must be one of " + "|".join(DEVICES))
    for sf in ("output_dir", "negative_prompt"):
        if sf in defaults_raw and not isinstance(defaults_raw[sf], str):
            errors.append(f"defaults: {sf!r} must be a string")
    for f in ("steps", "width", "height", "num_images"):
        if f in defaults_raw:
            v = defaults_raw[f]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                errors.append(f"defaults: {f!r} must be a positive integer")
    for f in ("cfg", "guidance_scale"):
        if f in defaults_raw:
            v = defaults_raw[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                errors.append(f"defaults: {f!r} must be a number")

    # Command-specific defaults: fail fast on fields the pipeline ignores
    if command == "image-gen":
        for f in ("cfg", "guidance_scale", "num_images"):
            if f in defaults_raw:
                errors.append(f"defaults: {f!r} only applies to i2i-gen batches")
    else:
        for f in ("width", "height"):
            if f in defaults_raw:
                errors.append(f"defaults: {f!r} only applies to image-gen batches")

    defaults = {**_BUILTIN_DEFAULTS, **_COMMAND_DEFAULTS[command], **defaults_raw}

    jobs_raw = data.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        errors.append("'jobs' must be a non-empty array")
        jobs_raw = []

    seen: dict[str, int] = {}
    valid_jobs: list[dict[str, Any]] = []
    for i, job in enumerate(jobs_raw):
        where = f"jobs[{i}]"
        if not isinstance(job, dict):
            errors.append(f"{where} must be a JSON object")
            continue

        unknown = (set(job) - JOB_FIELDS) - DEFAULTS_ONLY_FIELDS
        if unknown:
            errors.append(
                f"{where}: unknown fields {sorted(unknown)} (allowed: {sorted(JOB_FIELDS)})")
        forbidden = set(job) & DEFAULTS_ONLY_FIELDS
        if forbidden:
            errors.append(
                f"{where}: {sorted(forbidden)} can only be set in 'defaults', not per job")

        of = job.get("output_file")
        if not of or not isinstance(of, str):
            errors.append(f"{where}: 'output_file' is required, e.g. \"my-image.png\"")
        elif not OUTPUT_FILE_RE.match(of):
            errors.append(
                f"{where}: 'output_file' must be [A-Za-z0-9_-]+.png, got {of!r}")
        else:
            key = of.lower()
            if key in seen:
                errors.append(
                    f"{where}: duplicate output_file {of!r} (already used by jobs[{seen[key]}])")
            else:
                seen[key] = i

        p = job.get("prompt")
        if not isinstance(p, str) or not p.strip():
            errors.append(f"{where}: 'prompt' is required")

        if "seed" in job and (isinstance(job["seed"], bool) or not isinstance(job["seed"], int)):
            errors.append(f"{where}: 'seed' must be an integer")
        for f in ("steps", "width", "height", "num_images"):
            if f in job:
                v = job[f]
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    errors.append(f"{where}: {f!r} must be a positive integer")
        for f in ("cfg", "guidance_scale"):
            if f in job:
                v = job[f]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    errors.append(f"{where}: {f!r} must be a number")
        if "negative_prompt" in job and not isinstance(job["negative_prompt"], str):
            errors.append(f"{where}: 'negative_prompt' must be a string")

        # Command-specific job fields
        if command == "image-gen":
            for f in ("image", "cfg", "guidance_scale", "num_images"):
                if f in job:
                    errors.append(f"{where}: {f!r} only applies to i2i-gen batches")
        else:
            for f in ("width", "height"):
                if f in job:
                    errors.append(f"{where}: {f!r} only applies to image-gen batches")
            imgs = job.get("image")
            if imgs is None:
                errors.append(f"{where}: 'image' is required for i2i-gen jobs")
            else:
                if isinstance(imgs, str):
                    imgs = [imgs]
                if not isinstance(imgs, list) or not imgs \
                        or not all(isinstance(x, str) and x for x in imgs):
                    errors.append(f"{where}: 'image' must be a path or array of paths")
                else:
                    for x in imgs:
                        pp = Path(x)
                        if pp.is_absolute() or ".." in pp.parts:
                            errors.append(
                                f"{where}: image {x!r} must be relative to the batch root, no '..'")
                        elif not (root / pp).is_file():
                            errors.append(f"{where}: image file not found in batch: {x!r}")

        valid_jobs.append(job)

    if errors:
        raise BatchValidationError(errors)

    return BatchSpec(root=root, defaults=defaults, jobs=valid_jobs, source=source)


def _resolve_job(job: dict[str, Any], spec: BatchSpec) -> dict[str, Any]:
    """Merge defaults, resolve image paths, auto-assign a seed if missing."""
    command = spec.defaults["command"]
    r = {**spec.defaults}
    r.update({k: v for k, v in job.items() if v is not None})

    if command == "i2i-gen":
        imgs = r.get("image")
        if isinstance(imgs, str):
            imgs = [imgs]
        r["_image_paths"] = [str(spec.root / p) for p in (imgs or [])]

    if r.get("seed") is None:
        r["seed"] = random.randint(0, 2**31 - 1)

    return r


# ── Pipeline loading (delegates to gen.py / i2i.py loaders) ──

def _pick_device(requested: Optional[str]):
    import torch
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(device):
    import torch
    return torch.float32 if device.type in ("mps", "cpu") else torch.bfloat16


def load_pipeline(command: str, defaults: dict[str, Any], device_str: Optional[str] = None):
    """Load the pipeline once for the whole batch."""
    device = _pick_device(device_str or defaults.get("device"))
    dtype = _pick_dtype(device)
    logger.info("Device: %s (dtype=%s)", device, dtype)

    if command == "image-gen":
        from .gen import load_fp8_pipeline, load_full_pipeline
        if defaults.get("full_model"):
            pipe = load_full_pipeline(dtype, device)
        else:
            pipe = load_fp8_pipeline(dtype, device)
        pipe.to(device)
    else:
        from .i2i import load_gguf_pipeline, load_full_pipeline
        if defaults.get("full_model"):
            pipe = load_full_pipeline(dtype, device)
            pipe.to(device)
        else:
            pipe = load_gguf_pipeline(dtype, device, defaults.get("quant", "Q8_0"))
            logger.info("Enabling CPU offload (components swap in/out of VRAM)...")
            pipe.enable_model_cpu_offload()
    return pipe


# ── Execution ──

def run_batch(
    spec: BatchSpec,
    pipe,
    output_dir: Optional[str] = None,
    stop_on_error: bool = False,
    report_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run all jobs sequentially on one shared pipeline.

    Per-job failures are isolated and recorded; the batch continues unless
    stop_on_error is set or MAX_CONSECUTIVE_OOM CUDA OOMs occur in a row.
    The report is written even if the batch is interrupted (try/finally).
    """
    command = spec.defaults["command"]
    out_root = Path(output_dir) if output_dir else Path(spec.defaults["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        report_path = str(out_root / "batch_report.json")

    report: dict[str, Any] = {
        "batch": spec.source,
        "command": command,
        "output_dir": str(out_root),
        "started": _utcnow(),
        "finished": None,
        "duration_s": None,
        "summary": {"total": len(spec.jobs), "succeeded": 0, "failed": 0},
        "jobs": [],
    }

    total = len(spec.jobs)
    consecutive_oom = 0
    t_start = time.monotonic()

    try:
        for i, raw in enumerate(spec.jobs):
            job = _resolve_job(raw, spec)
            label = job["output_file"]
            prefix = f"[{i + 1}/{total}]"
            logger.info("%s %s — starting (seed=%s, steps=%s)",
                        prefix, label, job["seed"], job["steps"])
            entry: dict[str, Any] = {
                "index": i,
                "output_file": label,
                "prompt": job["prompt"],
                "seed": job["seed"],
                "status": None,
            }
            t0 = time.monotonic()
            err: Optional[Exception] = None
            outputs: list[str] = []
            try:
                outputs = _run_job(pipe, command, job, out_root)
            except Exception as e:  # noqa: BLE001 — every job failure is isolated
                err = e
            entry["duration_s"] = round(time.monotonic() - t0, 1)
            report["jobs"].append(entry)

            if err is None:
                entry["status"] = "ok"
                entry["output"] = outputs[0] if len(outputs) == 1 else outputs
                report["summary"]["succeeded"] += 1
                consecutive_oom = 0
                logger.info("%s %s — ok (%.1fs) → %s",
                            prefix, label, entry["duration_s"], ", ".join(outputs))
            else:
                entry["status"] = "error"
                entry["error"] = str(err)
                report["summary"]["failed"] += 1
                logger.error("%s %s — FAILED (%.1fs): %s",
                             prefix, label, entry["duration_s"], err)
                if _is_oom(err):
                    consecutive_oom += 1
                    if consecutive_oom >= MAX_CONSECUTIVE_OOM:
                        logger.error(
                            "Aborting batch: %d consecutive CUDA OOM failures — "
                            "GPU state is likely unrecoverable.", consecutive_oom)
                        break
                if stop_on_error:
                    logger.error("Stopping batch (--stop-on-error).")
                    break
            _empty_cache()
    finally:
        report["finished"] = _utcnow()
        report["duration_s"] = round(time.monotonic() - t_start, 1)
        _write_report(report, report_path)

    logger.info("Batch done: %d/%d succeeded.",
                report["summary"]["succeeded"], report["summary"]["total"])
    return report


def _run_job(pipe, command: str, job: dict[str, Any], out_root: Path) -> list[str]:
    """Execute one resolved job. Returns saved output paths."""
    import torch
    from PIL import Image

    gen = torch.manual_seed(job["seed"])
    out_path = out_root / job["output_file"]

    if command == "image-gen":
        with torch.inference_mode():
            image = pipe(
                prompt=job["prompt"],
                negative_prompt=job.get("negative_prompt") or None,
                height=job["height"],
                width=job["width"],
                num_inference_steps=job["steps"],
                guidance_scale=0.0,
                generator=gen,
            ).images[0]
        image.save(out_path)
        return [str(out_path)]

    # i2i-gen
    images = []
    for p in job["_image_paths"]:
        img = Image.open(p).convert("RGB")
        logger.info("  loaded %s (%dx%d)", p, img.size[0], img.size[1])
        images.append(img)

    with torch.inference_mode():
        out = pipe(
            image=images,
            prompt=job["prompt"],
            negative_prompt=job.get("negative_prompt", " "),
            num_inference_steps=job["steps"],
            true_cfg_scale=job["cfg"],
            guidance_scale=job["guidance_scale"],
            num_images_per_prompt=job["num_images"],
            generator=gen,
        )

    paths = []
    if len(out.images) == 1:
        out.images[0].save(out_path)
        paths.append(str(out_path))
    else:
        stem, ext = out_path.stem, out_path.suffix
        for idx, img in enumerate(out.images):
            p = out_path.with_name(f"{stem}_{idx}{ext}")
            img.save(p)
            paths.append(str(p))
    return paths


def _is_oom(err: BaseException) -> bool:
    return "out of memory" in str(err).lower()


def _empty_cache() -> None:
    """Best-effort CUDA cache cleanup between jobs (fragmentation insurance)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:  # noqa: BLE001 — cache cleanup must never fail a batch
        pass


def _write_report(report: dict[str, Any], path: str) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                     encoding="utf-8")
        logger.info("Batch report: %s", p)
    except OSError as e:
        logger.error("Failed to write batch report to %s: %s", path, e)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── CLI ──

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="image-gen-batch",
        description="Run multiple image generation jobs with one pipeline load",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--zip", dest="zip_path", help="Path to a batch zip archive")
    src.add_argument("--dir", dest="dir_path", help="Path to a directory containing job.json")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Override the batch-level output directory")
    parser.add_argument("--device", default=None, help="Override device: cuda, mps, or cpu")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Abort the batch on the first job failure")
    parser.add_argument("--report", default=None,
                        help="Batch report JSON path (default: <output-dir>/batch_report.json)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        spec = load_batch(args.zip_path or args.dir_path)
    except BatchValidationError as e:
        print("❌ Batch validation failed:", file=sys.stderr)
        for err in e.errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        spec.defaults["output_dir"] = args.output_dir

    command = spec.defaults["command"]
    logger.info("Batch: %d job(s), command=%s, from %s → %s",
                len(spec.jobs), command, spec.source, spec.defaults["output_dir"])

    pipe = load_pipeline(command, spec.defaults, device_str=args.device)
    try:
        report = run_batch(
            spec,
            pipe,
            output_dir=args.output_dir,
            stop_on_error=args.stop_on_error,
            report_path=args.report,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Batch interrupted — partial report saved.", file=sys.stderr)
        return 130
    finally:
        spec.cleanup()
        del pipe
        _empty_cache()

    summary = report["summary"]
    print(f"\n✅ Batch finished: {summary['succeeded']}/{summary['total']} succeeded, "
          f"{summary['failed']} failed")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
