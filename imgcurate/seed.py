"""Seed a diverse image set from Open Images v7 via FiftyOne.

Cross-platform: uses Path everywhere, requires explicit args (no hardcoded
volumes), and the multiprocessing pool is created from within a function
called from `if __name__ == "__main__"` (via the CLI), which is the
contract required on Windows (spawn start method).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import random
from functools import partial
from pathlib import Path

logger = logging.getLogger(__name__)


CLASSES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"


def _process_one(args: tuple[int, str], out_dir: str, side: int) -> str | None:
    """Worker function for the multiprocessing pool. Must be top-level."""
    from PIL import Image
    import os

    index, filepath = args
    try:
        img = Image.open(filepath)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Center crop to (side, side) — upscale first if image is smaller
        w, h = img.size
        scale = max(side / w, side / h, 1.0)
        if scale != 1.0:
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
            w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        if img.size != (side, side):
            img = img.resize((side, side), Image.LANCZOS)

        out_path = os.path.join(out_dir, f"{index + 1}.jpg")
        img.save(out_path, "JPEG", quality=92, optimize=True)
        return None
    except Exception as e:
        return f"Could not process {filepath}: {e}"


def run_seed(
    output_dir: Path,
    num_categories: int = 1000,
    images_per_category: int = 2,
    image_side: int = 300,
    seed: int | None = None,
) -> dict:
    """Download diverse Open Images sampling and center-crop to image_side."""
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "Seeding requires fiftyone + pandas. Install: pip install imgcurate[seed]"
        ) from e

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)

    logger.info("Fetching Open Images class list...")
    df = pd.read_csv(CLASSES_URL)
    all_classes = df["DisplayName"].tolist()
    chosen = random.sample(all_classes, min(num_categories, len(all_classes)))
    logger.info("Sampling %d categories, up to %d images each",
                len(chosen), images_per_category)

    dataset_name = f"imgcurate_seed_{random.randint(1_000_000, 9_999_999)}"
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=chosen,
        max_samples_per_class=images_per_category,
        only_matching=True,
        dataset_name=dataset_name,
    )
    n_loaded = len(dataset)
    logger.info("Downloaded %d source images", n_loaded)

    filepaths = dataset.values("filepath")
    args_list = list(enumerate(filepaths))

    # mp.Pool on Windows uses spawn; the worker (_process_one) must be at
    # module top-level (it is) and we must be invoked from inside __main__
    # (the CLI handles that).
    num_workers = max(1, mp.cpu_count() - 1)
    logger.info("Processing with %d worker(s)...", num_workers)

    worker = partial(_process_one, out_dir=str(output_dir), side=image_side)
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker, args_list)

    errors = [r for r in results if r is not None]
    n_ok = n_loaded - len(errors)

    # Clean up FiftyOne dataset registration (samples already downloaded)
    try:
        fo.delete_dataset(dataset_name)
    except Exception:
        pass

    if errors:
        logger.warning("%d files failed:", len(errors))
        for err in errors[:10]:
            logger.warning("  %s", err)
        if len(errors) > 10:
            logger.warning("  ... and %d more", len(errors) - 10)

    return {
        "downloaded": n_loaded,
        "processed": n_ok,
        "failed": len(errors),
        "output_dir": str(output_dir),
    }
