"""Export survivors as center-cropped, numbered JPEGs.

Atomic writes (temp + rename) so a Ctrl+C never leaves a half-written file.
A manifest CSV next to the output records the mapping from original path to
output filename.
"""
from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from lsict.core import (
    atomic_save_image,
    center_crop_square_resize,
    ensure_dir,
    safe_open_pil,
)

logger = logging.getLogger(__name__)


def clear_dir_files(d: Path, keep_extensions: Optional[set[str]] = None) -> int:
    """Remove files in d, optionally only those with matching extensions.
    Subdirectories are left alone. Returns count removed.
    """
    if not d.exists():
        return 0
    removed = 0
    for child in d.iterdir():
        if not child.is_file():
            continue
        if keep_extensions is None or child.suffix.lower() in keep_extensions:
            try:
                child.unlink()
                removed += 1
            except OSError as e:
                logger.warning("Couldn't remove %s: %s", child, e)
    return removed


def export_numbered(
    survivors: list[Path],
    kept_dir: Path,
    side: int = 300,
    jpeg_quality: int = 92,
    clean_first: bool = True,
    sort_key=None,
) -> tuple[int, Path]:
    """Center-crop survivors to (side, side) JPEGs named 1.jpg, 2.jpg, ...

    Returns (count_exported, manifest_path).
    """
    ensure_dir(kept_dir)

    if clean_first:
        removed = clear_dir_files(kept_dir, keep_extensions={".jpg", ".jpeg"})
        if removed:
            logger.info("Cleaned %d existing JPEGs from %s", removed, kept_dir)

    if sort_key is None:
        sort_key = lambda p: str(p)
    survivors = sorted(survivors, key=sort_key)

    manifest_path = kept_dir / "manifest.csv"
    count = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "filename", "original_path"])

        for i, src in enumerate(tqdm(survivors, desc="Export 300x300", unit="img"),
                                start=1):
            im = safe_open_pil(src)
            if im is None:
                logger.warning("Skipping unreadable: %s", src)
                continue
            try:
                processed = center_crop_square_resize(im, side=side)
            except Exception as e:
                logger.warning("Crop/resize failed for %s: %s", src, e)
                continue
            out_name = f"{i}.jpg"
            out_path = kept_dir / out_name
            try:
                atomic_save_image(
                    processed, out_path,
                    format="JPEG", quality=jpeg_quality, optimize=True,
                )
                w.writerow([i, out_name, str(src)])
                count += 1
            except OSError as e:
                logger.warning("Couldn't write %s: %s", out_path, e)

    logger.info("Exported %d images to %s (manifest: %s)", count, kept_dir, manifest_path)
    return count, manifest_path
