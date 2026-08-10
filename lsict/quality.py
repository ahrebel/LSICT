"""Image quality filters: blur, grayscale, and uniform borders.

All three metrics come from ONE decode per image and are cached, so re-runs
— or re-runs with different thresholds — never re-read the files:

  sharpness   — variance of Laplacian, computed at a normalized resolution
                so the threshold means the same thing for a 12 MP photo and
                a 600 px thumbnail. Blurry images score low; ~100 is a
                reasonable cutoff for most photo sets.
  saturation  — mean HSV saturation (0-255). Grayscale/B&W images sit near
                zero even when saved as RGB files.
  border_frac — the thickest uniform-colored band along any edge, as a
                fraction of that side. Letterboxing and solid frames score
                high. Note: images with genuinely plain edges (product
                shots on white, big skies) can also trigger this — check
                the Review tab.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from lsict.cache import Cache
from lsict.core import safe_open_pil

logger = logging.getLogger(__name__)

SHARPNESS_SIDE = 800   # resolution the sharpness metric is normalized to
DETAIL_SIDE = 256      # saturation / border checks run on a small thumbnail
BORDER_TOL = 12        # max channel spread for a row/col to count as "flat"


def _flat_band(lines) -> int:
    """Count consecutive uniform rows (or columns) from one edge inward."""
    n = 0
    for line in lines:
        if int(line.max()) - int(line.min()) <= BORDER_TOL:
            n += 1
        else:
            break
    return n


def _max_border_fraction(arr: np.ndarray) -> float:
    h, w = arr.shape[:2]
    if not h or not w:
        return 0.0
    top = _flat_band(arr)
    bottom = _flat_band(arr[::-1])
    cols = arr.transpose(1, 0, 2)
    left = _flat_band(cols)
    right = _flat_band(cols[::-1])
    return max(top / h, bottom / h, left / w, right / w)


def compute_quality_metrics(path: Path) -> Optional[tuple[float, float, float]]:
    """(sharpness, saturation, border_frac) for one image; None if unreadable."""
    im = safe_open_pil(path)
    if im is None:
        return None

    im_s = im.copy()
    im_s.thumbnail((SHARPNESS_SIDE, SHARPNESS_SIDE), Image.BILINEAR)
    gray = np.asarray(im_s.convert("L"))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    im_t = im.copy()
    im_t.thumbnail((DETAIL_SIDE, DETAIL_SIDE), Image.BILINEAR)
    saturation = float(np.asarray(im_t.convert("HSV"))[:, :, 1].mean())
    border_frac = _max_border_fraction(np.asarray(im_t).astype(np.int16))

    return sharpness, saturation, border_frac


def run_quality_filter(
    files: list[Path],
    cache: Cache,
    min_sharpness: float = 0.0,
    drop_grayscale: bool = False,
    drop_bordered: bool = False,
    grayscale_max_saturation: float = 10.0,
    border_min_frac: float = 0.03,
) -> dict[Path, str]:
    """Return {path: reason} for files failing the enabled checks.

    Reasons: 'blurry', 'grayscale', 'bordered'. Metrics are cached in
    chunks, so an interrupted run resumes where it stopped.
    """
    metrics: dict[Path, tuple[float, float, float]] = {}
    need: list[Path] = []
    for p in files:
        m = cache.get_quality(p)
        if m is not None:
            metrics[p] = m
        else:
            need.append(p)

    if need:
        logger.info("Quality: %d cached, %d to compute",
                    len(files) - len(need), len(need))
        chunk_size = 256
        pbar = tqdm(total=len(need), desc="Quality check", unit="img")
        for i in range(0, len(need), chunk_size):
            with cache.transaction():
                for p in need[i:i + chunk_size]:
                    m = compute_quality_metrics(p)
                    if m is not None:
                        cache.set_quality(p, *m)
                        metrics[p] = m
            pbar.update(min(chunk_size, len(need) - i))
        pbar.close()

    rejected: dict[Path, str] = {}
    for p, (sharpness, saturation, border_frac) in metrics.items():
        if min_sharpness > 0 and sharpness < min_sharpness:
            rejected[p] = "blurry"
        elif drop_grayscale and saturation <= grayscale_max_saturation:
            rejected[p] = "grayscale"
        elif drop_bordered and border_frac >= border_min_frac:
            rejected[p] = "bordered"
    return rejected
