"""Core cross-platform utilities."""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

# ---------- HEIC/HEIF support (registers Pillow opener at import) ----------
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False
    logger.debug("pillow-heif not installed; .heic/.heif files will be skipped")

# ---------- Platform flags ----------
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Avoid PIL throwing on huge images (e.g., scans). Tune to taste.
Image.MAX_IMAGE_PIXELS = 933_120_000  # ~933 MP

IMG_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif",
    ".heic", ".heif",  # only loadable if pillow-heif installed
})


# ---------- Path utilities ----------

def normalize_path(p: Path | str) -> Path:
    """Resolve to absolute path. On Windows, apply long-path prefix if needed."""
    path = Path(p).expanduser().resolve()
    # Windows MAX_PATH is 260; the \\?\ prefix bypasses it for most APIs.
    if IS_WINDOWS:
        s = str(path)
        if len(s) > 240 and not s.startswith("\\\\?\\"):
            # UNC paths take \\?\UNC\server\share, local paths take \\?\C:\...
            if s.startswith("\\\\"):
                path = Path("\\\\?\\UNC\\" + s.lstrip("\\"))
            else:
                path = Path("\\\\?\\" + s)
    return path


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def cache_key(path: Path) -> str:
    """Canonical string key for cache lookups.

    Windows filesystem is case-insensitive; lowercase to avoid duplicate entries
    when the same file is referenced with different casing.
    """
    s = str(path.resolve())
    return s.lower() if IS_WINDOWS else s


def next_nonconflicting_path(dst: Path) -> Path:
    """Find a non-existing variant of dst by appending ' (1)', ' (2)', ..."""
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    n = 1
    while True:
        cand = dst.with_name(f"{stem} ({n}){ext}")
        if not cand.exists():
            return cand
        n += 1


def scan_images(roots: list[Path]) -> Iterator[tuple[Path, Path]]:
    """Yield (image_path, source_root) for each image under each root."""
    for root in roots:
        if not root.exists():
            logger.warning("Input root does not exist: %s", root)
            continue
        for p in root.rglob("*"):
            try:
                if p.is_file() and is_image_file(p):
                    yield (p, root)
            except OSError as e:
                # Permission errors, broken symlinks, etc. — log and skip.
                logger.debug("Skipping %s: %s", p, e)


def count_image_files(root: Path) -> int:
    """Count images under a root for progress bars."""
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and is_image_file(p):
                n += 1
        except OSError:
            pass
    return n


# ---------- Image I/O ----------

def safe_open_pil(path: Path, apply_exif: bool = True) -> Optional[Image.Image]:
    """Open an image with EXIF orientation correction. Returns None on failure."""
    try:
        with Image.open(path) as im:
            im.load()
            if apply_exif:
                im = ImageOps.exif_transpose(im)
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        return None


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> tuple[int, float]:
    """(size, mtime) — cheap cache-invalidation key."""
    st = path.stat()
    return (st.st_size, st.st_mtime)


# ---------- Atomic file ops ----------

def atomic_save_image(im: Image.Image, dst: Path, **save_kwargs) -> None:
    """Save image atomically: write to .tmp, then os.replace to final name."""
    ensure_dir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        im.save(tmp, **save_kwargs)
        # os.replace is atomic on POSIX and Windows (overwrites if dst exists).
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def mirror_to_unkept(
    src: Path,
    src_root: Path,
    unkept_root: Path,
    copy_instead: bool,
    overwrite: bool,
) -> Optional[Path]:
    """Mirror src into unkept_root, preserving relative tree.

    Incremental-safe:
      - If destination already exists with identical SHA-256, skip.
      - Otherwise, overwrite or generate non-conflicting name.

    Returns the destination Path, or None if the source no longer exists.
    """
    if not src.exists():
        return None
    rel = src.relative_to(src_root)
    dst = unkept_root / rel
    ensure_dir(dst.parent)

    if dst.exists():
        try:
            if file_sha256(src) == file_sha256(dst):
                # Already mirrored. If moving, remove the source.
                if not copy_instead:
                    try:
                        src.unlink()
                    except OSError as e:
                        logger.warning("Couldn't remove already-mirrored %s: %s", src, e)
                return dst
        except OSError:
            pass

        if overwrite:
            try:
                dst.unlink()
            except OSError:
                pass
        else:
            dst = next_nonconflicting_path(dst)

    try:
        if copy_instead:
            shutil.copy2(str(src), str(dst))
        else:
            shutil.move(str(src), str(dst))
        return dst
    except OSError as e:
        logger.error("Failed to mirror %s -> %s: %s", src, dst, e)
        return None


# ---------- Image processing ----------

def center_crop_square_resize(im: Image.Image, side: int = 300) -> Image.Image:
    """Center-crop to square then resize to (side, side)."""
    w, h = im.size
    # Upscale first if too small (so we never crop below `side`).
    scale = max(side / w, side / h, 1.0)
    if scale != 1.0:
        im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)
        w, h = im.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    im = im.crop((left, top, left + s, top + s))
    if im.size != (side, side):
        im = im.resize((side, side), Image.LANCZOS)
    return im


def image_sharpness(im: Image.Image) -> float:
    """Variance of Laplacian — higher means sharper. Used as tie-breaker for
    representative selection."""
    import cv2
    arr = np.asarray(im.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


# ---------- Device picking ----------

def pick_device(prefer: str = "auto") -> str:
    """Choose torch device string: 'cuda', 'mps', or 'cpu'.

    prefer: 'auto' (default) chooses best available; otherwise tries the named
    device and falls back to cpu if unavailable.
    """
    import torch

    if prefer == "cpu":
        return "cpu"

    has_cuda = torch.cuda.is_available()
    has_mps = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )

    if prefer == "cuda":
        return "cuda" if has_cuda else "cpu"
    if prefer == "mps":
        return "mps" if has_mps else "cpu"

    # auto
    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def yolo_device_arg(device_str: str) -> str | int:
    """ultralytics YOLO accepts 'cpu', 'mps', or int GPU index."""
    if device_str == "cuda":
        return 0
    return device_str  # 'cpu' or 'mps'
