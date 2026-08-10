"""Shared pipeline implementations used by both the CLI and the GUI.

The CLI subcommands are thin wrappers around these functions; the GUI calls
them directly. Errors are raised as ValueError/RuntimeError (not sys.exit)
so callers can present them however they like.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from lsict.cache import Cache, default_cache_path
from lsict.core import (
    HEIC_AVAILABLE,
    ensure_dir,
    mirror_to_unkept,
    normalize_path,
    pick_device,
    scan_images,
)

logger = logging.getLogger(__name__)

OUTCOMES_CSV = "outcomes.csv"


def collect_inputs(input_dirs: list) -> tuple[list[Path], dict[Path, Path]]:
    """Resolve input roots, scan all images, build path->root map.

    Raises ValueError if a root doesn't exist.
    """
    roots = [normalize_path(p) for p in input_dirs]
    for r in roots:
        if not r.exists():
            raise ValueError(f"Input root does not exist: {r}")

    all_imgs: list[Path] = []
    roots_map: dict[Path, Path] = {}
    for p, root in scan_images(roots):
        all_imgs.append(p)
        roots_map[p] = root
    logger.info("Scanned %d images across %d input root(s).", len(all_imgs), len(roots))
    if not HEIC_AVAILABLE:
        n_heic = sum(1 for p in all_imgs if p.suffix.lower() in {".heic", ".heif"})
        if n_heic:
            logger.warning(
                "%d HEIC/HEIF files found but pillow-heif not installed — they will be skipped.",
                n_heic,
            )
    return all_imgs, roots_map


def _open_cache(cache_db: Optional[str], no_cache: bool, anchor_dir: Path) -> Cache:
    if no_cache:
        return Cache(Path(":memory:"))
    db = Path(cache_db).expanduser().resolve() if cache_db else default_cache_path(anchor_dir)
    logger.info("Using cache: %s", db)
    return Cache(db)


def _write_outcomes(unkept_dir: Path, outcomes: dict[Path, str],
                    roots_map: dict[Path, Path]) -> Path:
    """Record every rejected file and why, next to the UNKEPT mirror.

    Columns: rel_path (position inside UNKEPT), reason, original_path.
    The GUI review tab uses this to caption rejected images.
    """
    out = unkept_dir / OUTCOMES_CSV
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "reason", "original_path"])
        for p, reason in sorted(outcomes.items(), key=lambda kv: str(kv[0])):
            try:
                rel = p.relative_to(roots_map[p]).as_posix()
            except Exception:
                rel = p.name
            w.writerow([rel, reason, str(p)])
    return out


def run_full_pipeline(
    input_dirs: list,
    kept_dir,
    unkept_dir,
    *,
    copy_instead: bool = True,
    overwrite_unkept: bool = False,
    device: str = "auto",
    face_backend: str = "auto",
    yolo_conf: float = 0.25,
    yolo_batch: int = 32,
    similarity: float = 0.90,
    phash_hamming: int = 30,
    clip_batch: int = 256,
    clip_max_size: int = 512,
    use_faiss: bool = False,
    faiss_k: int = 50,
    rep_policy: str = "sharpest",
    kept_side: int = 300,
    jpeg_quality: int = 92,
    clean_kept: bool = True,
    skip_detect: bool = False,
    min_sharpness: float = 0.0,
    drop_grayscale: bool = False,
    drop_bordered: bool = False,
    cache_db: Optional[str] = None,
    no_cache: bool = False,
) -> dict:
    """Full pipeline: detect people/faces -> dedup -> export.

    Returns a summary dict (same shape the CLI prints as JSON).
    """
    # Preflight optional dependencies BEFORE any long stage runs, so a
    # missing package fails in a second instead of mid-pipeline.
    if use_faiss:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ValueError(
                "FAISS was requested but faiss is not installed. Either "
                "install it (pip install faiss-cpu) or turn the FAISS "
                "option off — the pHash method works fine below ~50k images."
            )

    all_imgs, roots_map = collect_inputs(input_dirs)
    if not all_imgs:
        logger.warning("No images to process.")
        return {"total_input": 0, "exported": 0}

    kept_dir = normalize_path(kept_dir)
    unkept_dir = normalize_path(unkept_dir)
    ensure_dir(kept_dir)
    ensure_dir(unkept_dir)

    device = pick_device(device)
    logger.info("Device: %s", device)

    cache = _open_cache(cache_db, no_cache, anchor_dir=kept_dir)

    try:
        rejected: set[Path] = set()
        outcomes: dict[Path, str] = {}

        # ----- Stage 0: quality filter (blur / grayscale / borders) -----
        # Runs first: it's far cheaper than YOLO, so pruning here saves work.
        if min_sharpness > 0 or drop_grayscale or drop_bordered:
            from lsict.quality import run_quality_filter
            qrej = run_quality_filter(
                all_imgs, cache,
                min_sharpness=min_sharpness,
                drop_grayscale=drop_grayscale,
                drop_bordered=drop_bordered,
            )
            for p, reason in qrej.items():
                mirror_to_unkept(p, roots_map[p], unkept_dir,
                                 copy_instead, overwrite_unkept)
                rejected.add(p)
                outcomes[p] = reason
            logger.info("Quality filter rejected %d image(s).", len(qrej))

        # ----- Stage 1: detect -----
        if not skip_detect:
            from lsict.detect import run_detection
            det = run_detection(
                files=[p for p in all_imgs if p not in rejected],
                cache=cache,
                device=device,
                face_backend=face_backend,
                yolo_batch=yolo_batch,
                yolo_conf=yolo_conf,
            )
            for p, info in det.items():
                if info["has_person"]:
                    mirror_to_unkept(p, roots_map[p], unkept_dir,
                                     copy_instead, overwrite_unkept)
                    rejected.add(p)
                    outcomes[p] = "person"
                elif info["has_face"]:
                    mirror_to_unkept(p, roots_map[p], unkept_dir,
                                     copy_instead, overwrite_unkept)
                    rejected.add(p)
                    outcomes[p] = "face"
        else:
            logger.info("Skipping detect stage (skip_detect)")

        # ----- Stage 2: dedup -----
        survivors = [p for p in all_imgs if p not in rejected]
        # Re-check existence (some may have been moved)
        survivors = [p for p in survivors if p.exists()]
        logger.info("Survivors entering dedup: %d", len(survivors))

        from lsict.dedup import (
            find_exact_duplicates,
            find_near_duplicates_phash,
            find_near_duplicates_faiss,
            pick_representative,
        )
        exact_groups = find_exact_duplicates(survivors, cache)
        for sha, group in exact_groups.items():
            rep = pick_representative(group, policy=rep_policy)
            for f in group:
                if f != rep:
                    mirror_to_unkept(f, roots_map[f], unkept_dir,
                                     copy_instead, overwrite_unkept)
                    rejected.add(f)
                    outcomes[f] = "duplicate_exact"

        survivors = [p for p in survivors if p not in rejected]
        survivors = [p for p in survivors if p.exists()]

        if use_faiss:
            near_groups = find_near_duplicates_faiss(
                survivors, cache, device,
                similarity=similarity,
                clip_batch=clip_batch,
                clip_max_size=clip_max_size,
                faiss_k=faiss_k,
            )
        else:
            near_groups = find_near_duplicates_phash(
                survivors, cache, device,
                phash_hamming=phash_hamming,
                similarity=similarity,
                clip_batch=clip_batch,
                clip_max_size=clip_max_size,
            )
        for group in near_groups:
            rep = pick_representative(group, policy=rep_policy)
            for f in group:
                if f != rep:
                    mirror_to_unkept(f, roots_map[f], unkept_dir,
                                     copy_instead, overwrite_unkept)
                    rejected.add(f)
                    outcomes[f] = "duplicate_near"

        # ----- Stage 3: export -----
        final_survivors = [p for p in all_imgs if p not in rejected and p.exists()]
        logger.info("Final survivors to export: %d", len(final_survivors))

        from lsict.export import export_numbered
        count, manifest = export_numbered(
            final_survivors,
            kept_dir,
            side=kept_side,
            jpeg_quality=jpeg_quality,
            clean_first=clean_kept,
        )

        outcomes_csv = _write_outcomes(unkept_dir, outcomes, roots_map)

        return {
            "total_input": len(all_imgs),
            "rejected_blurry": sum(1 for v in outcomes.values() if v == "blurry"),
            "rejected_grayscale": sum(1 for v in outcomes.values() if v == "grayscale"),
            "rejected_bordered": sum(1 for v in outcomes.values() if v == "bordered"),
            "rejected_person": sum(1 for v in outcomes.values() if v == "person"),
            "rejected_face": sum(1 for v in outcomes.values() if v == "face"),
            "rejected_exact_dup": sum(1 for v in outcomes.values() if v == "duplicate_exact"),
            "rejected_near_dup": sum(1 for v in outcomes.values() if v == "duplicate_near"),
            "exported": count,
            "kept_dir": str(kept_dir),
            "unkept_dir": str(unkept_dir),
            "manifest_csv": str(manifest),
            "outcomes_csv": str(outcomes_csv),
            "cache_info": cache.info() if cache else None,
        }
    finally:
        cache.close()


def run_nsfw_screen(
    src_dir,
    dst_dir,
    *,
    device: str = "auto",
    backend: str = "auto",
    threshold: float = 0.5,
    batch: int = 32,
    copy: bool = False,
    cache_db: Optional[str] = None,
    no_cache: bool = False,
    cache_anchor=None,
) -> dict:
    """NSFW screen: move/copy only SAFE images from src_dir to dst_dir."""
    src = normalize_path(src_dir)
    dst = normalize_path(dst_dir)
    if not src.exists():
        raise ValueError(f"Source folder does not exist: {src}")
    device = pick_device(device)
    anchor = normalize_path(cache_anchor) if cache_anchor else dst
    cache = _open_cache(cache_db, no_cache, anchor_dir=anchor)
    try:
        from lsict.nsfw import screen_safe
        return screen_safe(
            src_dir=src,
            dst_dir=dst,
            cache=cache,
            device=device,
            threshold=threshold,
            backend=backend,
            batch=batch,
            copy=copy,
        )
    finally:
        cache.close()
