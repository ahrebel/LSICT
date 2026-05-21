"""imgcurate CLI.

Subcommands:
  run     Full pipeline: detect people/faces -> dedup -> export
  detect  Just people/face filtering -> UNKEPT
  dedup   Just dedup -> UNKEPT
  export  Just numbered 300x300 export
  nsfw    NSFW screen — move SAFE to FinalImageSet
  seed    Seed a diverse set from Open Images v7
  cache   Inspect / prune / clear the SQLite cache
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from imgcurate import __version__
from imgcurate.cache import Cache, default_cache_path
from imgcurate.core import (
    HEIC_AVAILABLE,
    IS_WINDOWS,
    ensure_dir,
    is_image_file,
    mirror_to_unkept,
    normalize_path,
    pick_device,
    scan_images,
)


# ---------- logging setup ----------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down chatty libraries
    for noisy in ("PIL", "matplotlib", "fiftyone", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------- arg parsing helpers ----------

def add_common_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", action="append", required=True,
                   help="Input folder (repeatable for multiple roots)")


def add_unkept_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--unkept", required=True,
                   help="Folder to mirror rejected files into")
    p.add_argument("--copy-instead", action="store_true",
                   help="Copy instead of move to UNKEPT (recommended for safety)")
    p.add_argument("--overwrite-unkept", action="store_true",
                   help="Overwrite same-name files in UNKEPT instead of adding ' (1)'")


def add_device_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                   help="Compute device for YOLO/CLIP/NSFW (default: auto)")


def add_cache_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache-db", default=None,
                   help="Path to SQLite cache (default: <kept>/.imgcurate_cache.sqlite)")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable cache (always recompute everything)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imgcurate",
        description="Large-scale image curation: dedup, people/face filter, "
                    "NSFW screen, square-crop export.",
    )
    p.add_argument("--version", action="version", version=f"imgcurate {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = p.add_subparsers(dest="command", required=True)

    # run
    pr = sub.add_parser("run", help="Full pipeline: detect -> dedup -> export")
    add_common_input_args(pr)
    pr.add_argument("--kept", required=True, help="Output folder for 300x300 numbered JPEGs")
    add_unkept_args(pr)
    add_device_args(pr)
    add_cache_args(pr)
    pr.add_argument("--face-backend", choices=["mediapipe", "haar", "auto", "off"],
                    default="auto",
                    help="Face detector. 'auto' uses mediapipe if installed, else haar")
    pr.add_argument("--yolo-conf", type=float, default=0.25,
                    help="YOLO person-confidence threshold")
    pr.add_argument("--yolo-batch", type=int, default=32, help="YOLO batch size")
    pr.add_argument("--similarity", type=float, default=0.90,
                    help="CLIP cosine threshold for near-dup grouping (0..1)")
    pr.add_argument("--phash-hamming", type=int, default=30,
                    help="pHash Hamming threshold for candidate pairing (0..256)")
    pr.add_argument("--clip-batch", type=int, default=256, help="CLIP batch size")
    pr.add_argument("--clip-max-size", type=int, default=512,
                    help="Resize long edge before CLIP preprocess")
    pr.add_argument("--use-faiss", action="store_true",
                    help="Use FAISS HNSW over CLIP embeddings instead of pHash bucketing "
                         "(recommended for >50k images; requires faiss-cpu)")
    pr.add_argument("--faiss-k", type=int, default=50,
                    help="FAISS: top-k neighbors per image (default 50)")
    pr.add_argument("--rep-policy",
                    choices=["sharpest", "largest", "newest", "oldest", "first"],
                    default="sharpest",
                    help="How to pick a representative from a duplicate group")
    pr.add_argument("--kept-side", type=int, default=300, help="Output JPEG side (px)")
    pr.add_argument("--jpeg-quality", type=int, default=92)
    pr.add_argument("--no-clean-kept", action="store_true",
                    help="Don't wipe existing JPEGs in KEPT before export")
    pr.add_argument("--skip-detect", action="store_true",
                    help="Skip people/face detection (e.g., when resuming)")
    pr.set_defaults(func=cmd_run)

    # detect
    pd = sub.add_parser("detect", help="Just people/face detection -> UNKEPT")
    add_common_input_args(pd)
    add_unkept_args(pd)
    add_device_args(pd)
    add_cache_args(pd)
    pd.add_argument("--cache-db-required", action="store_true",
                    help=argparse.SUPPRESS)  # unused, here so detect requires --cache-db OR uses one
    pd.add_argument("--cache-anchor", required=False, default=None,
                    help="Folder where the cache should live (default: alongside --unkept)")
    pd.add_argument("--face-backend", choices=["mediapipe", "haar", "auto", "off"],
                    default="auto")
    pd.add_argument("--yolo-conf", type=float, default=0.25)
    pd.add_argument("--yolo-batch", type=int, default=32)
    pd.set_defaults(func=cmd_detect)

    # dedup
    pdd = sub.add_parser("dedup", help="Just deduplication -> UNKEPT")
    add_common_input_args(pdd)
    add_unkept_args(pdd)
    add_device_args(pdd)
    add_cache_args(pdd)
    pdd.add_argument("--cache-anchor", required=False, default=None)
    pdd.add_argument("--similarity", type=float, default=0.90)
    pdd.add_argument("--phash-hamming", type=int, default=30)
    pdd.add_argument("--clip-batch", type=int, default=256)
    pdd.add_argument("--clip-max-size", type=int, default=512)
    pdd.add_argument("--use-faiss", action="store_true")
    pdd.add_argument("--faiss-k", type=int, default=50)
    pdd.add_argument("--rep-policy",
                     choices=["sharpest", "largest", "newest", "oldest", "first"],
                     default="sharpest")
    pdd.set_defaults(func=cmd_dedup)

    # export
    pe = sub.add_parser("export", help="Center-crop survivors -> numbered 300x300 JPEGs")
    add_common_input_args(pe)
    pe.add_argument("--unkept", required=True,
                    help="UNKEPT folder used as a mask to exclude already-rejected files")
    pe.add_argument("--kept", required=True)
    pe.add_argument("--kept-side", type=int, default=300)
    pe.add_argument("--jpeg-quality", type=int, default=92)
    pe.add_argument("--no-clean-kept", action="store_true")
    pe.set_defaults(func=cmd_export)

    # nsfw
    pn = sub.add_parser("nsfw", help="NSFW screen -> move SAFE images to dst")
    pn.add_argument("--src", required=True, help="Folder to scan recursively")
    pn.add_argument("--dst", required=True, help="Destination for SAFE images")
    add_device_args(pn)
    add_cache_args(pn)
    pn.add_argument("--cache-anchor", default=None)
    pn.add_argument("--backend", choices=["classifier", "clip", "auto"], default="auto",
                    help="'classifier' is more accurate (needs transformers); "
                         "'clip' is zero-shot, no extra deps; 'auto' prefers classifier")
    pn.add_argument("--threshold", type=float, default=0.5,
                    help="NSFW prob >= threshold => UNSAFE. Lower = stricter")
    pn.add_argument("--batch", type=int, default=32)
    pn.add_argument("--copy", action="store_true", help="Copy instead of move")
    pn.set_defaults(func=cmd_nsfw)

    # seed
    ps = sub.add_parser("seed", help="Seed a diverse set from Open Images v7")
    ps.add_argument("--output", required=True, help="Where to write the cropped JPEGs")
    ps.add_argument("--num-categories", type=int, default=1000)
    ps.add_argument("--images-per-category", type=int, default=2)
    ps.add_argument("--side", type=int, default=300)
    ps.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    ps.set_defaults(func=cmd_seed)

    # cache
    pc = sub.add_parser("cache", help="Inspect / prune / clear the SQLite cache")
    pc.add_argument("--db", required=True, help="Path to cache db")
    pc.add_argument("action", choices=["info", "prune", "clear"])
    pc.set_defaults(func=cmd_cache)

    return p


# ---------- command implementations ----------

def _collect_inputs(input_args: list[str]) -> tuple[list[Path], dict[Path, Path]]:
    """Resolve --input args, validate, scan all images, build roots_map."""
    roots = [normalize_path(p) for p in input_args]
    for r in roots:
        if not r.exists():
            logging.error("Input root does not exist: %s", r)
            sys.exit(2)

    all_imgs: list[Path] = []
    roots_map: dict[Path, Path] = {}
    for p, root in scan_images(roots):
        all_imgs.append(p)
        roots_map[p] = root
    logging.info("Scanned %d images across %d input root(s).", len(all_imgs), len(roots))
    if not HEIC_AVAILABLE:
        n_heic = sum(1 for p in all_imgs if p.suffix.lower() in {".heic", ".heif"})
        if n_heic:
            logging.warning(
                "%d HEIC/HEIF files found but pillow-heif not installed — they will be skipped.",
                n_heic,
            )
    return all_imgs, roots_map


def _open_cache(args, anchor_dir: Path) -> Optional[Cache]:
    if getattr(args, "no_cache", False):
        return None
    db = args.cache_db or default_cache_path(anchor_dir)
    db = Path(db).expanduser().resolve()
    logging.info("Using cache: %s", db)
    return Cache(db)


def cmd_run(args) -> int:
    """Full pipeline."""
    all_imgs, roots_map = _collect_inputs(args.input)
    if not all_imgs:
        logging.warning("No images to process.")
        return 0

    kept_dir = normalize_path(args.kept)
    unkept_dir = normalize_path(args.unkept)
    ensure_dir(kept_dir)
    ensure_dir(unkept_dir)

    device = pick_device(args.device)
    logging.info("Device: %s", device)

    cache = _open_cache(args, anchor_dir=kept_dir)
    if cache is None:
        # Build an in-memory cache so the modules still work, but it won't persist
        cache = Cache(Path(":memory:"))

    try:
        rejected: set[Path] = set()
        outcomes: dict[Path, str] = {}

        # ----- Stage 1: detect -----
        if not args.skip_detect:
            from imgcurate.detect import run_detection
            det = run_detection(
                files=all_imgs,
                cache=cache,
                device=device,
                face_backend=args.face_backend,
                yolo_batch=args.yolo_batch,
                yolo_conf=args.yolo_conf,
            )
            for p, info in det.items():
                if info["has_person"]:
                    mirror_to_unkept(p, roots_map[p], unkept_dir,
                                     args.copy_instead, args.overwrite_unkept)
                    rejected.add(p)
                    outcomes[p] = "person"
                elif info["has_face"]:
                    mirror_to_unkept(p, roots_map[p], unkept_dir,
                                     args.copy_instead, args.overwrite_unkept)
                    rejected.add(p)
                    outcomes[p] = "face"
        else:
            logging.info("Skipping detect stage (--skip-detect)")

        # ----- Stage 2: dedup -----
        survivors = [p for p in all_imgs if p not in rejected]
        # Re-check existence (some may have been moved)
        survivors = [p for p in survivors if p.exists()]
        logging.info("Survivors entering dedup: %d", len(survivors))

        from imgcurate.dedup import (
            find_exact_duplicates,
            find_near_duplicates_phash,
            find_near_duplicates_faiss,
            pick_representative,
        )
        exact_groups = find_exact_duplicates(survivors, cache)
        for sha, group in exact_groups.items():
            rep = pick_representative(group, policy=args.rep_policy)
            for f in group:
                if f != rep:
                    mirror_to_unkept(f, roots_map[f], unkept_dir,
                                     args.copy_instead, args.overwrite_unkept)
                    rejected.add(f)
                    outcomes[f] = "duplicate_exact"

        survivors = [p for p in survivors if p not in rejected]
        survivors = [p for p in survivors if p.exists()]

        if args.use_faiss:
            near_groups = find_near_duplicates_faiss(
                survivors, cache, device,
                similarity=args.similarity,
                clip_batch=args.clip_batch,
                clip_max_size=args.clip_max_size,
                faiss_k=args.faiss_k,
            )
        else:
            near_groups = find_near_duplicates_phash(
                survivors, cache, device,
                phash_hamming=args.phash_hamming,
                similarity=args.similarity,
                clip_batch=args.clip_batch,
                clip_max_size=args.clip_max_size,
            )
        for group in near_groups:
            rep = pick_representative(group, policy=args.rep_policy)
            for f in group:
                if f != rep:
                    mirror_to_unkept(f, roots_map[f], unkept_dir,
                                     args.copy_instead, args.overwrite_unkept)
                    rejected.add(f)
                    outcomes[f] = "duplicate_near"

        # ----- Stage 3: export -----
        final_survivors = [p for p in all_imgs if p not in rejected and p.exists()]
        logging.info("Final survivors to export: %d", len(final_survivors))

        from imgcurate.export import export_numbered
        count, manifest = export_numbered(
            final_survivors,
            kept_dir,
            side=args.kept_side,
            jpeg_quality=args.jpeg_quality,
            clean_first=not args.no_clean_kept,
        )

        summary = {
            "total_input": len(all_imgs),
            "rejected_person": sum(1 for v in outcomes.values() if v == "person"),
            "rejected_face": sum(1 for v in outcomes.values() if v == "face"),
            "rejected_exact_dup": sum(1 for v in outcomes.values() if v == "duplicate_exact"),
            "rejected_near_dup": sum(1 for v in outcomes.values() if v == "duplicate_near"),
            "exported": count,
            "kept_dir": str(kept_dir),
            "unkept_dir": str(unkept_dir),
            "manifest_csv": str(manifest),
            "cache_info": cache.info() if cache else None,
        }
        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        if cache is not None:
            cache.close()


def cmd_detect(args) -> int:
    all_imgs, roots_map = _collect_inputs(args.input)
    if not all_imgs:
        return 0
    unkept_dir = normalize_path(args.unkept)
    ensure_dir(unkept_dir)
    device = pick_device(args.device)
    anchor = normalize_path(args.cache_anchor) if args.cache_anchor else unkept_dir
    cache = _open_cache(args, anchor) or Cache(Path(":memory:"))
    try:
        from imgcurate.detect import run_detection
        det = run_detection(
            files=all_imgs,
            cache=cache,
            device=device,
            face_backend=args.face_backend,
            yolo_batch=args.yolo_batch,
            yolo_conf=args.yolo_conf,
        )
        moved = 0
        for p, info in det.items():
            if info["has_person"] or info["has_face"]:
                if mirror_to_unkept(p, roots_map[p], unkept_dir,
                                    args.copy_instead, args.overwrite_unkept):
                    moved += 1
        print(json.dumps({"total": len(all_imgs), "moved_to_unkept": moved}, indent=2))
        return 0
    finally:
        cache.close()


def cmd_dedup(args) -> int:
    all_imgs, roots_map = _collect_inputs(args.input)
    if not all_imgs:
        return 0
    unkept_dir = normalize_path(args.unkept)
    ensure_dir(unkept_dir)
    device = pick_device(args.device)
    anchor = normalize_path(args.cache_anchor) if args.cache_anchor else unkept_dir
    cache = _open_cache(args, anchor) or Cache(Path(":memory:"))

    try:
        from imgcurate.dedup import (
            find_exact_duplicates,
            find_near_duplicates_phash,
            find_near_duplicates_faiss,
            pick_representative,
        )
        rejected: set[Path] = set()
        # Exact
        exact_groups = find_exact_duplicates(all_imgs, cache)
        for group in exact_groups.values():
            rep = pick_representative(group, policy=args.rep_policy)
            for f in group:
                if f != rep:
                    mirror_to_unkept(f, roots_map[f], unkept_dir,
                                     args.copy_instead, args.overwrite_unkept)
                    rejected.add(f)
        survivors = [p for p in all_imgs if p not in rejected and p.exists()]

        # Near
        if args.use_faiss:
            near_groups = find_near_duplicates_faiss(
                survivors, cache, device,
                similarity=args.similarity,
                clip_batch=args.clip_batch,
                clip_max_size=args.clip_max_size,
                faiss_k=args.faiss_k,
            )
        else:
            near_groups = find_near_duplicates_phash(
                survivors, cache, device,
                phash_hamming=args.phash_hamming,
                similarity=args.similarity,
                clip_batch=args.clip_batch,
                clip_max_size=args.clip_max_size,
            )
        for group in near_groups:
            rep = pick_representative(group, policy=args.rep_policy)
            for f in group:
                if f != rep:
                    mirror_to_unkept(f, roots_map[f], unkept_dir,
                                     args.copy_instead, args.overwrite_unkept)
                    rejected.add(f)

        print(json.dumps({
            "total": len(all_imgs),
            "rejected_exact": sum(len(g) - 1 for g in exact_groups.values()),
            "rejected_near": sum(len(g) - 1 for g in near_groups),
            "survivors": len(all_imgs) - len(rejected),
        }, indent=2))
        return 0
    finally:
        cache.close()


def cmd_export(args) -> int:
    """Export survivors (= input minus what's in UNKEPT)."""
    all_imgs, roots_map = _collect_inputs(args.input)
    unkept_dir = normalize_path(args.unkept)
    kept_dir = normalize_path(args.kept)
    ensure_dir(kept_dir)

    # Build the set of relative paths present in UNKEPT
    unkept_rels: set[str] = set()
    if unkept_dir.exists():
        for p in tqdm(list(unkept_dir.rglob("*")), desc="Index UNKEPT"):
            if p.is_file() and is_image_file(p):
                try:
                    unkept_rels.add(p.relative_to(unkept_dir).as_posix())
                except Exception:
                    pass

    survivors: list[Path] = []
    for p in all_imgs:
        rel = p.relative_to(roots_map[p]).as_posix()
        if rel not in unkept_rels:
            survivors.append(p)

    from imgcurate.export import export_numbered
    count, manifest = export_numbered(
        survivors,
        kept_dir,
        side=args.kept_side,
        jpeg_quality=args.jpeg_quality,
        clean_first=not args.no_clean_kept,
    )
    print(json.dumps({
        "total_input": len(all_imgs),
        "unkept_mirrored": len(unkept_rels),
        "exported": count,
        "manifest": str(manifest),
    }, indent=2))
    return 0


def cmd_nsfw(args) -> int:
    src = normalize_path(args.src)
    dst = normalize_path(args.dst)
    device = pick_device(args.device)
    anchor = normalize_path(args.cache_anchor) if args.cache_anchor else dst
    cache = _open_cache(args, anchor) or Cache(Path(":memory:"))
    try:
        from imgcurate.nsfw import screen_safe
        result = screen_safe(
            src_dir=src,
            dst_dir=dst,
            cache=cache,
            device=device,
            threshold=args.threshold,
            backend=args.backend,
            batch=args.batch,
            copy=args.copy,
        )
        print(json.dumps(result, indent=2))
        return 0
    finally:
        cache.close()


def cmd_seed(args) -> int:
    from imgcurate.seed import run_seed
    result = run_seed(
        output_dir=normalize_path(args.output),
        num_categories=args.num_categories,
        images_per_category=args.images_per_category,
        image_side=args.side,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_cache(args) -> int:
    db = Path(args.db).expanduser().resolve()
    if not db.exists() and args.action != "info":
        logging.error("Cache db not found: %s", db)
        return 2
    cache = Cache(db)
    try:
        if args.action == "info":
            print(json.dumps(cache.info(), indent=2))
        elif args.action == "prune":
            n = cache.prune_missing()
            print(json.dumps({"pruned": n, **cache.info()}, indent=2))
        elif args.action == "clear":
            cache.clear()
            print(json.dumps({"cleared": True, **cache.info()}, indent=2))
    finally:
        cache.close()
    return 0


# ---------- entry ----------

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    logging.info("imgcurate %s on %s",
                 __version__, "Windows" if IS_WINDOWS else sys.platform)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
