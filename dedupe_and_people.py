#!/usr/bin/env python3
"""
Incremental-safe pipeline to build:
- KEPT (no people, no duplicates) -> numbered 300x300 JPGs
- UNKEPT (people + duplicates) -> mirrored folder tree of the inputs

Now supports MULTIPLE input roots:
  --input <dir1> --input "<dir2 with spaces>" [--input <dir3> ...]

Incremental behavior:
- UNKEPT: If a destination file already exists:
    * If same SHA-256: skip (no duplicate copy).
    * If different: write "<name> (1).ext", "<name> (2).ext", ... unless --overwrite-unkept is set.
- KEPT: Rebuilt each run (old files removed) unless --no-clean-kept is set.

Writes CSV at KEPT/run_log.csv:
    original_path, outcome, kept_index, unkept_relpath

Dependencies:
    pip install pillow imagehash numpy tqdm torch open-clip-torch ultralytics opencv-python
"""

import argparse, os, shutil, hashlib, json, csv
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from tqdm import tqdm
import numpy as np

from PIL import Image, UnidentifiedImageError
import imagehash
import torch
import open_clip

# --- People detection ---
_HAS_ULTRALYTICS = True
try:
    from ultralytics import YOLO
except Exception:
    _HAS_ULTRALYTICS = False

import cv2

# -------------------- Utilities --------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def clear_dir(d: Path):
    if not d.exists():
        return
    for child in d.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)

def safe_open_pil(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def next_nonconflicting_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    n = 1
    while True:
        candidate = dst.with_name(f"{stem} ({n}){ext}")
        if not candidate.exists():
            return candidate
        n += 1

def copy_or_move_preserving_tree_incremental(
    src: Path,
    src_root: Path,
    dst_root: Path,
    copy_instead: bool,
    overwrite_unkept: bool
) -> Path:
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    ensure_dir(dst.parent)

    if dst.exists():
        try:
            if file_sha256(src) == file_sha256(dst):
                return dst  # identical; skip
        except Exception:
            pass
        if overwrite_unkept:
            if copy_instead:
                shutil.copy2(str(src), str(dst))
            else:
                dst.unlink(missing_ok=True)
                shutil.move(str(src), str(dst))
            return dst
        else:
            dst = next_nonconflicting_path(dst)

    if copy_instead:
        shutil.copy2(str(src), str(dst))
    else:
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
    return dst

def device_pick() -> torch.device:
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -------------------- CLIP & pHash --------------------

def load_clip(device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    return model, preprocess

def image_to_clip(im: Image.Image, preprocess, device: torch.device):
    return preprocess(im).unsqueeze(0).to(device)

def embed_images_clip(paths: List[Path], preprocess, clip_model, device, max_size=512, batch=256) -> np.ndarray:
    embs: List[np.ndarray] = []
    try:
        out_dim = clip_model.visual.output_dim
    except Exception:
        out_dim = 512

    for i in tqdm(range(0, len(paths), batch), desc="CLIP embedding"):
        slice_paths = paths[i:i+batch]
        imgs = []
        val_idx = []
        for j, p in enumerate(slice_paths):
            im = safe_open_pil(p)
            if im is None:
                imgs.append(None)
            else:
                im.thumbnail((max_size, max_size), Image.BICUBIC)
                imgs.append(im)
                val_idx.append(j)

        if not val_idx:
            embs.extend([np.zeros(out_dim, dtype=np.float32)] * len(slice_paths))
            continue

        input_t = torch.cat([image_to_clip(imgs[j], preprocess, device) for j in val_idx], dim=0)
        with torch.no_grad():
            feats = clip_model.encode_image(input_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        out = feats.detach().cpu().numpy().astype(np.float32)
        k = 0
        for j in range(len(slice_paths)):
            if imgs[j] is None:
                embs.append(np.zeros(out.shape[1], dtype=np.float32))
            else:
                embs.append(out[k]); k += 1

    return np.vstack(embs)

def compute_phash(path: Path, hash_size: int = 16) -> Optional[int]:
    im = safe_open_pil(path)
    if im is None:
        return None
    try:
        h = imagehash.phash(im, hash_size=hash_size)  # 256-bit
        return int(str(h), 16)
    except Exception:
        return None

def hamming_distance_256(a: int, b: int) -> int:
    return (a ^ b).bit_count()

class UnionFind:
    def __init__(self, items: List):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
    def components(self):
        groups = {}
        for x in self.parent:
            r = self.find(x)
            groups.setdefault(r, set()).add(x)
        return list(groups.values())

def pick_representative(files: List[Path], policy="largest") -> Path:
    if policy == "largest":
        return max(files, key=lambda p: p.stat().st_size if p.exists() else -1)
    elif policy == "newest":
        return max(files, key=lambda p: p.stat().st_mtime if p.exists() else -1)
    elif policy == "oldest":
        return min(files, key=lambda p: p.stat().st_mtime if p.exists() else float("inf"))
    return files[0]

# -------------------- People / Face detection --------------------

def load_detectors():
    yolo = None
    if _HAS_ULTRALYTICS:
        try:
            yolo = YOLO("yolov8n.pt")  # downloads first time
        except Exception:
            yolo = None
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(face_cascade_path) if os.path.exists(face_cascade_path) else None
    return yolo, face

def has_face_opencv(img_path: Path, face: Optional[cv2.CascadeClassifier], min_size=(40, 40)) -> bool:
    if face is None:
        return False
    try:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        faces = face.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4, minSize=min_size)
        return len(faces) > 0
    except Exception:
        return False

def person_or_face_filter(
    files: List[Path],
    roots_map: Dict[Path, Path],
    unkept_root: Path,
    copy_instead: bool,
    overwrite_unkept: bool,
    batch: int = 64,
    yolo_conf: float = 0.25
) -> Tuple[Set[Path], Dict[Path, str]]:
    moved: Set[Path] = set()
    reasons: Dict[Path, str] = {}
    yolo, face_cv = load_detectors()

    # YOLO person detection
    if yolo is not None and len(files) > 0:
        pbar = tqdm(total=len(files), desc="YOLO people detect", unit="img")
        for i in range(0, len(files), batch):
            chunk = files[i:i+batch]
            try:
                results = yolo([str(p) for p in chunk], conf=yolo_conf, classes=[0])  # class 0 = 'person'
            except Exception:
                results = [None] * len(chunk)

            for p, res in zip(chunk, results):
                has_person = False
                try:
                    if res is not None and getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
                        has_person = True
                except Exception:
                    has_person = False

                if has_person and p not in moved:
                    copy_or_move_preserving_tree_incremental(
                        p, roots_map[p], unkept_root, copy_instead, overwrite_unkept
                    )
                    moved.add(p)
                    reasons[p] = "person"
            pbar.update(len(chunk))
        pbar.close()

    # Face-only fallback
    remaining = [p for p in files if p not in moved]
    if face_cv is not None and len(remaining) > 0:
        for p in tqdm(remaining, desc="Face detect (OpenCV)", unit="img"):
            try:
                if has_face_opencv(p, face_cv):
                    copy_or_move_preserving_tree_incremental(
                        p, roots_map[p], unkept_root, copy_instead, overwrite_unkept
                    )
                    moved.add(p)
                    reasons[p] = "face"
            except Exception:
                pass

    return moved, reasons

# -------------------- Image export --------------------

def center_crop_square_and_resize(im: Image.Image, side: int = 300) -> Image.Image:
    w, h = im.size
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

def export_numbered_kept(kept_paths: List[Path], kept_dir: Path, side: int = 300, jpeg_quality: int = 92) -> int:
    ensure_dir(kept_dir)
    count = 0
    for i, p in enumerate(tqdm(sorted(kept_paths, key=lambda x: str(x)), desc="Export KEPT -> 300x300"), start=1):
        im = safe_open_pil(p)
        if im is None:
            continue
        im_proc = center_crop_square_and_resize(im, side=side)
        out = kept_dir / f"{i}.jpg"
        im_proc.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
        count += 1
    return count

# -------------------- Main pipeline --------------------

def main():
    ap = argparse.ArgumentParser(description="Create KEPT (no people, no duplicates) and UNKEPT (people + duplicates; mirrored; incremental-safe). Supports multiple --input.")
    ap.add_argument("--input", action="append", required=True, help="Input folder with images. Repeat for multiple roots.")
    ap.add_argument("--kept", required=True, help="Output folder for numbered 300x300 kept images")
    ap.add_argument("--unkept", required=True, help="Output folder for people and duplicates (mirrors input structure)")
    ap.add_argument("--similarity", type=float, default=0.90, help="CLIP cosine similarity threshold (0-1)")
    ap.add_argument("--phash-hamming", type=int, default=30, help="pHash Hamming threshold for candidate pairing (0-256)")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for CLIP embedding")
    ap.add_argument("--max-size", type=int, default=512, help="Resize long edge <= this before CLIP preprocess")
    ap.add_argument("--rep-policy", choices=["largest","newest","oldest","first"], default="largest",
                    help="Representative selection policy for duplicate groups")
    ap.add_argument("--copy-instead", action="store_true", help="Copy to UNKEPT instead of moving (recommended)")
    ap.add_argument("--overwrite-unkept", action="store_true", help="If a different file already exists at the mirrored UNKEPT path, overwrite it instead of creating ' (1).ext'.")
    ap.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for KEPT exports")
    ap.add_argument("--kept-side", type=int, default=300, help="Square size for KEPT exports (default 300)")
    ap.add_argument("--no-clean-kept", action="store_true", help="Do NOT wipe the KEPT folder before exporting (keeps old files; numbering may not be contiguous)")
    args = ap.parse_args()

    input_dirs = [Path(p).expanduser().resolve() for p in args.input]
    kept_dir   = Path(args.kept).expanduser().resolve()
    unkept_dir = Path(args.unkept).expanduser().resolve()
    ensure_dir(kept_dir); ensure_dir(unkept_dir)

    if not args.no_clean_kept:
        clear_dir(kept_dir)

    # Gather all images across ALL roots + map each file to its root
    roots_map: Dict[Path, Path] = {}
    all_imgs: List[Path] = []
    for root in input_dirs:
        for p in root.rglob("*"):
            if p.is_file() and is_image_file(p):
                all_imgs.append(p)
                roots_map[p] = root
    print(f"Found {len(all_imgs)} images across {len(input_dirs)} input roots.")

    # CSV log
    log_rows = []
    def log_entry(path: Path, outcome: str, kept_index: Optional[int]=None, unkept_rel: Optional[str]=None):
        log_rows.append({
            "original_path": str(path),
            "outcome": outcome,
            "kept_index": kept_index if kept_index is not None else "",
            "unkept_relpath": unkept_rel if unkept_rel is not None else ""
        })

    # 1) PEOPLE/FACE FILTER -> UNKEPT
    print("Stage 1/3: People/face filtering -> UNKEPT")
    moved_people, reasons_people = person_or_face_filter(
        files=all_imgs,
        roots_map=roots_map,
        unkept_root=unkept_dir,
        copy_instead=args.copy_instead,
        overwrite_unkept=args.overwrite_unkept,
        batch=64,
        yolo_conf=0.25
    )
    for p in moved_people:
        rel = p.relative_to(roots_map[p])
        log_entry(p, reasons_people.get(p, "person/face"), kept_index=None, unkept_rel=str(rel))

    # Remaining for dedup
    remaining = [p for p in all_imgs if p not in moved_people]

    # 2) DEDUP on remaining
    print("Stage 2/3: Dedup (SHA-256 exact -> pHash -> CLIP verify)")
    # 2a exact duplicates
    by_hash: Dict[str, List[Path]] = {}
    for p in tqdm(remaining, desc="Hashing SHA-256"):
        h = file_sha256(p); by_hash.setdefault(h, []).append(p)

    kept_paths: Set[Path] = set()
    for group in by_hash.values():
        if len(group) == 1:
            kept_paths.add(group[0])
        else:
            rep = pick_representative(group, policy=args.rep_policy)
            kept_paths.add(rep)
            for g in group:
                if g != rep:
                    dst = copy_or_move_preserving_tree_incremental(
                        g, roots_map[g], unkept_dir, args.copy_instead, args.overwrite_unkept
                    )
                    rel = g.relative_to(roots_map[g])
                    log_entry(g, "duplicate_exact", kept_index=None, unkept_rel=str(rel))

    # Recompute remaining (some moved to UNKEPT)
    remaining = [p for p in all_imgs if p not in moved_people and p not in {x for x in all_imgs if x not in kept_paths and any(x in v for v in by_hash.values())}]

    # 2b perceptual (pHash) candidates
    phashes: Dict[Path, int] = {}
    for p in tqdm(remaining, desc="Computing pHash"):
        hv = compute_phash(p, hash_size=16)
        if hv is not None:
            phashes[p] = hv

    # Bucket by top 32 bits
    buckets: Dict[int, List[Tuple[Path, int]]] = {}
    for p, hv in phashes.items():
        coarse = hv >> 224
        buckets.setdefault(coarse, []).append((p, hv))

    candidate_pairs: List[Tuple[Path, Path]] = []
    for items in tqdm(buckets.values(), desc="Bucket pair scan"):
        n = len(items)
        if n < 2: continue
        for i in range(n):
            pi, hi = items[i]
            for j in range(i+1, n):
                pj, hj = items[j]
                if hamming_distance_256(hi, hj) <= args.phash_hamming:
                    candidate_pairs.append((pi, pj))

    uf = UnionFind(list(phashes.keys()))
    for a, b in candidate_pairs:
        uf.union(a, b)
    candidate_sets = [list(s) for s in uf.components() if len(s) > 1]

    # CLIP verify
    device = device_pick()
    clip_model, preprocess = load_clip(device)

    for cand in tqdm(candidate_sets, desc="CLIP verify sets"):
        cand = [p for p in cand if p.exists()]
        if len(cand) < 2:
            continue
        embs = embed_images_clip(cand, preprocess, clip_model, device,
                                 max_size=args.max_size, batch=args.batch)
        sims = np.matmul(embs, embs.T)
        n = sims.shape[0]
        mask = (sims >= args.similarity).astype(np.uint8)
        np.fill_diagonal(mask, 0)

        comp = UnionFind(list(range(n)))
        for i in range(n):
            for j in range(i+1, n):
                if mask[i, j]:
                    comp.union(i, j)

        subgroups = [list(c) for c in comp.components() if len(c) > 1]
        for idxs in subgroups:
            files = [cand[k] for k in idxs if cand[k].exists()]
            if not files:
                continue
            rep = pick_representative(files, policy=args.rep_policy)
            kept_paths.add(rep)
            for f in files:
                if f != rep and f.exists():
                    copy_or_move_preserving_tree_incremental(
                        f, roots_map[f], unkept_dir, args.copy_instead, args.overwrite_unkept
                    )
                    rel = f.relative_to(roots_map[f])
                    log_entry(f, "duplicate_near", kept_index=None, unkept_rel=str(rel))

    # Survivors = not flagged to UNKEPT
    unkept_flagged = set([Path(row["original_path"]) for row in log_rows
                          if row["outcome"] in ("person","face","duplicate_exact","duplicate_near")])
    survivors = sorted([p for p in all_imgs if p not in unkept_flagged], key=lambda x: str(x))

    # 3) Export KEPT as numbered 300x300
    print("Stage 3/3: Export KEPT as 300x300 numbered JPEGs")
    kept_count = export_numbered_kept(survivors, kept_dir, side=args.kept_side, jpeg_quality=args.jpeg_quality)
    for idx, p in enumerate(survivors, start=1):
        log_entry(p, "kept", kept_index=idx, unkept_rel=None)

    # Write log
    ensure_dir(kept_dir)
    log_csv = kept_dir / "run_log.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["original_path","outcome","kept_index","unkept_relpath"])
        w.writeheader()
        for row in log_rows:
            w.writerow(row)

    summary = {
        "inputs": [str(d) for d in input_dirs],
        "kept_count": kept_count,
        "unkept_dir": str(unkept_dir),
        "kept_dir": str(kept_dir),
        "similarity": args.similarity,
        "phash_hamming": args.phash_hamming,
        "copy_instead": args.copy_instead,
        "overwrite_unkept": args.overwrite_unkept,
        "clean_kept": not args.no_clean_kept,
        "log_csv": str(log_csv)
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
