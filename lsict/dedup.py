"""Deduplication.

Three-stage:
  1. Exact dup detection via SHA-256 (cached).
  2. Perceptual near-dup candidate generation via pHash (cached, bucketed).
  3. Verification via CLIP cosine similarity (cached embeddings).

When `--use-faiss` is set and faiss-cpu is installed, we skip pHash bucketing
and use FAISS HNSW over CLIP embeddings directly for nearest-neighbor search.
This is dramatically faster on large (100k+) datasets and finds near-dups that
pHash bucketing misses (e.g., color shifts, mild cropping).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lsict.cache import Cache
from lsict.core import (
    file_sha256,
    image_sharpness,
    safe_open_pil,
)

logger = logging.getLogger(__name__)


# ---------- Hashing ----------

def compute_phash(path: Path, hash_size: int = 16) -> Optional[int]:
    """256-bit perceptual hash as an int. hash_size=16 yields 16*16=256 bits."""
    im = safe_open_pil(path)
    if im is None:
        return None
    try:
        h = imagehash.phash(im, hash_size=hash_size)
        return int(str(h), 16)
    except Exception:
        return None


def hamming256(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------- Union-Find ----------

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def components(self) -> list[list]:
        groups: dict = {}
        for x in self.parent:
            groups.setdefault(self.find(x), []).append(x)
        return list(groups.values())


# ---------- CLIP ----------

def load_clip(device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    return model, preprocess


def embed_images(
    paths: list[Path],
    cache: Cache,
    device: str,
    batch: int = 256,
    max_size: int = 512,
) -> np.ndarray:
    """Return (N, D) float32 normalized embeddings for `paths`.

    Cached embeddings are reused; only uncached files run through CLIP.
    """
    # 1) Load cached, find missing
    cached: dict[int, np.ndarray] = {}
    missing_idx: list[int] = []
    for i, p in enumerate(paths):
        e = cache.get_clip(p)
        if e is not None:
            cached[i] = e
        else:
            missing_idx.append(i)

    n_cached = len(cached)
    n_missing = len(missing_idx)
    if n_missing:
        logger.info(
            "CLIP: %d cached, %d to compute (%.1f%% cache hit)",
            n_cached, n_missing, 100.0 * n_cached / max(1, len(paths)),
        )
    else:
        logger.info("CLIP: all %d embeddings served from cache.", len(paths))

    # Probe dim
    dim = next(iter(cached.values())).shape[-1] if cached else None
    model = preprocess = None

    # 2) Compute missing in batches
    if missing_idx:
        model, preprocess = load_clip(device)
        try:
            out_dim = getattr(model.visual, "output_dim", 512)
        except Exception:
            out_dim = 512
        dim = out_dim if dim is None else dim

        pbar = tqdm(total=len(missing_idx), desc="CLIP embedding", unit="img")
        for i in range(0, len(missing_idx), batch):
            slice_ids = missing_idx[i:i + batch]
            tensors = []
            valid_ids = []
            for j in slice_ids:
                im = safe_open_pil(paths[j])
                if im is None:
                    cached[j] = np.zeros(dim, dtype=np.float32)
                    continue
                im.thumbnail((max_size, max_size), Image.BICUBIC)
                tensors.append(preprocess(im).unsqueeze(0))
                valid_ids.append(j)

            if tensors:
                with torch.no_grad():
                    batch_t = torch.cat(tensors, dim=0).to(device)
                    feats = model.encode_image(batch_t)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.detach().cpu().numpy().astype(np.float32)
                for k, j in enumerate(valid_ids):
                    cached[j] = feats_np[k]

            # Persist this batch
            with cache.transaction():
                for j in slice_ids:
                    if not np.allclose(cached[j], 0):
                        cache.set_clip(paths[j], cached[j])
            pbar.update(len(slice_ids))
        pbar.close()
        del model

    if dim is None:
        # No images at all
        return np.zeros((0, 512), dtype=np.float32)

    out = np.zeros((len(paths), dim), dtype=np.float32)
    for i in range(len(paths)):
        out[i] = cached.get(i, np.zeros(dim, dtype=np.float32))
    return out


# ---------- Exact duplicates ----------

def find_exact_duplicates(
    files: list[Path],
    cache: Cache,
) -> dict[str, list[Path]]:
    """Group files by SHA-256. Returns {sha: [paths]} (only groups with >1)."""
    by_sha: dict[str, list[Path]] = {}
    need_compute: list[Path] = []

    for p in files:
        sha = cache.get_sha256(p)
        if sha is None:
            need_compute.append(p)
        else:
            by_sha.setdefault(sha, []).append(p)

    if need_compute:
        logger.info(
            "SHA-256: %d cached, %d to compute",
            len(files) - len(need_compute), len(need_compute),
        )
        with cache.transaction():
            for p in tqdm(need_compute, desc="SHA-256", unit="file"):
                try:
                    sha = file_sha256(p)
                except OSError as e:
                    logger.warning("Can't hash %s: %s", p, e)
                    continue
                cache.set_sha256(p, sha)
                by_sha.setdefault(sha, []).append(p)

    return {sha: ps for sha, ps in by_sha.items() if len(ps) > 1}


# ---------- Near duplicates (pHash + CLIP) ----------

def find_near_duplicates_phash(
    files: list[Path],
    cache: Cache,
    device: str,
    phash_hamming: int,
    similarity: float,
    clip_batch: int,
    clip_max_size: int,
) -> list[list[Path]]:
    """Classic pipeline: pHash bucketing -> CLIP cosine verification.

    Returns: list of near-dup groups (each group has >=2 paths).
    """
    # 1) pHash
    phashes: dict[Path, int] = {}
    need_compute: list[Path] = []
    for p in files:
        h = cache.get_phash(p)
        if h is not None:
            phashes[p] = h
        else:
            need_compute.append(p)

    if need_compute:
        logger.info(
            "pHash: %d cached, %d to compute",
            len(files) - len(need_compute), len(need_compute),
        )
        with cache.transaction():
            for p in tqdm(need_compute, desc="pHash", unit="file"):
                h = compute_phash(p)
                if h is not None:
                    cache.set_phash(p, h)
                    phashes[p] = h

    # 2) Bucket by top 32 bits to keep pair scan tractable
    buckets: dict[int, list[tuple[Path, int]]] = {}
    for p, h in phashes.items():
        buckets.setdefault(h >> 224, []).append((p, h))

    candidate_pairs: list[tuple[Path, Path]] = []
    for items in tqdm(buckets.values(), desc="pHash bucket scan", unit="bucket"):
        n = len(items)
        for i in range(n):
            pi, hi = items[i]
            for j in range(i + 1, n):
                pj, hj = items[j]
                if hamming256(hi, hj) <= phash_hamming:
                    candidate_pairs.append((pi, pj))

    if not candidate_pairs:
        return []

    logger.info("pHash candidate pairs: %d", len(candidate_pairs))

    # 3) Union-find to cluster candidates
    uf_nodes = list({p for pair in candidate_pairs for p in pair})
    uf = UnionFind(uf_nodes)
    for a, b in candidate_pairs:
        uf.union(a, b)

    candidate_sets = [s for s in uf.components() if len(s) > 1]
    logger.info("pHash candidate clusters: %d", len(candidate_sets))

    # 4) CLIP verification per cluster
    near_dup_groups: list[list[Path]] = []
    if not candidate_sets:
        return near_dup_groups

    for cluster in tqdm(candidate_sets, desc="CLIP verify", unit="cluster"):
        cluster = [p for p in cluster if p.exists()]
        if len(cluster) < 2:
            continue
        embs = embed_images(
            cluster, cache, device, batch=clip_batch, max_size=clip_max_size
        )
        sims = embs @ embs.T
        n = sims.shape[0]
        np.fill_diagonal(sims, 0)
        mask = sims >= similarity

        comp = UnionFind(list(range(n)))
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    comp.union(i, j)
        for sub in comp.components():
            if len(sub) > 1:
                near_dup_groups.append([cluster[k] for k in sub])
    return near_dup_groups


def find_near_duplicates_faiss(
    files: list[Path],
    cache: Cache,
    device: str,
    similarity: float,
    clip_batch: int,
    clip_max_size: int,
    faiss_k: int = 50,
) -> list[list[Path]]:
    """FAISS-based near-dup detection over CLIP embeddings.

    Scales linearly via HNSW. Recommended for >50k images.
    """
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError(
            "faiss not installed. Install with: pip install faiss-cpu"
        ) from e

    # Embed everything (cached lookups make this cheap on re-runs)
    embs = embed_images(files, cache, device, batch=clip_batch, max_size=clip_max_size)
    n, d = embs.shape
    if n < 2:
        return []

    logger.info("Building FAISS HNSW index over %d embeddings (dim=%d)...", n, d)
    # Inner product on unit vectors == cosine similarity
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = max(64, faiss_k * 2)
    index.add(embs)

    # k+1 because each query returns itself with similarity ~1.0
    k = min(faiss_k + 1, n)
    logger.info("FAISS search: top-%d per image", k - 1)
    sims, idxs = index.search(embs, k)

    # Collect pairs above threshold (skip self)
    uf = UnionFind(list(range(n)))
    pair_count = 0
    for i in tqdm(range(n), desc="FAISS clustering", unit="img"):
        for rank in range(k):
            j = int(idxs[i, rank])
            if j == i or j < 0:
                continue
            if sims[i, rank] < similarity:
                break  # FAISS returns descending sims for HNSW MIPS
            if j > i:
                uf.union(i, j)
                pair_count += 1
    logger.info("FAISS pairs above %.2f: %d", similarity, pair_count)

    groups = [s for s in uf.components() if len(s) > 1]
    return [[files[k] for k in g] for g in groups]


# ---------- Representative picking ----------

def pick_representative(files: list[Path], policy: str = "sharpest") -> Path:
    """Choose the 'best' file from a duplicate group.

    Policies:
      - 'sharpest': highest variance-of-Laplacian (image sharpness)
      - 'largest':  largest file size on disk
      - 'newest':   most recent mtime
      - 'oldest':   oldest mtime
      - 'first':    lexicographically first (deterministic, no I/O)
    """
    files = [f for f in files if f.exists()]
    if not files:
        raise ValueError("No existing files in group")
    if len(files) == 1:
        return files[0]

    if policy == "largest":
        return max(files, key=lambda p: p.stat().st_size)
    if policy == "newest":
        return max(files, key=lambda p: p.stat().st_mtime)
    if policy == "oldest":
        return min(files, key=lambda p: p.stat().st_mtime)
    if policy == "first":
        return min(files, key=str)
    if policy == "sharpest":
        best = None
        best_score = -1.0
        for f in files:
            im = safe_open_pil(f)
            if im is None:
                continue
            try:
                s = image_sharpness(im)
            except Exception:
                s = 0.0
            if s > best_score:
                best_score = s
                best = f
        return best or files[0]

    raise ValueError(f"Unknown representative policy: {policy}")
