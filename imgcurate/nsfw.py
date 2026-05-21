"""NSFW screening with two backends:

  - 'classifier': pretrained image classifier (Falconsai/nsfw_image_detection
                  on HuggingFace). Much more accurate than zero-shot.
                  Requires `transformers`.
  - 'clip':       CLIP zero-shot with NSFW vs SFW prompts. No extra deps.

Only SAFE images are moved/copied to the destination. Scores are cached so
re-runs are cheap.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from imgcurate.cache import Cache
from imgcurate.core import (
    ensure_dir,
    next_nonconflicting_path,
    safe_open_pil,
)

logger = logging.getLogger(__name__)


NSFW_PROMPTS = [
    "pornography", "explicit sexual content", "nudity",
    "bare breasts", "female nipples", "male genitalia",
    "sex act", "erotic nude photo",
]
SFW_PROMPTS = [
    "a normal, safe photo", "a person fully clothed",
    "a landscape photo", "a city street photo", "an office scene",
]


# ---------- Backend: CLIP zero-shot ----------

class ClipNsfwBackend:
    """CLIP zero-shot NSFW vs SFW. Returns probability of NSFW per image."""

    def __init__(self, device: str):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            t_nsfw = tokenizer(NSFW_PROMPTS).to(device)
            t_sfw = tokenizer(SFW_PROMPTS).to(device)
            f_nsfw = self.model.encode_text(t_nsfw)
            f_sfw = self.model.encode_text(t_sfw)
            self.f_nsfw = f_nsfw / f_nsfw.norm(dim=-1, keepdim=True)
            self.f_sfw = f_sfw / f_sfw.norm(dim=-1, keepdim=True)

    def score_batch(self, paths: list[Path], max_side: int = 512) -> dict[Path, float]:
        tensors = []
        valid_paths = []
        for p in paths:
            im = safe_open_pil(p)
            if im is None:
                continue
            im.thumbnail((max_side, max_side), Image.BICUBIC)
            tensors.append(self.preprocess(im).unsqueeze(0))
            valid_paths.append(p)

        out: dict[Path, float] = {}
        if not tensors:
            return out

        with torch.no_grad():
            x = torch.cat(tensors, dim=0).to(self.device)
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sims_nsfw, _ = (feats @ self.f_nsfw.T).max(dim=1)
            sims_sfw, _ = (feats @ self.f_sfw.T).max(dim=1)
            logits = torch.stack([sims_sfw, sims_nsfw], dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
        probs_np = probs.detach().cpu().numpy()
        for p, pr in zip(valid_paths, probs_np):
            out[p] = float(pr)
        return out


# ---------- Backend: pretrained classifier ----------

class ClassifierNsfwBackend:
    """HuggingFace pipeline using Falconsai/nsfw_image_detection.

    Returns probability of the 'nsfw' class per image.
    """

    def __init__(self, device: str, model_id: str = "Falconsai/nsfw_image_detection"):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError(
                "transformers not installed. Install with: pip install transformers accelerate"
            ) from e
        # pipeline device: 'cuda', 'mps', or -1 for cpu
        pipe_device = device if device in ("cuda", "mps") else -1
        self.pipe = pipeline(
            "image-classification",
            model=model_id,
            device=pipe_device,
        )

    def score_batch(self, paths: list[Path], max_side: int = 512) -> dict[Path, float]:
        images = []
        valid_paths = []
        for p in paths:
            im = safe_open_pil(p)
            if im is None:
                continue
            im.thumbnail((max_side, max_side), Image.BICUBIC)
            images.append(im)
            valid_paths.append(p)

        out: dict[Path, float] = {}
        if not images:
            return out

        try:
            results = self.pipe(images)
        except Exception as e:
            logger.warning("Classifier batch failed: %s", e)
            return out

        # Each result is a list of {label, score}. We want score where label='nsfw'.
        for p, res in zip(valid_paths, results):
            nsfw_score = 0.0
            for entry in res:
                if str(entry["label"]).lower() in ("nsfw", "porn", "explicit"):
                    nsfw_score = max(nsfw_score, float(entry["score"]))
            out[p] = nsfw_score
        return out


def make_backend(backend: str, device: str):
    if backend == "classifier":
        return ClassifierNsfwBackend(device)
    if backend == "clip":
        return ClipNsfwBackend(device)
    if backend == "auto":
        try:
            return ClassifierNsfwBackend(device)
        except RuntimeError as e:
            logger.info("Classifier backend unavailable (%s); falling back to CLIP zero-shot.", e)
            return ClipNsfwBackend(device)
    raise ValueError(f"Unknown NSFW backend: {backend}")


# ---------- Main entry ----------

def screen_safe(
    src_dir: Path,
    dst_dir: Path,
    cache: Cache,
    device: str,
    threshold: float = 0.5,
    backend: str = "auto",
    batch: int = 32,
    copy: bool = False,
) -> dict:
    """Move (or copy) only SAFE images from src_dir to dst_dir.

    threshold: NSFW probability >= threshold => UNSAFE (excluded).
    Lower = stricter.
    """
    from imgcurate.core import is_image_file

    files = [p for p in src_dir.rglob("*") if p.is_file() and is_image_file(p)]
    if not files:
        return {"total": 0, "moved": 0, "blocked": 0, "skipped": 0}

    ensure_dir(dst_dir)

    # 1) Compute scores for any not cached
    need_score: list[Path] = []
    scores: dict[Path, float] = {}
    for p in files:
        s = cache.get_nsfw_score(p)
        if s is None:
            need_score.append(p)
        else:
            scores[p] = s

    if need_score:
        logger.info(
            "NSFW (%s): %d cached, %d to score",
            backend, len(files) - len(need_score), len(need_score),
        )
        be = make_backend(backend, device)
        for i in tqdm(range(0, len(need_score), batch), desc=f"NSFW score ({backend})"):
            chunk = need_score[i:i + batch]
            scored = be.score_batch(chunk)
            with cache.transaction():
                for p in chunk:
                    s = scored.get(p, float("nan"))
                    if not np.isnan(s):
                        cache.set_nsfw_score(p, s)
                        scores[p] = s

    # 2) Move/copy SAFE images
    moved = blocked = skipped = 0
    for p in tqdm(files, desc="Move SAFE -> dst", unit="img"):
        s = scores.get(p)
        if s is None:
            skipped += 1
            continue
        if s >= threshold:
            blocked += 1
            continue
        dst = next_nonconflicting_path(dst_dir / p.name)
        try:
            if copy:
                shutil.copy2(str(p), str(dst))
            else:
                shutil.move(str(p), str(dst))
            moved += 1
        except OSError as e:
            logger.warning("Failed to write %s: %s", dst, e)
            skipped += 1

    return {
        "total": len(files),
        "moved": moved,
        "blocked": blocked,
        "skipped": skipped,
        "threshold": threshold,
        "backend": backend,
    }
