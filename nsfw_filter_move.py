#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path
from typing import List, Optional
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import numpy as np
import torch, open_clip

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif"}

NSFW_PROMPTS = [
    "pornography", "explicit sexual content", "nudity",
    "bare breasts", "female nipples", "male genitalia",
    "sex act", "erotic nude photo"
]
SFW_PROMPTS = [
    "a normal, safe photo", "a person fully clothed",
    "a landscape photo", "a city street photo", "an office scene"
]

def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def safe_open(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

def next_free(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem} ({i}){ext}")
        if not cand.exists():
            return cand
        i += 1

def pick_device() -> torch.device:
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clip(device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()
    return model, preprocess

@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
    toks = tokenizer(texts)
    feats = model.encode_text(toks.to(device))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

@torch.no_grad()
def encode_images(model, preprocess, paths: List[Path], device: torch.device, max_side=512) -> torch.Tensor:
    ims = []
    for p in paths:
        im = safe_open(p)
        if im is None:
            ims.append(None)
        else:
            im.thumbnail((max_side, max_side), Image.BICUBIC)
            ims.append(preprocess(im))
    good_idx = [i for i,x in enumerate(ims) if x is not None]
    if not good_idx:
        return torch.zeros((len(paths), model.visual.output_dim), device="cpu")

    batch = torch.stack([ims[i] for i in good_idx], dim=0).to(device)
    feats = model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    out = torch.zeros((len(paths), feats.shape[1]), device=device)
    out[torch.tensor(good_idx, device=device)] = feats
    return out

def main():
    ap = argparse.ArgumentParser("Move ONLY SAFE images to FinalImageSet via CLIP zero-shot NSFW screen")
    ap.add_argument("--src", required=True, help="Folder to scan recursively")
    ap.add_argument("--dst", default=None, help="Destination (default: <src_parent>/FinalImageSet)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--nsfw-thresh", type=float, default=0.55,
                    help="If NSFW probability >= thresh -> UNSAFE. Lower = stricter. (default 0.55)")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve() if args.dst else (src.parent / "FinalImageSet")
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob("*") if p.is_file() and is_img(p)]
    if not files:
        print("No images found.")
        return

    device = pick_device()
    model, preprocess = load_clip(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # Encode prompts once
    text_features_nsfw = encode_texts(model, tokenizer, NSFW_PROMPTS, device)
    text_features_sfw  = encode_texts(model, tokenizer, SFW_PROMPTS,  device)

    moved = 0
    print(f"Screening {len(files)} images from '{src}' -> SAFE to '{dst}' | nsfw_thresh={args.nsfw_thresh}")

    for i in tqdm(range(0, len(files), args.batch), desc="NSFW (CLIP)"):
        batch_paths = files[i:i+args.batch]

        img_feats = encode_images(model, preprocess, batch_paths, device)
        # cosine sims (because features are normalized)
        sims_nsfw = (img_feats @ text_features_nsfw.T)   # [B, N_nsfw]
        sims_sfw  = (img_feats @ text_features_sfw.T)    # [B, N_sfw]

        # For each image, compare best NSFW vs best SFW, softmax to get probability of NSFW
        best_nsfw, _ = sims_nsfw.max(dim=1)
        best_sfw,  _ = sims_sfw.max(dim=1)

        logits = torch.stack([best_sfw, best_nsfw], dim=1)  # [B, 2]
        probs  = torch.softmax(logits, dim=1)               # [B, 2] -> [SAFE, NSFW]
        nsfw_prob = probs[:, 1].detach().cpu().numpy()

        for p, prob in zip(batch_paths, nsfw_prob):
            if np.isnan(prob):
                continue
            if prob < args.nsfw_thresh:
                out = next_free(dst / p.name)
                try:
                    if args.copy:
                        shutil.copy2(str(p), str(out))
                    else:
                        shutil.move(str(p), str(out))
                    moved += 1
                except Exception as e:
                    print(f"Failed to write {out}: {e}")

    print(f"\nDone. {'Copied' if args.copy else 'Moved'} {moved} SAFE images to: {dst}")

if __name__ == "__main__":
    main()
