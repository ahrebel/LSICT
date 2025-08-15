#!/usr/bin/env python3
import os, sys, hashlib, csv, json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image, UnidentifiedImageError
import imagehash, torch, open_clip

IMG_EXTS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff",".gif"}

def is_img(p:Path): return p.suffix.lower() in IMG_EXTS
def ensure_dir(d:Path): d.mkdir(parents=True, exist_ok=True)

def safe_open(p:Path)->Optional[Image.Image]:
    try:
        with Image.open(p) as im:
            im.load()
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def phash256(p:Path, hash_size=16)->Optional[int]:
    im=safe_open(p)
    if im is None: return None
    try:
        return int(str(imagehash.phash(im, hash_size=hash_size)),16)
    except Exception:
        return None

def hamming256(a:int,b:int)->int: return (a^b).bit_count()

class UF:
    def __init__(self, items): self.p={x:x for x in items}; self.r={x:0 for x in items}
    def f(self,x):
        if self.p[x]!=x: self.p[x]=self.f(self.p[x])
        return self.p[x]
    def u(self,a,b):
        ra,rb=self.f(a),self.f(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: self.p[ra]=rb
        elif self.r[ra]>self.r[rb]: self.p[rb]=ra
        else: self.p[rb]=ra; self.r[ra]+=1
    def comps(self):
        g={}
        for x in self.p:
            r=self.f(x); g.setdefault(r,set()).add(x)
        return list(g.values())

def center_crop_sq(im:Image.Image, side=300)->Image.Image:
    w,h=im.size
    s=min(w,h); im=im.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2))
    if im.size!=(side,side): im=im.resize((side,side), Image.LANCZOS)
    return im

def load_clip(device):
    model,_,pre= open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    model.eval(); return model, pre

def embed(paths:List[Path], pre, model, device, max_size=512, batch=256)->np.ndarray:
    embs=[]
    out_dim=getattr(model.visual,"output_dim",512)
    for i in tqdm(range(0,len(paths),batch), desc="CLIP embedding"):
        chunk=paths[i:i+batch]; imgs=[]; idx=[]
        for j,p in enumerate(chunk):
            im=safe_open(p)
            if im is None: imgs.append(None)
            else:
                im.thumbnail((max_size,max_size), Image.BICUBIC)
                imgs.append(im); idx.append(j)
        if not idx:
            embs.extend([np.zeros(out_dim, np.float32)]*len(chunk)); continue
        t=torch.cat([pre(imgs[j]).unsqueeze(0) for j in idx], dim=0).to(device)
        with torch.no_grad():
            f=model.encode_image(t); f=f/f.norm(dim=-1, keepdim=True)
        out=f.detach().cpu().numpy().astype(np.float32)
        k=0
        for j in range(len(chunk)):
            if imgs[j] is None: embs.append(np.zeros(out.shape[1], np.float32))
            else: embs.append(out[k]); k+=1
    return np.vstack(embs)

def count_img_files(root: Path) -> int:
    if not root.exists(): return 0
    cnt = 0
    for p in root.rglob("*"):
        if p.is_file() and is_img(p):
            cnt += 1
    return cnt

def main():
    import argparse
    ap=argparse.ArgumentParser("Resume dedupe + export using existing UNKEPT as people mask")
    ap.add_argument("--input", action="append", required=True, help="Repeat for multiple roots")
    ap.add_argument("--unkept", required=True)
    ap.add_argument("--kept", required=True)
    ap.add_argument("--similarity", type=float, default=0.90)
    ap.add_argument("--phash-hamming", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--kept-side", type=int, default=300)
    args=ap.parse_args()

    roots=[Path(p).expanduser().resolve() for p in args.input]
    unkept=Path(args.unkept).expanduser().resolve()
    kept=Path(args.kept).expanduser().resolve()
    ensure_dir(kept)

    # --- Index UNKEPT with progress ---
    unkept_rels=set()
    if unkept.exists():
        total_unkept = count_img_files(unkept)
        with tqdm(total=total_unkept, desc="Index UNKEPT", unit="file") as bar:
            for p in unkept.rglob("*"):
                if p.is_file() and is_img(p):
                    try:
                        unkept_rels.add(p.relative_to(unkept).as_posix())
                    except Exception:
                        pass
                    bar.update(1)
    print(f"UNKEPT has {len(unkept_rels)} mirrored files (will skip those)."); sys.stdout.flush()

    # --- Scan inputs & build survivor list with progress ---
    survivors=[]
    for root in roots:
        total_root = count_img_files(root)
        with tqdm(total=total_root, desc=f"Scan {root.name}", unit="file") as bar:
            for p in root.rglob("*"):
                if p.is_file() and is_img(p):
                    rel=p.relative_to(root).as_posix()
                    if rel not in unkept_rels:
                        survivors.append(p)
                    bar.update(1)
    print(f"Survivor candidates: {len(survivors)}"); sys.stdout.flush()

    # --- Stage 2a: exact duplicates among survivors ---
    by= {}
    for p in tqdm(survivors, desc="Hashing SHA-256", unit="file"):
        by.setdefault(sha256(p), []).append(p)
    dup_groups = [g for g in by.values() if len(g) > 1]
    singles    = [g[0] for g in by.values() if len(g) == 1]
    print(f"Exact-dup groups: {len(dup_groups)} | singles: {len(singles)}"); sys.stdout.flush()

    kept_set=set()
    for g in tqdm(dup_groups, desc="Choose reps (exact dup groups)", unit="group"):
        rep = max(g, key=lambda x: x.stat().st_size if x.exists() else -1)
        kept_set.add(rep)
    # add singles
    kept_set.update(singles)

    kept_list=sorted(kept_set, key=str)
    print(f"After exact dup: {len(kept_list)} to check for near-dup"); sys.stdout.flush()

    # --- pHash bucket + CLIP verify ---
    ph={}
    for p in tqdm(kept_list, desc="Computing pHash", unit="file"):
        hv=phash256(p)
        if hv is not None: ph[p]=hv

    # bucket by top 32 bits
    buckets={}
    for p,hv in ph.items():
        buckets.setdefault(hv>>224, []).append((p,hv))

    cand_pairs=[]
    for items in tqdm(list(buckets.values()), desc="Bucket pair scan", unit="bucket"):
        n=len(items)
        for i in range(n):
            pi,hi=items[i]
            for j in range(i+1,n):
                pj,hj=items[j]
                if hamming256(hi,hj) <= args.phash_hamming:
                    cand_pairs.append((pi,pj))
    print(f"Candidate pairs for CLIP: {len(cand_pairs)}"); sys.stdout.flush()

    # cluster via CLIP
    device = torch.device("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model,pre = load_clip(device)
    nodes=list({p for pair in cand_pairs for p in pair})
    uf=UF(list(range(len(nodes))))
    if nodes:
        embs=embed(nodes, pre, model, device, max_size=args.max_size, batch=args.batch)
        sims = embs @ embs.T
        n=len(nodes)
        for i in tqdm(range(n), desc="CLIP clustering", unit="row"):
            # union with j>i
            si = sims[i]
            for j in range(i+1, n):
                if si[j] >= args.similarity:
                    uf.u(i,j)

    # groups from near-dup
    near_dupe_groups=[]
    comps = uf.comps() if nodes else []
    for comp_ids in comps:
        near_dupe_groups.append([nodes[i] for i in comp_ids])
    print(f"Near-dup groups: {len(near_dupe_groups)}"); sys.stdout.flush()

    # choose reps from near-dup groups; remove others
    near_keep=set(kept_list)
    for group in tqdm(near_dupe_groups, desc="Prune near-dups", unit="group"):
        rep=max(group, key=lambda x: x.stat().st_size if x.exists() else -1)
        for f in group:
            if f!=rep and f in near_keep:
                near_keep.remove(f)
    final_list=sorted(near_keep, key=str)
    print(f"Final kept count to export: {len(final_list)}"); sys.stdout.flush()

    # --- Export KEPT as 300x300 numbered JPEGs ---
    for i,p in enumerate(tqdm(final_list, desc="Export KEPT -> 300x300", unit="img"), start=1):
        im=safe_open(p)
        if im is None: continue
        im=center_crop_sq(im, side=args.kept_side)
        im.save(kept/f"{i}.jpg", format="JPEG", quality=args.jpeg_quality, optimize=True)

    print(json.dumps({
        "inputs":[str(r) for r in roots],
        "unkept": str(unkept),
        "kept": str(kept),
        "exported": len(final_list)
    }, indent=2))

if __name__=="__main__":
    main()
