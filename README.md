# imgcurate

A cross-platform (macOS / Windows / Linux) toolkit for curating very large image folders. It:

- **filters out people and faces** (mirrors them to `UNKEPT`),
- finds **exact and near-duplicates** (also to `UNKEPT`),
- exports a clean **Kept** set as center-cropped **300×300 JPEGs** named `1.jpg`, `2.jpg`, …,
- (optional) screens for **NSFW** and moves only safe images to a final folder,
- (optional) seeds a diverse dataset from **Open Images v7**.

Designed for **incremental-safe** runs over **hundreds of thousands** of images — a SQLite cache means re-runs only process files that changed.

This is a rewrite of an earlier 3-script toolkit. Major changes are listed at the bottom.

---

## Installation

Python 3.10–3.12. macOS (Intel or Apple Silicon), Windows 10+, or Linux.

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[mediapipe]"          # better face detector
pip install -e ".[faiss]"              # fast near-dup search at scale
pip install -e ".[nsfw-classifier]"    # pretrained NSFW model (recommended)
pip install -e ".[seed]"               # Open Images seeding
pip install -e ".[all]"                # everything
```

On first run, model weights are auto-downloaded:

- YOLOv8n (~6 MB) for person detection
- OpenCV Haar cascade (bundled with opencv-python)
- CLIP ViT-B/32 (~600 MB) for near-dup verification & zero-shot NSFW
- MediaPipe face landmarker (small) if installed
- Falconsai/nsfw_image_detection (~120 MB) on first `nsfw` use if installed

---

## Quick start

### Full pipeline (one command)

```bash
imgcurate run \
    --input "D:/Photos/raw1" --input "D:/Photos/raw2" \
    --kept "D:/Photos/Kept" \
    --unkept "D:/Photos/Unkept" \
    --copy-instead \
    --similarity 0.90 \
    --phash-hamming 30
```

On macOS the paths just look different:

```bash
imgcurate run \
    --input "/Volumes/Drive/Photos/raw1" --input "/Volumes/Drive/Photos/raw2" \
    --kept "/Volumes/Drive/Photos/Kept" \
    --unkept "/Volumes/Drive/Photos/Unkept" \
    --copy-instead
```

What it does:

1. **Detect** — YOLO finds images containing people; remaining are face-checked (MediaPipe → Haar fallback). All flagged files are mirrored to `Unkept`.
2. **Dedup** — exact dupes by SHA-256 + near-dupes by pHash + CLIP cosine. Reps stay; others go to `Unkept`.
3. **Export** — survivors center-cropped to 300×300 JPEGs, numbered `1.jpg`, `2.jpg`, …

A `manifest.csv` is written next to the exports mapping each output number back to its original path.

### At scale (>50k images), use FAISS

```bash
imgcurate run \
    --input "C:/data/photos" \
    --kept "C:/data/Kept" --unkept "C:/data/Unkept" \
    --use-faiss --faiss-k 50 \
    --copy-instead
```

FAISS HNSW over CLIP embeddings finds near-dups in O(n log n) instead of the bucketed pHash approach's O(b·k²). On 250k images this is the difference between hours and minutes for the near-dup stage.

### NSFW screen (run after `imgcurate run`)

```bash
imgcurate nsfw \
    --src "D:/Photos/Kept" \
    --dst "D:/Photos/FinalImageSet" \
    --backend classifier \
    --threshold 0.5 \
    --copy
```

Backends: `classifier` (Falconsai, accurate, needs `transformers`), `clip` (zero-shot, no extras), `auto` (prefers classifier, falls back to CLIP). Only SAFE images are moved/copied; UNSAFE stay put.

### Seed from Open Images v7

```bash
imgcurate seed \
    --output "D:/Photos/seed" \
    --num-categories 1000 \
    --images-per-category 2 \
    --side 300 \
    --seed 42
```

---

## Subcommands

| Command   | What it does                                                |
| --------- | ----------------------------------------------------------- |
| `run`     | Full pipeline: detect → dedup → export                      |
| `detect`  | Just people/face filtering → UNKEPT                         |
| `dedup`   | Just exact + near deduplication → UNKEPT                    |
| `export`  | Center-crop survivors → numbered 300×300 JPEGs              |
| `nsfw`    | NSFW screen: move only SAFE images to a destination         |
| `seed`    | Seed a diverse set from Open Images v7                      |
| `cache`   | Inspect / prune / clear the SQLite cache                    |

Each subcommand supports `--help`. Every subcommand that needs models accepts `--device {auto,cuda,mps,cpu}`. `auto` picks CUDA → MPS → CPU.

### Resuming

The old `resume_dedupe_export.py` is gone — it's no longer needed. The SQLite cache makes a second `imgcurate run` cheap (it skips anything already computed). If you stopped after detection and want to resume from dedup:

```bash
imgcurate run --input ... --kept ... --unkept ... --skip-detect
```

---

## Cache

By default a SQLite cache lives at `<kept>/.imgcurate_cache.sqlite`. It stores:

- SHA-256 of each file (for exact-dup detection)
- 256-bit pHash
- CLIP embedding (~2 KB / image)
- Person/face detection results
- NSFW score

Cache validity is keyed on `(absolute_path, size, mtime)`. If you edit or replace a file, that row is invalidated and recomputed on the next pass. Files moved or renamed are simply re-cached at the new path.

```bash
imgcurate cache --db /path/to/.imgcurate_cache.sqlite info
imgcurate cache --db /path/to/.imgcurate_cache.sqlite prune    # drop rows for missing files
imgcurate cache --db /path/to/.imgcurate_cache.sqlite clear    # wipe everything
```

For 250k images the cache db is typically 600 MB–1 GB (CLIP embeddings dominate). Disable with `--no-cache` if you want recompute-everything-every-time behavior.

---

## Cross-platform notes

| Concern                          | Status                                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------- |
| Windows long paths (>260 chars)  | Auto-prefixed with `\\?\` in `normalize_path`                                                   |
| Non-ASCII paths on Windows       | All image reads go through `cv2.imdecode(np.fromfile(...))` to avoid the `cv2.imread` bug       |
| Multiprocessing on Windows       | `seed` uses a top-level worker function and is only spawned from inside `__main__` (CLI entry)  |
| Mac MPS / CUDA / CPU             | Auto-detected; explicit selection via `--device`                                                |
| Path case-insensitivity (Win)    | Cache keys lowercased on Windows                                                                |
| Atomic writes                    | All exports go through `os.replace(tmp, final)` — no half-written files on Ctrl+C               |
| EXIF orientation                 | Applied via `ImageOps.exif_transpose` before any further processing                             |
| HEIC / HEIF                      | Supported when `pillow-heif` is installed (it's a hard dep)                                     |

---

## Tuning

**Similarity threshold (`--similarity`, default 0.90)** — CLIP cosine cutoff for declaring near-duplicates. Higher = stricter. 0.95 catches only very tight reproductions; 0.85 will pull in same-scene-different-shot pairs.

**pHash threshold (`--phash-hamming`, default 30)** — Only matters when not using `--use-faiss`. Hamming distance over 256 bits for candidate pairing. Higher = more candidates (slower CLIP verify but better recall). 20–40 is reasonable.

**Representative policy (`--rep-policy`, default `sharpest`)** — Which file to keep from a duplicate group:

- `sharpest` — highest variance-of-Laplacian (best focus). Requires opening each file in the group.
- `largest` — largest on-disk size. Cheap.
- `newest` / `oldest` — by mtime.
- `first` — lexicographic. Fully deterministic, no I/O.

**Face backend (`--face-backend`, default `auto`)** — `mediapipe` is much better at angled/partial faces than the OpenCV Haar cascade; if `mediapipe` is installed, `auto` picks it. Use `--face-backend off` if you only want YOLO person detection.

**YOLO confidence (`--yolo-conf`, default 0.25)** — Lower = more aggressive person flagging. 0.15 is paranoid; 0.4 is permissive.

---

## What changed vs. the previous toolkit

| Old                                            | New                                                          |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Three overlapping scripts                      | One package, one CLI, six subcommands                        |
| No cache — every run recomputes everything     | SQLite cache keyed on `(path, size, mtime)`                  |
| pHash bucketing only                           | Add FAISS HNSW over CLIP embeddings (`--use-faiss`)          |
| OpenCV Haar face detector only                 | MediaPipe option (better recall on profiles/angles)          |
| CLIP zero-shot NSFW only                       | Add Falconsai classifier option (much more accurate)         |
| No HEIC / HEIF support                         | Supported via pillow-heif                                    |
| No EXIF orientation handling                   | Auto-applied                                                 |
| Non-atomic JPEG writes                         | Temp file + `os.replace`                                     |
| `largest` rep selection only                   | `sharpest` (default) / `largest` / `newest` / `oldest` / `first` |
| Hardcoded `/Volumes/...` in seed script        | Cross-platform, takes `--output`                             |
| `multiprocessing.Pool` without `__main__` guard (broke on Windows) | Worker is module-level; spawned only via CLI entry            |
| `cv2.imread` (fails on Windows non-ASCII paths) | Already used `cv2.imdecode(np.fromfile(...))` — kept         |
| Implicit device choice                         | Explicit `--device {auto,cuda,mps,cpu}`                      |
| Separate "resume" script                       | Just rerun — cache handles incremental                       |

---

## Safety

- Default is **copy** to `UNKEPT` via `--copy-instead`. Without that flag, files are *moved*. Test on a small subset first.
- Cache and outputs are independent of input files. Wiping outputs never touches your originals.
- `imgcurate cache prune` is safe — it only removes rows where the file no longer exists.

## License

MIT — see [LICENSE](LICENSE).

## AI use

Some parts of this code and these docs were drafted with help from AI tools. A human set the goals, reviewed output, and tested at 250k-image scale.
