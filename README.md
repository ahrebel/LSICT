# Large-Scale Image Curation Toolkit

A practical toolkit for curating *very large* image folders. It:

- removes **people/faces** (to an **Unkept** folder),
- finds **duplicates & near-duplicates** (also to **Unkept**),
- exports a clean **Kept** set as **center-cropped 300×300 JPEGs** named `1.jpg`, `2.jpg`, …
- (optional) screens for **NSFW** and moves only “safe” images to **FinalImageSet**.

> Designed to be **incremental-safe** and to handle **hundreds of thousands** of images.

---

## Contents

- `dedupe_and_people.py` — main pipeline: people/face filtering → dedup (exact + near) → export 300×300 Kept
- `resume_dedupe_export.py` — **resume** dedupe + export using an existing Unkept mirror (skip people/face step)
- `nsfw_filter_move.py` — stand-alone NSFW screen that moves **only SAFE** images to `FinalImageSet`
- `imageget.py` — utility to seed a diverse dataset from Open Images v7 via FiftyOne and save as 300×300

> **Note:** The **resume script is not always needed** — only use `resume_dedupe_export.py` if you already completed (or mostly completed) the people/face step and have a populated **Unkept** that you want to reuse.

---

## Installation

Tested with Python **3.10–3.12** on macOS (Apple Silicon). Use a virtualenv if you like.

**Core dependencies (main pipeline):**
```bash
python3 -m pip install pillow imagehash numpy tqdm torch open-clip-torch ultralytics opencv-python
````

**Optional: NSFW and seeding**

```bash
# If your nsfw_filter_move.py uses NudeNet:
python3 -m pip install "nudenet==2.0.6" pillow tqdm

# If your nsfw_filter_move.py uses CLIP (zero-shot):
python3 -m pip install open-clip-torch pillow tqdm

# For image seeding via Open Images v7:
python3 -m pip install fiftyone pandas
```

**Models downloaded on first use**

* YOLOv8n (`ultralytics`) for person detection (class `person`)
* OpenCV Haar cascade for faces
* CLIP ViT-B/32 (`open-clip-torch`) for near-dup verification (and zero-shot NSFW, if used)

---

## Quick Start

### A) Full clean run (two inputs)

```bash
python3 dedupe_and_people.py \
  --input "/path/to/data" \
  --input "/path/to/data2" \
  --kept "/path/to/Kept" \
  --unkept "/path/to/Unkept" \
  --similarity 0.90 \
  --phash-hamming 30 \
  --batch 256 \
  --max-size 512 \
  --copy-instead
```

**What happens**

1. **People/face filter** → matches are **mirrored** into `Unkept` (input tree preserved).
2. **Deduplication**

   * **Exact dups:** SHA-256 groups; keep a representative (policy configurable), send others to `Unkept`
   * **Near dups:** 256-bit pHash → CLIP similarity verify → send near-dups to `Unkept`
3. **Export Kept** as numbered **300×300** JPEGs (`1.jpg`, `2.jpg`, …)

**Safety:** Pass `--copy-instead` to avoid moving any originals.
**Log:** `KEPT/run_log.csv` (`original_path,outcome,kept_index,unkept_relpath`).

---

### B) Resume dedupe + export only (skip people/face)

Use this **only** if your `Unkept` already contains the people/face filtered images (e.g., you stopped after Stage 1). Otherwise, it’s **not needed** — just rerun the main pipeline.

```bash
python3 -u resume_dedupe_export.py \
  --input "/path/to/data" \
  --input "/path/to/data2" \
  --unkept "/path/to/Unkept" \
  --kept "/path/to/Kept" \
  --similarity 0.90 \
  --phash-hamming 30 \
  --batch 256 \
  --max-size 512
```

This skips detection, treats `Unkept` as a “mask” of already-rejected files, then runs dedupe + export.

---

### C) Optional: Final NSFW screen (move **only safe** images)

**Check which variant your `nsfw_filter_move.py` uses:**

* If it imports `NudeClassifier`, follow **NudeNet** instructions:

  ```bash
  python3 -m pip install "nudenet==2.0.6" pillow tqdm

  python3 -u nsfw_filter_move.py \
    --src "/path/to/Kept" \
    --dst "/path/to/FinalImageSet" \
    --unsafe-threshold 0.50 \
    --block-sexy        # optional: count 'sexy' as unsafe
  ```

* If it imports `open_clip`, follow **CLIP zero-shot** instructions:

  ```bash
  python3 -m pip install open-clip-torch pillow tqdm

  python3 -u nsfw_filter_move.py \
    --src "/path/to/Kept" \
    --dst "/path/to/FinalImageSet" \
    --nsfw-thresh 0.55   # lower = stricter, higher = more lenient
  ```

The script **moves/copies only SAFE images** into `FinalImageSet`, avoiding filename collisions by adding ` (1)`, ` (2)`, etc.

---

## Script Details & Options

### `dedupe_and_people.py`

* `--input PATH` (repeatable): one or more input roots
* `--kept PATH`: output folder for numbered 300×300 JPEGs
* `--unkept PATH`: mirrored folder tree for people/dupes
* `--copy-instead`: copy to Unkept instead of move (safer for originals)
* `--overwrite-unkept`: overwrite collisions in Unkept (default creates ` (1)`, ` (2)` suffixes)
* `--no-clean-kept`: don’t wipe Kept before export (numbering may become non-contiguous)
* `--similarity 0.90`: CLIP cosine threshold for near-dup grouping
* `--phash-hamming 30`: pHash prefilter tightness (0–256)
* `--batch 256`: CLIP batch size
* `--max-size 512`: long edge resize before CLIP
* `--kept-side 300`: output crop size (default 300)

### `resume_dedupe_export.py`

* Same dedup/export options as above
* **Skips detection** and uses existing `Unkept` to mask already-rejected files

### `nsfw_filter_move.py`

* NudeNet variant: `--unsafe-threshold` (0..1), `--block-sexy`, `--batch`, `--copy`
* CLIP variant: `--nsfw-thresh` (probability cutoff), `--batch`, `--copy`

### `imageget.py`

* Seeds a diverse set from Open Images v7 via FiftyOne
* Saves **300×300** crops to your target folder, with integer filenames
  (prints progress every 100 images)

---

## Performance & Tips

* **Apple Silicon:** CLIP will use **MPS** automatically; YOLO runs on CPU by default with `ultralytics`.
* **Very large datasets:** Stage 1 (people/face) is the longest. Stage 2 may appear “quiet” during hashing and grouping — watch CPU usage.
* **Incremental runs:** Add more images to any input folder and rerun — Unkept mirroring is **incremental-safe** (identical files skipped; differing content gets suffixed unless `--overwrite-unkept`). Kept numbering resets unless `--no-clean-kept`.
* **Disk space:** Ensure enough room for Unkept mirrors and Kept exports.
* **Live logs:** Use `python3 -u` for unbuffered progress output.

---

## FAQ

**Does it check duplicates across multiple folders?**
Yes — pass multiple `--input` roots; dedup considers them together.

**Does it delete my originals?**
Only if you *don’t* use `--copy-instead`. For safety, run with `--copy-instead` so Unkept receives **copies**.

**Where is the decision log?**
`KEPT/run_log.csv` lists every decision: `person`, `face`, `duplicate_exact`, `duplicate_near`, or `kept`.

**Do I always need the resume script?**
**No.** Use `resume_dedupe_export.py` only when you already have a populated `Unkept` from a prior run and want to **skip** people/face detection.

---

## Support & Maintenance

This project is **not actively maintained**. It’s provided **as is** as a starting point for experiments, and we **do not guarantee** it will work on your setup.

- No SLA or official support
- Issues and PRs are welcome; responses are best-effort
- **Use at your own risk:** back up your data/images and test on a small subset first
  - We strongly recommend duplicating your image folders before running any filters, in case something fails or behaves unexpectedly
- Dependencies may change — pin versions if you need reproducibility (e.g., keep a `requirements.txt`)

---

## AI Use

Some parts of this project (code and docs) were drafted with help from AI tools. We (humans) set the goals, reviewed outputs, tested, and made changes.

What AI helped with:
- Speeding up development and troubleshooting command-line inputs
- Suggesting improvements and design changes to improve speed for large (100k+) imagesets
- Drafting parts of README

Human review:
- Each code file was used and tested by a human to ensure it works properly and as expected
- Tested with 250k images
- If you spot an issue, please open an issue or a Pull Request and we’ll attempt to fix it

---

## License

MIT License (See files)
```
