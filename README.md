# LSICT

A cross-platform (macOS / Windows / Linux) app for turning very large image folders into clean image sets. Use it from the **browser GUI** or the command line. It can:

- filter out images containing **people or faces**,
- remove **exact and near-duplicates**,
- export the survivors as center-cropped **300×300 JPEGs** named `1.jpg`, `2.jpg`, …,
- screen for **NSFW** content and keep only safe images,
- seed a diverse starter set from **Open Images v7**.

Rejected files aren't deleted — they're mirrored to an `Unkept` folder, and every rejection is logged with its reason. A SQLite cache makes re-runs fast: only new or changed files are re-processed.

> **⚠️ No guarantees — always check the results by hand.**
> Filtering relies on machine-learning models (person/face detection, NSFW classification, similarity matching), and none of them are perfect: they will occasionally miss people, faces, duplicates, or unsafe content. If your image set is destined for sensitive work, treat LSICT as a first pass and **manually review the final set before using it** — the GUI's Review tab exists for exactly that.

---

## Install

One command installs everything — Python (if you don't have it), all dependencies, and the `lsict` app. You don't need Python, git, or anything else pre-installed.

**Step 1 — open a terminal and paste the line for your OS:**

macOS / Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/ahrebel/LSICT/main/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/ahrebel/LSICT/main/install.ps1 | iex"
```

The dependencies are ~2 GB (mostly PyTorch), so the first install takes a few minutes.

**Step 2 — open a NEW terminal window** (so the just-installed command is found) **and run:**

```bash
lsict gui
```

The app opens in your browser. That's it.

- To **upgrade** later, just re-run the Step 1 command.
- Model weights (YOLO, CLIP, …) download automatically the first time you run a job, not at install.

<details>
<summary><strong>Prefer to run the steps yourself? Manual install with uv or pip</strong></summary>

The install script above just automates these steps — here they are by hand.

**Step 1 — install [uv](https://docs.astral.sh/uv/)** (skip if you already have it):

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

**Step 2 — open a new terminal, then install LSICT with uv** (uv downloads a suitable Python automatically if the machine doesn't have one):

```bash
uv tool install --python 3.12 "lsict[gui] @ https://github.com/ahrebel/LSICT/archive/refs/heads/main.tar.gz"
```

**Step 3 — run it:**

```bash
lsict gui
```

Alternatively, if you already have Python 3.10–3.12 and prefer plain pip (no uv involved):

```bash
pip install "lsict[gui] @ https://github.com/ahrebel/LSICT/archive/refs/heads/main.tar.gz"
lsict gui
```

Optional extras — add them inside the brackets, comma-separated (e.g. `lsict[gui,faiss]`):

| Extra             | What it adds                          |
| ----------------- | ------------------------------------- |
| `gui`             | the browser app (`lsict gui`)         |
| `mediapipe`       | better face detection                 |
| `faiss`           | fast near-dup search for big sets     |
| `nsfw-classifier` | pretrained NSFW model (recommended)   |
| `seed`            | Open Images seeding                   |
| `all`             | everything above                      |

For development, work from a clone:

```bash
git clone https://github.com/ahrebel/LSICT && cd LSICT
pip install -e ".[gui]"
```

</details>

### Try it without installing anything (no trace)

Want to run LSICT once and leave **nothing** on the machine? This variant downloads everything — including Python and the model weights — into a single temporary folder, runs the app, and **deletes the whole folder when you stop the app or close the terminal**:

macOS / Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/ahrebel/LSICT/main/run-once.sh | sh
```

Windows (PowerShell) — stop with **Ctrl+C** so the cleanup runs:

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/ahrebel/LSICT/main/run-once.ps1 | iex"
```

Trade-off: because nothing is kept, **every launch re-downloads the ~2 GB of dependencies**. Use the regular install above if you'll run LSICT more than once. (Your image folders and exports are of course never deleted — only the temporary program folder is.)

---

## The app (GUI)

```bash
lsict gui
```

This starts the app and **automatically opens it in your default web browser** at `http://127.0.0.1:7860` (local only — your images never leave the machine; use `--no-browser` if you don't want the auto-open). Three tabs:

- **Curate** — point it at your input folder(s), choose Kept/Unkept folders, hit *Run pipeline*. A **live progress bar** shows the current stage and how far along it is (e.g. *YOLO person — 422/1,200*), with the full log streaming below it and a summary when it's done. Advanced settings (similarity, detectors, export size, …) are in a collapsible panel.
- **NSFW screen** — copy/move only SAFE-classified images from one folder to another.
- **Review** — side-by-side galleries: what was kept, and what was rejected **with the reason** (person / face / exact duplicate / near duplicate). This is the manual check the disclaimer above is about.

`lsict gui --port 8000` to change the port, `--no-browser` to not auto-open a tab.

---

## Command line

Everything the GUI does (and more) is also a CLI:

```bash
lsict run \
    --input "/path/to/photos" \
    --kept "/path/to/Kept" \
    --unkept "/path/to/Unkept" \
    --copy-instead
```

This detects people/faces, removes duplicates, and writes numbered 300×300 JPEGs to `Kept`, plus a `manifest.csv` mapping each output back to its original file and an `outcomes.csv` in `Unkept` recording why each file was rejected. `--copy-instead` copies rejects to `Unkept` instead of moving them (recommended while testing).

For sets over ~50k images, add `--use-faiss` (needs the `faiss` extra) to make duplicate search much faster.

```bash
lsict nsfw --src "/path/to/Kept" --dst "/path/to/FinalImageSet" --copy
lsict seed --output "/path/to/seed" --num-categories 1000 --images-per-category 2
```

### Subcommands

| Command  | What it does                                        |
| -------- | --------------------------------------------------- |
| `gui`    | Launch the browser app                              |
| `run`    | Full pipeline: detect → dedup → export              |
| `detect` | Just people/face filtering                          |
| `dedup`  | Just duplicate removal                              |
| `export` | Just the numbered 300×300 export                    |
| `nsfw`   | Keep only SAFE images                               |
| `seed`   | Seed a diverse set from Open Images v7              |
| `cache`  | Inspect / prune / clear the SQLite cache            |

Every subcommand supports `--help` for its full options.

### Useful options

- `--similarity` (default 0.90) — how alike two images must be to count as near-duplicates. Higher = stricter.
- `--rep-policy` (default `sharpest`) — which image to keep from a duplicate group: `sharpest`, `largest`, `newest`, `oldest`, or `first`.
- `--yolo-conf` (default 0.25) — person-detection confidence. Lower flags more aggressively.
- `--face-backend` (default `auto`) — face detector: `mediapipe`, `yunet`, or `haar`. `auto` uses the best one available (Haar requires OpenCV 4; it was removed in OpenCV 5).
- `--device` — `auto` (default), `cuda`, `mps`, or `cpu`.

---

## Cache

A SQLite cache (default: `<kept>/.lsict_cache.sqlite`; caches from older versions are picked up automatically) stores hashes, embeddings, and detection results so re-runs only process changed files. If you stop a run partway, just run it again — it picks up where it left off.

```bash
lsict cache --db /path/to/.lsict_cache.sqlite info    # or: prune, clear
```

Disable with `--no-cache`.

---

## Safety notes

- Without `--copy-instead` (CLI) or with the copy checkbox off (GUI), rejected files are **moved** to `Unkept`. Test on a small subset first.
- Originals are never modified; exports and the cache are separate files.
- The GUI binds to `127.0.0.1` only — nothing is exposed to the network.
- And again: **model-based filtering is imperfect** — manually check the final set before using it anywhere sensitive.

## License

MIT — see [LICENSE](LICENSE).

## AI use

Some parts of this code and these docs were drafted with help from AI tools.
