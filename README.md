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

Python 3.10–3.12.

### On any device (one command)

```bash
pip install "lsict[gui] @ git+https://github.com/ahrebel/LSICT@main"
```

That's it — this pulls LSICT straight from this repo, downloads every dependency, and puts the `lsict` command on your PATH. Then run `lsict gui`.

Prefer an isolated install that can't conflict with anything else on the machine? Use [pipx](https://pipx.pypa.io) or [uv](https://docs.astral.sh/uv/) instead of pip:

```bash
uv tool install "lsict[gui] @ git+https://github.com/ahrebel/LSICT@main"
```

(uv will even download a suitable Python if the machine doesn't have one.)

Heads-up: the dependencies include PyTorch (~2 GB), so the first install takes a while. Model weights (YOLOv8n, CLIP, etc.) download automatically on first *use*, not at install.

### Optional extras

Add extras inside the brackets, comma-separated — e.g. `"lsict[gui,faiss] @ git+..."`:

| Extra             | What it adds                          |
| ----------------- | ------------------------------------- |
| `gui`             | the browser app (`lsict gui`)         |
| `mediapipe`       | better face detection                 |
| `faiss`           | fast near-dup search for big sets     |
| `nsfw-classifier` | pretrained NSFW model (recommended)   |
| `seed`            | Open Images seeding                   |
| `all`             | everything above                      |

### From a local clone (development)

```bash
git clone https://github.com/ahrebel/LSICT && cd LSICT
pip install -e ".[gui]"
```

---

## The app (GUI)

```bash
lsict gui
```

Opens LSICT in your browser at `http://127.0.0.1:7860` (local only — your images never leave the machine). Three tabs:

- **Curate** — point it at your input folder(s), choose Kept/Unkept folders, hit *Run pipeline*. Live log while it works, summary when it's done. Advanced settings (similarity, detectors, export size, …) are in a collapsible panel.
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
