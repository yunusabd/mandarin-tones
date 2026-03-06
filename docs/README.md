<https://yunusabd.github.io/mandarin-tones/>

# GitHub Pages (blog)

This folder is the source for the project’s GitHub Pages site.

**Tone evaluation (record your own voice):** With `conda activate mandarin`, from the repo root run the same tone-eval models on a single file: `python scripts/run_tone_eval.py --audio-file path/to/my.wav`, or record from the mic: `python scripts/run_tone_eval.py --record`. Use `--output results/recorded.csv` to save results to CSV. Recording requires `sounddevice` (in `requirements.txt`).

- **Content:** `index.html` — blog post “Can multi-modal LLMs recognize Mandarin tones?” with figures and playable audio.
- **Conda:** The project uses the `mandarin` env. Run `conda activate mandarin` before the commands below.
- **Assets:** After generating figures and audio in the repo, run from the repo root:
  ```bash
  conda activate mandarin && python scripts/prepare_github_pages.py
  ```
  That copies into `docs/` (figures are **resized** to max width 1440px and **compressed**):
  - `figures/pitch_contours.png`, `figures/macro_f1.png`, `figures/confusion_gemini3pro.png`, `figures/9m-screenshot.png`
  - To **add a new image**: put it in `figures/`, add its filename to `FIGURES` in `scripts/prepare_github_pages.py`, and add `<img src="figures/…">` in `index.html`.
  - To **crop or set max width** for one image, set `FIGURE_OPTIONS` in that script, e.g. `"9m-screenshot.png": {"crop": (0, 100, 800, 1200), "max_width": 720}`.
  - `audio/synthetic/tone1.wav` … `tone4.wav`
  - `audio/syllables/cmn-bai1.mp3` … `cmn-bai4.mp3`
  - `cursor_testing_llms_for_mandarin_tone_r.md` (AI conversation log; keep the canonical file in repo root)

**Test locally:** Serve the docs over HTTP from the repo root:
```bash
conda activate mandarin && python -m http.server 8000 --directory docs
```
Then open **http://localhost:8000**.

**Enable Pages:** Repository → Settings → Pages → Source: **Deploy from a branch** → Branch: **main** → Folder: **/docs** → Save. The site will be at `https://<user>.github.io/mandarin-tones/`.
