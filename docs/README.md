# GitHub Pages (blog)

This folder is the source for the project’s GitHub Pages site.

- **Content:** `index.html` — blog post “Can multi-modal LLMs recognize Mandarin tones?” with figures and playable audio.
- **Assets:** After generating figures and audio in the repo, run from the repo root:
  ```bash
  python scripts/prepare_github_pages.py
  ```
  That copies into `docs/`:
  - `figures/pitch_contours.png`, `figures/macro_f1.png`
  - `audio/synthetic/tone1.wav` … `tone4.wav`
  - `audio/syllables/cmn-bai1.mp3` … `cmn-bai4.mp3`
  - `cursor_testing_llms_for_mandarin_tone_r.md` (AI conversation log; keep the canonical file in repo root)

**Test locally:** The conversation log is loaded via `fetch()`, which fails with `file://` (CORS). Serve the docs over HTTP from the repo root:
```bash
python -m http.server 8000 --directory docs
```
Then open **http://localhost:8000** in your browser.

**Enable Pages:** Repository → Settings → Pages → Source: **Deploy from a branch** → Branch: **main** → Folder: **/docs** → Save. The site will be at `https://<user>.github.io/mandarin-tones/`.
