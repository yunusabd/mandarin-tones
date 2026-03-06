"""
Copy figures, audio, and conversation log into docs/ for GitHub Pages.
Run from repo root. Creates docs/figures/, docs/audio/synthetic/, docs/audio/syllables/,
and copies the conversation .md into docs/.

Figures are optimized when copied: resized to max width (default 1440px for retina)
and saved with compression. Per-image options (crop, max_width) can be set in
FIGURE_OPTIONS below — e.g. crop=(left, top, right, bottom) in pixels.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES_SRC = ROOT / "figures"
SYNTHETIC_SRC = ROOT / "synthetic_tones"
SYLLABLES_SRC = ROOT / "audio-cmn" / "64k" / "syllabs"

DOCS_FIGURES = DOCS / "figures"
DOCS_AUDIO_SYNTHETIC = DOCS / "audio" / "synthetic"
DOCS_AUDIO_SYLLABLES = DOCS / "audio" / "syllables"

FIGURES = [
    "pitch_contours.png",
    "macro_f1.png",
    "confusion_gemini3pro.png",
    "confusion_gemini31pro.png",
    "macro_f1_hao.png",
    "confusion_gemini3pro_hao.png",
    "macro_f1_local.png",
    "confusion_local.png",
    "9m-screenshot.png",
]
SYNTHETIC = [f"tone{i}.wav" for i in range(1, 5)]
SYLLABLES = ["cmn-bai1.mp3", "cmn-bai2.mp3", "cmn-bai3.mp3", "cmn-bai4.mp3"]

# Per-image overrides: "crop": (left, top, right, bottom) in pixels, and/or "max_width": int.
# Example: "9m-screenshot.png": {"crop": (0, 100, 800, 1200), "max_width": 720}
FIGURE_OPTIONS: dict[str, dict] = {}
DEFAULT_MAX_WIDTH = 1440


def optimize_and_copy_figure(src: Path, dest: Path, options: dict) -> None:
    img = Image.open(src)
    if src.suffix.lower() in (".jpg", ".jpeg"):
        img = img.convert("RGB")
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    w, h = img.size

    if "crop" in options:
        l, t, r, b = options["crop"]
        img = img.crop((l, t, r, b))
        w, h = img.size

    max_w = options.get("max_width", DEFAULT_MAX_WIDTH)
    if w > max_w:
        ratio = max_w / w
        new_size = (max_w, int(h * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in (".jpg", ".jpeg"):
        img.convert("RGB").save(dest, "JPEG", quality=88, optimize=True)
    else:
        img.save(dest, "PNG", optimize=True)


def copy_figures(src_dir: Path, dest_dir: Path, names: list[str]) -> tuple[list[str], list[str]]:
    copied, missing = [], []
    for name in names:
        src = src_dir / name
        if not src.exists():
            missing.append(str(src))
            continue
        dest = dest_dir / name
        try:
            optimize_and_copy_figure(src, dest, FIGURE_OPTIONS.get(name, {}))
            copied.append(name)
        except Exception:
            # Fallback: copy as-is if Pillow fails (e.g. not an image)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            copied.append(name)
    return copied, missing


def copy_files(src_dir: Path, dest_dir: Path, names: list[str]) -> tuple[list[str], list[str]]:
    copied, missing = [], []
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.exists():
            dest = dest_dir / name
            dest.write_bytes(src.read_bytes())
            copied.append(name)
        else:
            missing.append(str(src))
    return copied, missing


def main() -> None:
    print("Preparing docs/ for GitHub Pages...")
    all_missing = []

    copied, missing = copy_figures(FIGURES_SRC, DOCS_FIGURES, FIGURES)
    print(f"  figures: {len(copied)} copied (optimized) -> docs/figures/")
    all_missing.extend(missing)

    copied, missing = copy_files(SYNTHETIC_SRC, DOCS_AUDIO_SYNTHETIC, SYNTHETIC)
    print(f"  synthetic: {len(copied)} copied -> docs/audio/synthetic/")
    all_missing.extend(missing)

    copied, missing = copy_files(SYLLABLES_SRC, DOCS_AUDIO_SYLLABLES, SYLLABLES)
    print(f"  syllables: {len(copied)} copied -> docs/audio/syllables/")
    all_missing.extend(missing)

    conv_md = "cursor_testing_llms_for_mandarin_tone_r.md"
    conv_src = ROOT / conv_md
    if conv_src.exists():
        (DOCS / conv_md).write_bytes(conv_src.read_bytes())
        print(f"  conversation: {conv_md} -> docs/")
    else:
        all_missing.append(str(conv_src))

    if all_missing:
        print("\nMissing:")
        for m in all_missing:
            print(f"  - {m}")
        print("(Run generate_tones.py for WAVs; ensure audio-cmn/64k/syllabs has bai samples; keep conversation .md in repo root.)")
    else:
        print("\nDone. Enable GitHub Pages: Settings → Pages → Source: Deploy from branch → main → /docs.")


if __name__ == "__main__":
    main()
