"""
Copy figures, audio, and conversation log into docs/ for GitHub Pages.
Run from repo root. Creates docs/figures/, docs/audio/synthetic/, docs/audio/syllables/,
and copies the conversation .md into docs/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES_SRC = ROOT / "figures"
SYNTHETIC_SRC = ROOT / "synthetic_tones"
SYLLABLES_SRC = ROOT / "audio-cmn" / "64k" / "syllabs"

DOCS_FIGURES = DOCS / "figures"
DOCS_AUDIO_SYNTHETIC = DOCS / "audio" / "synthetic"
DOCS_AUDIO_SYLLABLES = DOCS / "audio" / "syllables"

FIGURES = ["pitch_contours.png", "macro_f1.png", "confusion_gemini3pro.png"]
SYNTHETIC = [f"tone{i}.wav" for i in range(1, 5)]
SYLLABLES = ["cmn-bai1.mp3", "cmn-bai2.mp3", "cmn-bai3.mp3", "cmn-bai4.mp3"]


def copy_files(src_dir: Path, dest_dir: Path, names: list[str]) -> list[str]:
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

    copied, missing = copy_files(FIGURES_SRC, DOCS_FIGURES, FIGURES)
    print(f"  figures: {len(copied)} copied -> docs/figures/")
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
