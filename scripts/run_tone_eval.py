"""
Send Mandarin tone audio (WAV or MP3) to selected LLMs with a tone-definition prompt;
save results to CSV for analysis.

Loads .env for API keys (OPENAI_API_KEY, GOOGLE_API_KEY from GEMINI_API_KEY).

Usage:
  python run_tone_eval.py
      → synthetic_tones/ + manifest.json (tone1.wav … tone4.wav)
  python run_tone_eval.py --audio-dir audio_syllabs --manifest audio_syllabs/manifest_cai.json
      → audio_syllabs/ with cmn-cai1.mp3 … cmn-cai4.mp3 (put those files in audio_syllabs/)
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
from pathlib import Path

# Load .env before litellm
_root = Path(__file__).resolve().parent.parent
_env_path = _root / ".env"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

import litellm

# Models to query. All support audio input + text output.
# Gemini: 2.0 Flash, 2.5 Pro, etc. (https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate)
# We do not pass modalities/audio for Gemini so the API returns text only.
MODELS = [
    "openai/gpt-audio-2025-08-28",
    "openai/gpt-4o-audio-preview",  # full 4o audio; mini often refuses to process audio
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.0-flash",  # GA; audio in + text out per Vertex migration table
]

DEFAULT_AUDIO_DIR = _root / "synthetic_tones"
DEFAULT_MANIFEST = _root / "synthetic_tones" / "manifest.json"
RESULTS_DIR = _root / "results"

TONE_DEFINITIONS = """In Mandarin Chinese, syllables can have one of four lexical tones based on pitch contour (we ignore the 5th, neutral tone):
- Tone 1: flat, relatively high pitch
- Tone 2: rising pitch
- Tone 3: pitch dips then rises
- Tone 4: pitch falls from high to low

Listen to the attached audio. It is a single syllable with one of these four pitch contours.

Reply with:
1) The pinyin you heard, including the tone number (e.g. cai1, ma2, lü3).
2) The tone number alone: 1, 2, 3, or 4."""


def load_manifest(manifest_path: Path) -> dict[str, int]:
    with open(manifest_path) as f:
        return json.load(f)


def encode_audio(path: Path) -> tuple[str, str]:
    """Return (base64_data, format). Same as docs: raw file bytes, base64.b64encode(...).decode('utf-8')."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    fmt = "mp3" if path.suffix.lower() == ".mp3" else "wav"
    return b64, fmt


def parse_predicted_tone(content: str | None) -> str:
    """Extract tone 1–4 from model response; return '' if unclear."""
    if not content:
        return ""
    content = content.strip()
    # Prefer explicit "tone N" or "Tone N"
    m = re.search(r"\b[Tt]one\s*[:\s]*([1-4])\b", content, re.I)
    if m:
        return m.group(1)
    # Prefer digit after "2)" (we asked for "2) The tone number alone")
    m = re.search(r"2\)\s*([1-4])\b", content)
    if m:
        return m.group(1)
    # Standalone digit 1–4 that is NOT a list label (not followed by ")" )
    m = re.search(r"\b([1-4])(?!\))\b", content)
    if m:
        return m.group(1)
    # Fallback: tone digit at end of pinyin (e.g. cai4 -> 4)
    m = re.search(r"[a-zü]+([1-4])\b", content, re.I)
    if m:
        return m.group(1)
    return ""


def parse_heard_pinyin(content: str | None) -> str:
    """Extract pinyin with tone (e.g. cai1, ma2) from model response; return '' if not found."""
    if not content:
        return ""
    content = content.strip()
    # Pinyin syllable (letters, optional ü) followed by tone 1-4
    m = re.search(r"\b([a-zü]+)([1-4])\b", content, re.I)
    if m:
        return f"{m.group(1).lower()}{m.group(2)}"
    return ""


def run_one(model: str, audio_path: Path, true_tone: int) -> tuple[str, str, str]:
    """Call model with audio; return (predicted_tone, heard_pinyin, raw_content)."""
    encoded, fmt = encode_audio(audio_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TONE_DEFINITIONS},
                {
                    "type": "input_audio",
                    "input_audio": {"data": encoded, "format": fmt},
                },
            ],
        },
    ]
    try:
        kwargs = {"model": model, "messages": messages, "timeout": 90}
        if model.startswith("openai/"):
            # OpenAI audio models require modalities + audio output config even for text reply
            kwargs["modalities"] = ["text", "audio"]
            kwargs["audio"] = {"voice": "alloy", "format": "wav"}
        # Gemini: do not pass modalities or audio; it returns "only supports text output" otherwise
        resp = litellm.completion(**kwargs)
        msg = resp.choices[0].message
        # With audio output, text may be in content or in audio.transcript
        content = (getattr(msg, "content", None) or "").strip()
        if not content and getattr(msg, "audio", None) is not None:
            a = msg.audio
            content = (getattr(a, "transcript", None) or getattr(a, "text", None) or "").strip()
        if not content and hasattr(msg, "__dict__"):
            # Fallback: capture any text-like field for debugging
            for key in ("content", "text", "transcript"):
                val = getattr(msg, key, None)
                if val and isinstance(val, str):
                    content = val.strip()
                    break
        pred = parse_predicted_tone(content)
        pinyin = parse_heard_pinyin(content)
        return pred, pinyin, content
    except Exception as e:
        return "", "", str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tone evaluation on audio files.")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory containing audio files (default: synthetic_tones)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON manifest mapping filename -> tone 1-4 (default: synthetic_tones/manifest.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: results/tone_eval.csv or results/tone_eval_cai.csv)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names; if set, only these models are run (e.g. gemini/gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge new results into existing output CSV (replaces rows for same model+audio_file)",
    )
    args = parser.parse_args()
    audio_dir = args.audio_dir if args.audio_dir.is_absolute() else _root / args.audio_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else _root / args.manifest

    manifest = load_manifest(manifest_path)
    files = sorted(manifest.keys(), key=lambda f: manifest[f])
    models_to_run = [m.strip() for m in args.models.split(",")] if args.models else MODELS
    # Count how many we will actually run (existing files only)
    to_run = [(m, f) for m in models_to_run for f in files if (audio_dir / f).exists()]
    total = len(to_run)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = args.output
    if out_csv is None:
        if "15syllables" in str(manifest_path):
            out_csv = RESULTS_DIR / "tone_eval_15syllables.csv"
        elif "cai" in str(manifest_path):
            out_csv = RESULTS_DIR / "tone_eval_cai.csv"
        else:
            out_csv = RESULTS_DIR / "tone_eval.csv"
    out_csv = out_csv if out_csv.is_absolute() else _root / out_csv

    fieldnames = ["model", "audio_file", "true_tone", "predicted_tone", "heard_pinyin", "raw_response"]
    existing_rows: list[dict[str, str | int]] = []
    if args.append and out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        replace_keys = {(m, f) for m, f in to_run}
        existing_rows = [r for r in existing_rows if (r["model"], r["audio_file"]) not in replace_keys]
        print(f"Append mode: keeping {len(existing_rows)} existing rows, running {total} new.")

    rows = []
    for idx, (model, filename) in enumerate(to_run, start=1):
        true_tone = manifest[filename]
        audio_path = audio_dir / filename
        print(f"  [{idx}/{total}] {model} / {filename} ...", flush=True)
        pred, heard_pinyin, raw = run_one(model, audio_path, true_tone)
        rows.append({
            "model": model,
            "audio_file": filename,
            "true_tone": true_tone,
            "predicted_tone": pred or "",
            "heard_pinyin": heard_pinyin or "",
            "raw_response": raw.replace("\n", " ").strip(),
        })

    all_rows = existing_rows + rows
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_csv} ({len(rows)} new)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
