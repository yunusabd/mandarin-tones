"""
Generate synthetic WAV files with single frequency curves for the four Mandarin tones.

T1: flat, relatively high F0
T2: rising frequency
T3: dip then rise
T4: fall from relatively high

Parameters can be overridden by results/suggested_tone_params.json (from scripts/analyze_bai_tones.py -o results/suggested_tone_params.json).
Output: 16-bit PCM WAV in synthetic_tones/ plus a manifest for LLM eval.
"""

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parent.parent
_PARAMS_FILE = _ROOT / "results" / "suggested_tone_params.json"

# --- Default parameters (overridden by _PARAMS_FILE if present) ---
DURATION_MS = 280  # typical syllable length
SAMPLE_RATE = 16000  # Hz
RANGE_HZ = 40.0
F0_HIGH = 220.0  # Hz (relatively high)
F0_LOW = F0_HIGH - RANGE_HZ  # 180 Hz (lower end)

if _PARAMS_FILE.exists():
    _p = json.loads(_PARAMS_FILE.read_text())
    DURATION_MS = int(_p.get("DURATION_MS", DURATION_MS))
    RANGE_HZ = float(_p.get("RANGE_HZ", RANGE_HZ))
    F0_HIGH = float(_p.get("F0_HIGH", F0_HIGH))
    F0_LOW = float(_p.get("F0_LOW", F0_LOW))

F0_MID = (F0_HIGH + F0_LOW) / 2  # T3 start
F0_DIP = F0_LOW  # T3 dips to bottom of range
AMPLITUDE = 0.8  # linear, before int16 conversion
FADE_MS = 10     # short fade-in/fade-out to avoid clicks

OUTPUT_DIR = _ROOT / "synthetic_tones"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def _time_axis(duration_ms: float, sample_rate: int) -> np.ndarray:
    n = int(duration_ms * sample_rate / 1000)
    return np.arange(n, dtype=float) / sample_rate


def _fade(n_samples: int, sample_rate: int, fade_ms: float) -> np.ndarray:
    fade_len = int(fade_ms * sample_rate / 1000)
    fade_len = min(fade_len, n_samples // 2)
    up = np.linspace(0, 1, fade_len)
    down = np.linspace(1, 0, fade_len)
    mid = np.ones(n_samples - 2 * fade_len)
    return np.concatenate([up, mid, down]).astype(np.float64)


def f0_t1(t: np.ndarray) -> np.ndarray:
    """T1: flat tone, relatively high frequency."""
    return np.full_like(t, F0_HIGH)


def f0_t2(t: np.ndarray) -> np.ndarray:
    """T2: rising frequency."""
    progress = t / (t[-1] - t[0]) if t[-1] > t[0] else np.ones_like(t)
    return F0_LOW + (F0_HIGH - F0_LOW) * progress


def f0_t3(t: np.ndarray) -> np.ndarray:
    """T3: first dips then rises."""
    n = len(t)
    t_end = t[-1] - t[0] if t[-1] > t[0] else 1.0
    # First half: mid -> dip; second half: dip -> high
    f0 = np.empty_like(t)
    mid_i = n // 2
    progress_first = np.linspace(0, 1, mid_i)
    progress_second = np.linspace(0, 1, n - mid_i)
    f0[:mid_i] = F0_MID + (F0_DIP - F0_MID) * progress_first
    f0[mid_i:] = F0_DIP + (F0_HIGH - F0_DIP) * progress_second
    return f0


def f0_t4(t: np.ndarray) -> np.ndarray:
    """T4: frequency falls from relatively high value."""
    progress = t / (t[-1] - t[0]) if t[-1] > t[0] else np.ones_like(t)
    return F0_HIGH + (F0_LOW - F0_HIGH) * progress


def f0_to_wav(f0: np.ndarray, sample_rate: int, amplitude: float, fade_ms: float) -> np.ndarray:
    """Convert F0 curve (Hz per sample) to int16 mono WAV samples."""
    # phase[t] = 2*pi * cumsum(F0)/fs  (integral of angular frequency)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    samples = amplitude * np.sin(phase)
    # Apply fade to avoid clicks
    n = len(samples)
    fade = _fade(n, sample_rate, fade_ms)
    samples = samples * fade
    # Clip and convert to int16
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767).astype(np.int16)


def generate_all(
    output_dir: Path,
    duration_ms: float = DURATION_MS,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    fade_ms: float = FADE_MS,
) -> dict:
    """Generate tone1.wav .. tone4.wav and return manifest {filename: tone_label}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t = _time_axis(duration_ms, sample_rate)
    contours = [
        (f0_t1, 1, "tone1.wav"),
        (f0_t2, 2, "tone2.wav"),
        (f0_t3, 3, "tone3.wav"),
        (f0_t4, 4, "tone4.wav"),
    ]
    manifest = {}
    for f0_fn, tone_num, filename in contours:
        f0 = f0_fn(t)
        samples = f0_to_wav(f0, sample_rate, amplitude, fade_ms)
        path = output_dir / filename
        wavfile.write(str(path), sample_rate, samples)
        manifest[filename] = tone_num
    return manifest


def main() -> None:
    manifest = generate_all(
        OUTPUT_DIR,
        duration_ms=DURATION_MS,
        sample_rate=SAMPLE_RATE,
        amplitude=AMPLITUDE,
        fade_ms=FADE_MS,
    )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote 4 WAVs to {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
