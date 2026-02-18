"""
Create pitch contour visualizations from the synthetic tone definitions.

Plots F0 (Hz) vs time (ms) for the four Mandarin tones. Output is suitable
for blog posts or documentation.
"""

import sys
from pathlib import Path

import numpy as np

# Import tone definitions from the generator (same contours as in the WAVs)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_tones import (
    DURATION_MS,
    F0_HIGH,
    F0_LOW,
    SAMPLE_RATE,
    f0_t1,
    f0_t2,
    f0_t3,
    f0_t4,
)

import matplotlib.pyplot as plt

# Output
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
PITCH_CONTOURS_PNG = FIGURES_DIR / "pitch_contours.png"


def get_time_axis_ms() -> np.ndarray:
    n = int(DURATION_MS * SAMPLE_RATE / 1000)
    t_sec = np.arange(n, dtype=float) / SAMPLE_RATE
    return t_sec * 1000  # ms


def plot_pitch_contours(save_path: Path | None = None) -> None:
    t_ms = get_time_axis_ms()
    contours = [
        (f0_t1(t_ms / 1000), "T1", "High level (flat)"),
        (f0_t2(t_ms / 1000), "T2", "Rising"),
        (f0_t3(t_ms / 1000), "T3", "Dipping then rising"),
        (f0_t4(t_ms / 1000), "T4", "Falling"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (f0, label, desc) in zip(axes, contours):
        ax.plot(t_ms, f0, color="C0", linewidth=2)
        ax.set_title(f"{label}: {desc}", fontsize=12)
        ax.set_ylabel("F0 (Hz)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_ms[-1])
        # Y-axis spans human tone range (~40 Hz) with small padding
        pad = (F0_HIGH - F0_LOW) * 0.25
        ax.set_ylim(F0_LOW - pad, F0_HIGH + pad)
    axes[0].set_xlabel("Time (ms)")
    axes[1].set_xlabel("Time (ms)")
    axes[2].set_xlabel("Time (ms)")
    axes[3].set_xlabel("Time (ms)")
    fig.suptitle("Mandarin tone pitch contours (synthetic)", fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def main() -> None:
    plot_pitch_contours(save_path=PITCH_CONTOURS_PNG)


if __name__ == "__main__":
    main()
