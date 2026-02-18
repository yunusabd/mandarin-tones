"""
Plot macro F1 per model from tone evaluation CSV. For blog and reports.
"""

import argparse
import csv
import sys
from pathlib import Path

# Reuse analysis logic
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tone_results import (
    TONES,
    confusion_matrix,
    load_results,
    precision_recall_f1,
)

import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "results" / "tone_eval_15syllables.csv"


def macro_f1(rows: list[dict], model: str) -> float:
    cm = confusion_matrix(rows, model)
    total = 0.0
    for t in TONES:
        _, _, f1 = precision_recall_f1(cm, t)
        total += f1
    return total / 4.0


def short_name(model: str) -> str:
    if "gemini-3-pro" in model:
        return "Gemini 3.0 Pro"
    if "gemini-2.5-pro" in model:
        return "Gemini 2.5 Pro"
    if "gemini-2.0-flash" in model:
        return "Gemini 2.0 Flash"
    if "gpt-4o-audio" in model:
        return "GPT-4o Audio"
    if "gpt-audio" in model:
        return "GPT Audio"
    return model.split("/")[-1]


def plot_macro_f1(csv_path: Path, save_path: Path | None = None) -> None:
    rows = load_results(csv_path)
    models = sorted({r["model"] for r in rows})
    scores = [macro_f1(rows, m) for m in models]
    labels = [short_name(m) for m in models]
    colors = plt.cm.viridis([0.2 + 0.6 * i / max(len(models) - 1, 1) for i in range(len(models))])

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro F1")
    ax.set_title("Mandarin tone recognition: macro F1 by model (15 syllables × 4 tones)")
    ax.axvline(0.25, color="gray", linestyle="--", alpha=0.7, label="Random (25%)")
    for bar, s in zip(bars, scores):
        ax.text(s + 0.02, bar.get_y() + bar.get_height() / 2, f"{s:.2f}", va="center", fontsize=10)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot macro F1 bar chart from tone eval CSV.")
    parser.add_argument("csv", type=Path, nargs="?", default=DEFAULT_CSV, help="Results CSV")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1
    out = args.output or FIGURES_DIR / "macro_f1.png"
    out = out if out.is_absolute() else Path.cwd() / out
    plot_macro_f1(csv_path, save_path=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
