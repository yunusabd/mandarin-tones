"""
Compute per-model confusion matrix, precision, recall, and F1 for tone predictions.

Reads a results CSV from run_tone_eval.py (columns: model, audio_file, true_tone,
predicted_tone, ...). Rows with empty predicted_tone are excluded from metrics.

Usage:
  python scripts/analyze_tone_results.py results/tone_eval_15syllables.csv
  python scripts/analyze_tone_results.py results/tone_eval_cai.csv --output results/metrics.txt
"""

import argparse
import csv
import sys
from pathlib import Path

TONES = [1, 2, 3, 4]

# TABLE II: number of samples per tone in training data (baseline prior)
BASELINE_COUNTS = {1: 160_885, 2: 179_606, 3: 122_707, 4: 253_441}


def baseline_percentages() -> dict[int, float]:
    total = sum(BASELINE_COUNTS.values())
    return {t: round(100 * BASELINE_COUNTS[t] / total, 1) for t in TONES}


def load_results(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def confusion_matrix(rows: list[dict], model: str) -> list[list[int]]:
    """4x4 matrix: rows = true tone, cols = predicted tone. Only rows with valid predicted_tone."""
    cm = [[0] * 4 for _ in range(4)]
    for r in rows:
        if r.get("model") != model:
            continue
        try:
            true_t = int(r["true_tone"])
            pred = (r.get("predicted_tone") or "").strip()
            if pred not in ("1", "2", "3", "4"):
                continue
            pred_t = int(pred)
            if 1 <= true_t <= 4 and 1 <= pred_t <= 4:
                cm[true_t - 1][pred_t - 1] += 1
        except (ValueError, KeyError):
            continue
    return cm


def precision_recall_f1(cm: list[list[int]], tone: int) -> tuple[float, float, float]:
    """Precision, recall, F1 for the given tone (1-4)."""
    t = tone - 1
    tp = cm[t][t]
    fp = sum(cm[i][t] for i in range(4) if i != t)
    fn = sum(cm[t][j] for j in range(4) if j != t)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def metrics_per_model(rows: list[dict], model: str) -> str:
    cm = confusion_matrix(rows, model)
    lines = [f"\n{'='*60}", f"Model: {model}", "=" * 60]
    # Confusion matrix (rows = true, cols = pred)
    lines.append("\nConfusion matrix (rows = true tone, cols = predicted tone):")
    lines.append("        pred 1   pred 2   pred 3   pred 4")
    for i, row in enumerate(cm):
        lines.append(f"true {i+1}   " + "   ".join(f"{v:6d}" for v in row))
    # Per-tone metrics
    lines.append("\nPer-tone metrics:")
    lines.append("Tone   Precision  Recall    F1")
    macro_f1 = 0.0
    for tone in TONES:
        p, r, f1 = precision_recall_f1(cm, tone)
        macro_f1 += f1
        lines.append(f"  {tone}    {p:.4f}     {r:.4f}     {f1:.4f}")
    macro_f1 /= 4
    lines.append(f"\nMacro F1: {macro_f1:.4f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze tone eval CSV: confusion matrix, P/R/F1.")
    parser.add_argument("csv", type=Path, help="Results CSV from run_tone_eval.py")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Write report to file")
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else Path.cwd() / args.csv
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        return 1

    rows = load_results(csv_path)
    models = sorted({r["model"] for r in rows})
    if not models:
        print("No model rows in CSV.", file=sys.stderr)
        return 1

    pct = baseline_percentages()
    report = [
        f"Tone evaluation metrics (from {csv_path})",
        f"Total rows: {len(rows)}",
        "",
        f"Note: Training-data baseline (TABLE II): T1 {pct[1]}%, T2 {pct[2]}%, T3 {pct[3]}%, T4 {pct[4]}%.",
        "A bias toward predicting 4 may reflect that prior as well as acoustic cues.",
        "",
    ]
    for model in models:
        report.append(metrics_per_model(rows, model))

    text = "\n".join(report)
    if args.output:
        out = args.output if args.output.is_absolute() else Path.cwd() / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
