from __future__ import annotations

import sys
from pathlib import Path

from posture_guard.dataset import (
    compare_to_baseline,
    format_comparison,
    format_summary,
    load_snapshots,
    separability,
    summarize_snapshots,
)

DEFAULT_PATH = Path("history/debug_snapshots.jsonl")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    snapshots = load_snapshots(path)
    if not snapshots:
        print(f"No hay snapshots en {path}")
        return 1

    # Group by scenario AND detected view, so "good @ left" vs "good @ right"
    # separate automatically without any manual monitor labeling.
    summary = summarize_snapshots(snapshots, group_by_view=True)
    baseline_label = argv[2] if len(argv) > 2 else sorted(summary)[0]
    print(f"Baseline: {baseline_label}\n")
    print(format_summary(summary))

    metric_names = sorted({name for stats in summary.values() for name in stats})
    if metric_names:
        print("\nSeparabilidad por metrica (mayor = discrimina mejor):")
        for name in metric_names:
            print(f"  {name:<24} {separability(summary, name):.2f}")

    comparison = compare_to_baseline(summary, baseline_label)
    if comparison:
        print(f"\nComparacion contra baseline '{baseline_label}':")
        print(format_comparison(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
