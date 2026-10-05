from __future__ import annotations

import sys
from pathlib import Path

from posture_guard.dataset import format_summary, load_snapshots, separability, summarize_snapshots

DEFAULT_PATH = Path("history/debug_snapshots.jsonl")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    snapshots = load_snapshots(path)
    if not snapshots:
        print(f"No hay snapshots en {path}")
        return 1

    summary = summarize_snapshots(snapshots)
    print(format_summary(summary))

    metric_names = sorted({name for stats in summary.values() for name in stats})
    if metric_names:
        print("\nSeparabilidad por metrica (mayor = discrimina mejor):")
        for name in metric_names:
            print(f"  {name:<24} {separability(summary, name):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
