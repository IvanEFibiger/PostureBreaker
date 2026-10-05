from __future__ import annotations

import argparse
from pathlib import Path

from posture_guard.dataset import load_snapshots
from posture_guard.validation import build_run_comparison, build_view_difference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara dos corridas de validacion V2.")
    parser.add_argument("run_a", type=Path, help="JSONL de la corrida A")
    parser.add_argument("run_b", type=Path, help="JSONL de la corrida B")
    parser.add_argument("--baseline", default="good", help="Escenario baseline (default: good)")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Incluye metricas V1 legacy ademas de las V2",
    )
    args = parser.parse_args(argv)

    snapshots_a = load_snapshots(args.run_a)
    snapshots_b = load_snapshots(args.run_b)
    if not snapshots_a or not snapshots_b:
        print(
            f"Faltan snapshots: {args.run_a} ({len(snapshots_a)}) / "
            f"{args.run_b} ({len(snapshots_b)})"
        )
        return 1

    print(
        build_run_comparison(
            args.run_a.stem,
            snapshots_a,
            args.run_b.stem,
            snapshots_b,
            baseline_scenario=args.baseline,
            include_legacy=args.include_legacy,
        )
    )
    for path, snapshots in ((args.run_a, snapshots_a), (args.run_b, snapshots_b)):
        print(f"\n--- {path.stem} ---")
        print(build_view_difference(snapshots, include_legacy=args.include_legacy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
