from __future__ import annotations

import sys
from pathlib import Path

from posture_guard.dataset import (
    compare_to_view_baselines,
    format_response_matrix,
    format_summary,
    format_view_comparison,
    load_snapshots,
    metric_response_matrix,
    separability,
    summarize_snapshots,
)

DEFAULT_PATH = Path("history/debug_snapshots.jsonl")
DEFAULT_BASELINE_SCENARIO = "good"


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    snapshots = load_snapshots(path)
    if not snapshots:
        print(f"No hay snapshots en {path}")
        return 1

    baseline_scenario = argv[2] if len(argv) > 2 else DEFAULT_BASELINE_SCENARIO

    # Group by scenario AND detected view, so each scenario is summarized under
    # the view it was captured in without any manual monitor labeling.
    summary = summarize_snapshots(snapshots, group_by_view=True)
    print(f"Escenario baseline: {baseline_scenario}\n")
    print(format_summary(summary))

    metric_names = sorted({name for stats in summary.values() for name in stats})
    if metric_names:
        print("\nSeparabilidad por metrica (mayor = discrimina mejor):")
        for name in metric_names:
            print(f"  {name:<24} {separability(summary, name):.2f}")

    print(
        "\nEffect es un indicador de separabilidad intra-metrica; "
        "no comparar su magnitud entre metricas distintas."
    )

    # Each scenario is compared only against the baseline of its own view.
    comparison = compare_to_view_baselines(summary, baseline_scenario)
    if comparison:
        print(f"\nComparacion por vista contra '{baseline_scenario}' de la misma vista:")
        print(format_view_comparison(comparison))

        print("\nMatriz de respuesta por escenario (efecto por metrica, ojo con el cross-talk):")
        print(format_response_matrix(metric_response_matrix(comparison)))
    else:
        print(f"\nNo hay baselines '{baseline_scenario}' por vista para comparar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
