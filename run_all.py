"""Run the complete experiment sequence from the repository root."""

from __future__ import annotations

import argparse
import subprocess
import sys


STAGES = [
    ("01", "experiments.run_experiment"),
    ("02", "experiments.experiment_topology"),
    ("03", "analysis.analyze_gesnr"),
    ("04", "experiments.experiment4_validation"),
    ("05", "experiments.experiment5_local_gesnr"),
    ("06", "experiments.experiment6_strict_validation"),
    ("06b", "experiments.experiment6B_numerical_verification"),
    ("07", "experiments.experiment7_spectral_detection"),
    ("08", "experiments.experiment8_general_covariance"),
    ("09", "experiments.experiment9_signal_geometry"),
    ("10", "experiments.experiment10_final_benchmark"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Experiments 1-10 in dependency order."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List stages without executing them.",
    )
    parser.add_argument(
        "--start-at",
        choices=[stage for stage, _ in STAGES],
        default=STAGES[0][0],
        help="First stage to execute (default: 01).",
    )
    parser.add_argument(
        "--stop-after",
        choices=[stage for stage, _ in STAGES],
        default=STAGES[-1][0],
        help="Last stage to execute (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for stage, module in STAGES:
            print(f"{stage}: python -m {module}")
        return 0

    start = next(i for i, item in enumerate(STAGES) if item[0] == args.start_at)
    stop = next(i for i, item in enumerate(STAGES) if item[0] == args.stop_after)
    if start > stop:
        raise SystemExit("--start-at must not come after --stop-after")

    for stage, module in STAGES[start : stop + 1]:
        print(f"\n=== Running stage {stage}: {module} ===", flush=True)
        subprocess.run([sys.executable, "-m", module], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
