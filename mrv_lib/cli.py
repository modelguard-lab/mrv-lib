"""
Command-line interface for mrv-lib.

Example:

    mrv-lib market_data.csv --resolution 5m 1h 1d --model HMM
"""

import argparse
from typing import List

import pandas as pd

from mrv_lib import Scanner, detect_boundary


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mrv-lib",
        description=(
            "Market Regime Validity diagnostics: "
            "Representation Stability (RSS) and Identifiability Index."
        ),
    )
    parser.add_argument(
        "csv",
        help="Path to OHLCV market data CSV file.",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        nargs="+",
        default=["5m", "1h", "1d"],
        help="Temporal resolutions to test, e.g. 5m 1h 1d (default: 5m 1h 1d).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="HMM",
        help="Regime model type label (passed through to Scanner; default: HMM).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)

    data = pd.read_csv(args.csv)

    scanner = Scanner(resolution=args.resolution)
    results = scanner.run_representation_test(data, model=args.model)

    boundary = detect_boundary(data)

    print(f"File: {args.csv}")
    print(f"Resolutions: {args.resolution}")
    print(f"Model: {args.model}")
    print(f"RSS (Representation Stability Score): {results.rss_score:.4f}")
    print(
        f"Identifiability Index: {boundary.index:.4f} "
        f"(threshold={boundary.threshold:.2f})"
    )
    if boundary.is_collapsed:
        print("Status: COLLAPSED (Inference Collapse Zone)")
    else:
        print("Status: STABLE (outside collapse zone)")


if __name__ == "__main__":
    main()

