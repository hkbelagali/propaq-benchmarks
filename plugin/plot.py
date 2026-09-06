#!/usr/bin/env python3
"""Render the runtime plot from a plugin benchmark npz snapshot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.io_utils import load_records_npz  # noqa: E402
from run import PLOTS, RESULTS, plot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS / "runtime_vs_qaoa_layers.npz")
    parser.add_argument("--out", type=Path, default=PLOTS / "runtime_vs_qaoa_layers.png")
    args = parser.parse_args()

    rows = load_records_npz(str(args.results))
    if not rows:
        raise SystemExit(f"no samples found in {args.results}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, args.out)
    print(f"Wrote {args.out} and {args.out.with_suffix('.pgf')}")


if __name__ == "__main__":
    main()
