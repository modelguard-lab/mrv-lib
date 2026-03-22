"""
mrv — Model Risk Validator

Usage:
    python run.py download config.yaml     # download data from IB
    python run.py run config.yaml          # validate → report
    python run.py run config.yaml rep      # explicit validator
    python run.py report                   # latest result.json → pdf
    python run.py report path/result.json  # specific json → pdf
"""

import argparse

from mrv.pipeline import run, download, report
from mrv.utils.config import load

from pathlib import Path

_VALIDATORS = ("rep",)  # res, temp: planned


def _find_latest_json(cfg):
    """Find the most recent result.json under report_dir."""
    report_dir = Path(cfg.get("validator", {}).get("report_dir", "reports"))
    if not report_dir.exists():
        raise FileNotFoundError(f"No reports directory: {report_dir}")
    candidates = sorted(report_dir.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No result.json found under {report_dir}/")
    print(f"Using: {candidates[0]}")
    return str(candidates[0])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mrv",
        description="Model Risk Validator — regime diagnostics for financial institutions.",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Validate → report.")
    p_run.add_argument("config", nargs="?", default=None)
    p_run.add_argument("validator", nargs="?", default="rep", choices=_VALIDATORS)

    p_dl = sub.add_parser("download", help="Download data only.")
    p_dl.add_argument("config", nargs="?", default=None)

    p_rpt = sub.add_parser("report", help="Generate PDF from JSON (default: latest).")
    p_rpt.add_argument("json", nargs="?", default=None, help="Path to result.json. If omitted, uses latest.")
    p_rpt.add_argument("-t", "--template", default=None)
    p_rpt.add_argument("-c", "--config", default=None)

    args = parser.parse_args(argv)

    if args.command == "run":
        run(config=args.config, validator=args.validator)
    elif args.command == "download":
        download(config=args.config)
    elif args.command == "report":
        cfg = load(args.config) if args.config else load()
        json_path = args.json
        if json_path is None:
            json_path = _find_latest_json(cfg)
        report(json_path=json_path, template=args.template, cfg=cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
