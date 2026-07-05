"""
mrv CLI -- installed as ``mrv`` command via pip install mrv-lib.

Usage:
    mrv run config.yaml              # run rep + report
    mrv run config.yaml rep          # representation invariance only
    mrv run config.yaml report       # regenerate report from last result.json
    mrv download config.yaml         # download data
    mrv init                         # scaffold a config.yaml in current dir

Resolution invariance (Paper 2) is labels-first: fit your own regime models at
each frequency and call ``mrv.validate_res(labels=...)`` from the Python API.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_latest_json(cfg):
    report_dir = Path(cfg.get("validator", {}).get("report_dir", "reports"))
    if not report_dir.exists():
        raise FileNotFoundError(f"No reports directory: {report_dir}")
    candidates = sorted(
        report_dir.glob("*/result.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No result.json found under {report_dir}/")
    return str(candidates[0])


def _run_rep(cfg):
    """Run representation invariance validator."""
    from mrv.pipeline import run
    result = run(cfg=cfg, validator="rep")
    return result


def _run_report(cfg):
    """Regenerate report from the latest result.json."""
    from mrv.pipeline import report
    json_path = _find_latest_json(cfg)
    pdf = report(json_path)
    if pdf:
        print(f"PDF: {pdf}")


def _run_all(cfg):
    """Run rep + report (resolution invariance is labels-first via the Python API)."""
    steps = [
        ("Step 1/2: Representation Invariance", _run_rep),
        ("Step 2/2: Report", _run_report),
    ]
    for label, fn in steps:
        print(f"=== {label} ===")
        try:
            fn(cfg)
        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)
    print("=== Done ===")


_VALIDATORS = {
    "rep": _run_rep,
    "report": _run_report,
}


def _cmd_init(args):
    """Scaffold a config.yaml in the current directory."""
    dest = Path("config.yaml")
    if dest.exists() and not args.force:
        print("config.yaml already exists. Use --force to overwrite.")
        return

    src = Path(__file__).parent / "default_config.yaml"
    shutil.copy2(src, dest)
    print(f"Created {dest.resolve()}")
    print("Edit config.yaml to set your asset paths, then run: mrv run config.yaml")


def _cmd_run(args):
    """Run validations."""
    from mrv.utils.config import load
    from mrv.utils.log import setup
    cfg = load(args.config)
    setup(cfg)

    if args.validator:
        fn = _VALIDATORS.get(args.validator)
        if fn is None:
            print(f"Unknown validator: {args.validator}. Choose from: {', '.join(_VALIDATORS)}")
            return
        fn(cfg)
    else:
        _run_all(cfg)


def _cmd_download(args):
    """Download data."""
    from mrv.pipeline import download
    download(config=args.config)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mrv",
        description=(
            "Model Risk Validator -- specification invariance testing for "
            "financial models."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run validations + generate report.")
    p_run.add_argument("config", help="Path to config.yaml")
    p_run.add_argument(
        "validator", nargs="?", default=None,
        choices=list(_VALIDATORS.keys()),
        help="Run a specific validator only (default: all)",
    )

    p_dl = sub.add_parser("download", help="Download data (Yahoo Finance or IB Gateway).")
    p_dl.add_argument("config", nargs="?", default=None)

    p_init = sub.add_parser("init", help="Scaffold a config.yaml in the current directory.")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config.yaml")

    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "download":
        _cmd_download(args)
    elif args.command == "init":
        _cmd_init(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
