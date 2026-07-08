"""
mrv.validator.report -- Fill LaTeX template with JSON data and compile to PDF.

All text lives in the template. This module only:
1. Replaces ``{{KEY}}`` placeholders with data values
2. Evaluates ``%% IF_xxx`` / ``%% ELSE`` / ``%% ELIF_xxx`` / ``%% ENDIF`` conditionals
3. Expands ``%% BEGIN_ASSET`` / ``%% END_ASSET`` blocks per asset
4. Compiles .tex -> .pdf via pdflatex

Both representation (Paper 1) and resolution (Paper 2) validator JSON are
supported.  The renderer branches on ``data["test"]``: the resolution case is
rendered from ``overall_mean_ari`` (the Spearman / factor-set sections are
omitted, since resolution invariance has no ordering layer).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional

from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD

logger = logging.getLogger(__name__)

__all__ = [
    "generate_report",
]

_RES_TEST = "resolution_invariance"


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _tex(s: str) -> str:
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    return s


def _ari_table(labels: list, values: list, threshold: float = 0.65) -> str:
    n = len(labels)
    cols = " ".join(["l"] + ["r"] * n)
    header = " & ".join([""] + [f"\\textbf{{Set {j}}}" for j in range(n)]) + " \\\\"
    rows = ""
    for i, row in enumerate(values):
        cells = []
        for j, v in enumerate(row):
            if i == j:
                cells.append(f"{v:.3f}")
            elif v < threshold:
                cells.append(f"\\cellcolor{{mrvredbg}}{v:.3f}")
            else:
                cells.append(f"{v:.3f}")
        rows += f"  \\textbf{{Set {i}}} & " + " & ".join(cells) + " \\\\\n"
    return (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header}\n"
            f"\\midrule\n{rows}\\bottomrule\n\\end{{tabular}}")


# ---------------------------------------------------------------------------
# Conditional block engine
# ---------------------------------------------------------------------------

def _eval_conditionals(text: str, flags: Dict[str, bool]) -> str:
    """
    Process conditional blocks. Supports:
        %% IF_xxx ... %% ELIF_yyy ... %% ELSE ... %% ENDIF

    *flags* maps condition names (e.g. "PARTITION_PASS") to bool.
    """
    result = []
    stack = []  # [(active, consumed)]
    active = True

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("%% IF_"):
            cond = stripped[6:]
            val = flags.get(cond, False)
            stack.append((active, val))
            active = active and val
        elif stripped.startswith("%% ELIF_"):
            if not stack:
                continue
            cond = stripped[8:]
            val = flags.get(cond, False)
            parent_active, prev_consumed = stack[-1]
            active = parent_active and (not prev_consumed) and val
            if active:
                stack[-1] = (parent_active, True)
        elif stripped == "%% ELSE":
            if not stack:
                continue
            parent_active, prev_consumed = stack[-1]
            active = parent_active and not prev_consumed
        elif stripped == "%% ENDIF":
            # Restore *active* to the context that was in effect before this IF
            # opened. The frame's first element is exactly that parent-active
            # state (parent-active AND every ancestor branch taken), so a nested
            # conditional inside a non-taken outer branch cannot re-enable output.
            if stack:
                parent_active, _ = stack.pop()
                active = parent_active
            else:
                active = True
        else:
            if active:
                result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Per-asset value extraction (branches on validator type)
# ---------------------------------------------------------------------------

def _asset_mean_ari(a: Dict[str, Any], is_res: bool) -> float:
    """Mean ARI for an asset, from res or rep keys."""
    if is_res:
        v = a.get("overall_mean_ari")
    else:
        v = a.get("mean_ari")
    return float(v) if v is not None else float("nan")


# ---------------------------------------------------------------------------
# Asset block expansion
# ---------------------------------------------------------------------------

_ASSET_RE = re.compile(r"%% BEGIN_ASSET\s*\n(.*?)%% END_ASSET", re.DOTALL)


def _expand_assets(text: str, data: Dict[str, Any]) -> str:
    match = _ASSET_RE.search(text)
    if not match:
        return text

    is_res = data.get("test") == _RES_TEST
    block_tpl = match.group(1)
    assets = data.get("assets", {})
    ari_threshold = data.get("ari_threshold", ARI_THRESHOLD)
    sp_threshold = data.get("spearman_threshold", SPEARMAN_THRESHOLD)
    expanded = ""

    for name, a in assets.items():
        ari_data = a["ari_matrix"]
        mean_ari = _asset_mean_ari(a, is_res)
        ari_is_nan = not math.isfinite(mean_ari)
        if ari_is_nan:
            # Insufficient data: the mean ARI is NaN/None. A bare
            # ``nan < threshold`` is False, which would fail open to a green
            # "Pass"; render a neutral N/A instead so this cannot masquerade as
            # a pass while the dashboard shows a fail.
            color = "mrvgray"
        else:
            color = "mrvred" if mean_ari < ari_threshold else "mrvgreen"

        if is_res:
            # Resolution invariance: no ordering layer, no min-ARI key.
            n_units = len(a.get("frequencies", ari_data.get("labels", [])))
            min_ari = mean_ari
            mean_sp_str = "N/A"
            if ari_is_nan:
                finding = (
                    f"\\textcolor{{mrvgray}}{{\\textbf{{N/A: insufficient data}}}}: "
                    f"cross-frequency ARI could not be computed for {_tex(name)} "
                    f"(too few aligned observations)."
                )
            elif mean_ari < ari_threshold:
                finding = (
                    f"\\textcolor{{mrvred}}{{\\textbf{{Partition: FAIL}}}}: "
                    f"cross-frequency ARI = {mean_ari:.3f} "
                    f"($<$ {ari_threshold}). "
                    f"Regime labels for {_tex(name)} change across time frequencies."
                )
            else:
                finding = (
                    f"\\textcolor{{mrvgreen}}{{\\textbf{{Pass}}}}: "
                    f"cross-frequency ARI = {mean_ari:.3f}; "
                    f"acceptable stability for {_tex(name)}."
                )
        elif ari_is_nan:
            n_units = a.get("n_factor_sets", a.get("n_specs", 0))
            min_ari = mean_ari
            mean_sp = a.get("mean_spearman", float("nan"))
            mean_sp_str = f"{mean_sp:.4f}" if math.isfinite(mean_sp) else "N/A"
            finding = (
                f"\\textcolor{{mrvgray}}{{\\textbf{{N/A: insufficient data}}}}: "
                f"mean ARI could not be computed for {_tex(name)} "
                f"(too few observations)."
            )
        else:
            n_units = a.get("n_factor_sets", a.get("n_specs", 0))
            min_ari = float(a.get("min_ari", mean_ari))
            mean_sp = a.get("mean_spearman", float("nan"))
            mean_sp_str = f"{mean_sp:.4f}"
            if mean_ari < ari_threshold and mean_sp >= sp_threshold:
                finding = (
                    f"\\textcolor{{mrvred}}{{\\textbf{{Partition: FAIL}}}}: "
                    f"ARI = {mean_ari:.3f}. "
                    f"\\textcolor{{mrvgreen}}{{\\textbf{{Ordering: PASS}}}}: "
                    f"Spearman = {mean_sp:.3f}. "
                    f"Categorical labels are unstable for {_tex(name)}, "
                    "but risk ranking is preserved."
                )
            elif mean_ari < ari_threshold:
                finding = (
                    f"\\textcolor{{mrvred}}{{\\textbf{{Partition: FAIL}}}}: "
                    f"ARI = {mean_ari:.3f}. "
                    f"\\textcolor{{mrvred}}{{\\textbf{{Ordering: FAIL}}}}: "
                    f"Spearman = {mean_sp:.3f}. "
                    f"Both labels and risk ordering are unstable for {_tex(name)}."
                )
            else:
                finding = (
                    f"\\textcolor{{mrvgreen}}{{\\textbf{{Pass}}}}: "
                    f"ARI = {mean_ari:.3f}, Spearman = {mean_sp:.3f}; "
                    f"acceptable stability for {_tex(name)}."
                )

        b = block_tpl
        b = b.replace("{{ASSET_NAME}}", _tex(name))
        b = b.replace("{{N_OBS}}", f"{a.get('n_obs', 0):,}")
        b = b.replace("{{N_FACTOR_SETS}}", str(n_units))
        b = b.replace("{{MEAN_ARI}}", "N/A" if ari_is_nan else f"{mean_ari:.4f}")
        b = b.replace(
            "{{MIN_ARI}}",
            "N/A" if not math.isfinite(min_ari) else f"{min_ari:.4f}",
        )
        b = b.replace("{{MEAN_SPEARMAN}}", mean_sp_str)
        b = b.replace("{{ARI_COLOR}}", color)
        b = b.replace(
            "{{ARI_TABLE}}",
            _ari_table(ari_data["labels"], ari_data["values"], ari_threshold),
        )
        b = b.replace("{{HEATMAP_FILE}}", a.get("heatmap_png", ""))
        b = b.replace("{{FINDING_TEXT}}", finding)
        expanded += b

    return text[:match.start()] + expanded + text[match.end():]


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _render(template: str, data: Dict[str, Any]) -> str:
    """Fill template from JSON: conditionals -> placeholders -> asset blocks."""

    is_res = data.get("test") == _RES_TEST
    date_str = data.get("generated", "")[:10]
    model = data.get("model", "user_supplied")
    n_states = data.get("n_states") or "n/a"
    dr = data.get("date_range", {})
    date_range = f"{dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}"
    overall_ari = data.get("overall_mean_ari")
    overall_sp = data.get("overall_mean_spearman")
    partition_pass = data.get("partition_pass", False)
    ordering_pass = data.get("ordering_pass", False)
    ari_threshold = data.get("ari_threshold", ARI_THRESHOLD)
    sp_threshold = data.get("spearman_threshold", SPEARMAN_THRESHOLD)
    assets = data.get("assets", {})

    # Condition flags for template
    flags = {
        "PARTITION_PASS": partition_pass,
        "PARTITION_FAIL": not partition_pass,
        "ORDERING_PASS": ordering_pass,
        "ORDERING_FAIL": not ordering_pass,
        "PARTITION_FAIL_ORDERING_PASS": (not partition_pass) and ordering_pass,
        # Resolution invariance has no ordering / Spearman layer; the template
        # uses IS_REP to gate the ordering row and ordering methodology prose.
        "IS_RES": is_res,
        "IS_REP": not is_res,
    }

    # Step 1: Evaluate conditionals
    out = _eval_conditionals(template, flags)

    # Step 2: Factor set list (representation only; resolution has no factor sets).
    # Guard against an empty enumerate, which is a LaTeX error.
    item_lines = []
    if is_res:
        _first_asset = next(iter(assets.values()), {})
        freqs = ", ".join(_first_asset.get("frequencies", [])) or "5m, 15m, 1h, 1d"
        item_lines.append(f"  \\item Cross-frequency regime labels ({_tex(freqs)}).")
    else:
        factor_sets = data.get("factor_sets", [])
        if factor_sets:
            for fs in factor_sets:
                item_lines.append(f"  \\item \\textbf{{Set {fs['index']}:}} {_tex(fs['label'])}")
        else:
            for i, label in enumerate(data.get("spec_labels", [])):
                item_lines.append(f"  \\item \\textbf{{Set {i}:}} {_tex(str(label))}")
    if not item_lines:
        item_lines.append("  \\item (specifications listed in the per-asset detail).")
    factor_items = (
        "\\begin{enumerate}[leftmargin=2em]\n"
        + "\n".join(item_lines)
        + "\n\\end{enumerate}"
    )

    # Step 3: Dashboard table (pure data)
    dash_rows = ""
    for name, a in assets.items():
        a_ari = _asset_mean_ari(a, is_res)
        if math.isfinite(a_ari):
            ac = "mrvred" if a_ari < ari_threshold else "mrvgreen"
            a_ari_str = f"{a_ari:.3f}"
        else:
            # Insufficient data: neutral gray, never a green fail-open.
            ac = "mrvgray"
            a_ari_str = "N/A"
        if is_res:
            sp_cell = "N/A"
            sc = "mrvgray"
            os_ = "---"
        else:
            sp = a.get("mean_spearman", float("nan"))
            sc = "mrvgreen" if sp >= sp_threshold else "mrvred"
            sp_cell = f"\\textcolor{{{sc}}}{{\\textbf{{{sp:.3f}}}}}"
            os_ = "\\statuspass" if a.get("ordering_pass") else "\\statusfail"
        ps = "\\statuspass" if a.get("partition_pass") else "\\statusfail"
        n_obs_disp = a.get("n_obs", 0)
        dash_rows += (
            f"{_tex(name)} & "
            f"\\textcolor{{{ac}}}{{\\textbf{{{a_ari_str}}}}} & "
            f"{sp_cell} & "
            f"{n_obs_disp:,} & {ps} & {os_} \\\\\n"
        )
    oc = "mrvred" if not partition_pass else "mrvgreen"
    overall_ps = "\\statuspass" if partition_pass else "\\statusfail"
    _oa_str = f"{overall_ari:.3f}" if overall_ari is not None else "N/A"
    if is_res:
        _os_cell = "N/A"
        overall_os = "---"
    else:
        sc = "mrvgreen" if ordering_pass else "mrvred"
        _os_str = f"{overall_sp:.3f}" if overall_sp is not None else "N/A"
        _os_cell = f"\\textcolor{{{sc}}}{{\\textbf{{{_os_str}}}}}"
        overall_os = "\\statuspass" if ordering_pass else "\\statusfail"
    dash_rows += (
        f"\\midrule\n\\textbf{{Overall}} & "
        f"\\textcolor{{{oc}}}{{\\textbf{{{_oa_str}}}}} & "
        f"{_os_cell} & "
        f"--- & {overall_ps} & "
        f"{overall_os} \\\\\n"
    )
    dashboard_caption = (
        "Cross-frequency resolution stability dashboard."
        if is_res
        else "Two-layer representation stability dashboard."
    )
    dashboard = (
        "\\begin{table}[H]\n\\centering\n"
        "\\renewcommand{\\arraystretch}{1.4}\n"
        "\\begin{tabularx}{\\textwidth}{l c c c c c}\n\\toprule\n"
        "\\textbf{Asset} & \\textbf{Mean ARI} & \\textbf{Spearman} "
        "& \\textbf{Obs} & \\textbf{Partition} & \\textbf{Ordering} \\\\\n"
        f"& \\footnotesize($\\geq${ari_threshold}) & \\footnotesize($\\geq${sp_threshold}) "
        "& & & \\\\\n\\midrule\n"
        f"{dash_rows}"
        "\\bottomrule\n\\end{tabularx}\n"
        f"\\caption{{{dashboard_caption}}}\n\\end{{table}}"
    )

    # Step 4: Simple value replacements
    out = out.replace("{{DATE}}", date_str)
    out = out.replace("{{MODEL}}", _tex(str(model)))
    out = out.replace("{{N_STATES}}", str(n_states))
    out = out.replace("{{DATE_RANGE}}", date_range)
    out = out.replace("{{ASSET_LIST}}", ", ".join(_tex(a) for a in assets.keys()))
    out = out.replace("{{FACTOR_SETS}}", factor_items)
    out = out.replace("{{DASHBOARD_TABLE}}", dashboard)
    out = out.replace(
        "{{OVERALL_MEAN_ARI}}", f"{overall_ari:.3f}" if overall_ari is not None else "N/A"
    )
    out = out.replace(
        "{{OVERALL_MEAN_SPEARMAN}}",
        f"{overall_sp:.3f}" if overall_sp is not None else "N/A",
    )
    out = out.replace("{{ARI_THRESHOLD}}", str(ari_threshold))
    out = out.replace("{{SPEARMAN_THRESHOLD}}", str(sp_threshold))

    # Step 5: Expand per-asset blocks
    out = _expand_assets(out, data)

    return out


# ---------------------------------------------------------------------------
# Template resolution (works for pip-installed users via importlib.resources)
# ---------------------------------------------------------------------------

def _load_template_text(
    template: Optional[str | Path],
    v_cfg: Dict[str, Any],
) -> str:
    """Resolve the report template to its text.

    Resolution order:
    1. Explicit ``template`` argument (must exist on disk).
    2. ``validator.report_template`` config path, if it exists on disk.
    3. The template shipped inside the installed ``mrv`` package.
    """
    if template:
        tpl = Path(template)
        if not tpl.exists():
            raise FileNotFoundError(f"Template not found: {tpl}")
        return tpl.read_text(encoding="utf-8")

    cfg_tpl = v_cfg.get("report_template")
    if cfg_tpl:
        tpl = Path(cfg_tpl)
        if tpl.exists():
            return tpl.read_text(encoding="utf-8")

    # Packaged default -- shipped as mrv/templates/template.tex.
    return (
        resources.files("mrv")
        .joinpath("templates", "template.tex")
        .read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    json_path: str | Path,
    template: Optional[str | Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Generate a PDF report from validator JSON + LaTeX template.

    Works for both representation (Paper 1) and resolution (Paper 2) result
    JSON.  The ``.tex`` file is always written; the PDF is produced only when
    ``pdflatex`` is available on ``PATH`` (otherwise ``None`` is returned).

    Parameters
    ----------
    json_path : path to result.json
    template : path to .tex template (default: packaged specification-invariance
        template, or ``validator.report_template`` from ``cfg`` if it exists).
    cfg : mrv config dict
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    cfg = cfg or {}
    v_cfg = cfg.get("validator", {})
    template_text = _load_template_text(template, v_cfg)

    run_dir = json_path.parent
    tex_path = run_dir / f"{run_dir.name}.tex"

    output = _render(template_text, data)
    tex_path.write_text(output, encoding="utf-8")
    logger.info("LaTeX -> %s", tex_path)

    return _compile_pdf(tex_path)


def _compile_pdf(tex_path: Path) -> Optional[Path]:
    """Run pdflatex twice (for TOC) and clean up aux files."""
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        logger.warning("pdflatex not found; wrote .tex only, skipped PDF compile.")
        return None
    tex_path = tex_path.resolve()
    for n in (1, 2):
        logger.info("pdflatex pass %d: %s", n, tex_path.name)
        r = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            logger.error("pdflatex failed (pass %d):\n%s", n, r.stdout[-2000:])
            return None
    pdf = tex_path.with_suffix(".pdf")
    if pdf.exists():
        logger.info("PDF -> %s", pdf)
        for ext in (".aux", ".log", ".toc", ".out"):
            f = tex_path.with_suffix(ext)
            if f.exists():
                f.unlink()
        return pdf
    return None
