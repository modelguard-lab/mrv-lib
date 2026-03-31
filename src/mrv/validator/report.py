"""
mrv.validator.report — Fill LaTeX template with JSON data and compile to PDF.

All text lives in the template. This module only:
1. Replaces ``{{KEY}}`` placeholders with data values
2. Evaluates ``%% IF_xxx`` / ``%% ELSE`` / ``%% ELIF_xxx`` / ``%% ENDIF`` conditionals
3. Expands ``%% BEGIN_ASSET`` / ``%% END_ASSET`` blocks per asset
4. Compiles .tex → .pdf via pdflatex
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD

logger = logging.getLogger(__name__)


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

_COND_RE = re.compile(
    r"%% (IF_\w+|ELIF_\w+|ELSE|ENDIF)\s*\n",
)


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
            stack.pop()
            active = all(s[0] for s in stack) if stack else True
        else:
            if active:
                result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Asset block expansion
# ---------------------------------------------------------------------------

_ASSET_RE = re.compile(r"%% BEGIN_ASSET\s*\n(.*?)%% END_ASSET", re.DOTALL)


def _expand_assets(text: str, data: Dict[str, Any]) -> str:
    match = _ASSET_RE.search(text)
    if not match:
        return text

    block_tpl = match.group(1)
    assets = data.get("assets", {})
    ari_threshold = data.get("ari_threshold", ARI_THRESHOLD)
    sp_threshold = data.get("spearman_threshold", SPEARMAN_THRESHOLD)
    expanded = ""

    for name, a in assets.items():
        ari_data = a["ari_matrix"]
        mean_ari = a["mean_ari"]
        min_ari = a["min_ari"]
        mean_sp = a.get("mean_spearman", float("nan"))
        color = "mrvred" if mean_ari < ari_threshold else "mrvgreen"

        # Per-asset finding text
        if mean_ari < ari_threshold and mean_sp >= sp_threshold:
            finding = (
                f"\\textcolor{{mrvred}}{{\\textbf{{Partition: FAIL}}}} --- "
                f"ARI = {mean_ari:.3f}. "
                f"\\textcolor{{mrvgreen}}{{\\textbf{{Ordering: PASS}}}} --- "
                f"Spearman = {mean_sp:.3f}. "
                f"Categorical labels are unstable for {_tex(name)}, "
                "but risk ranking is preserved."
            )
        elif mean_ari < ari_threshold:
            finding = (
                f"\\textcolor{{mrvred}}{{\\textbf{{Partition: FAIL}}}} --- "
                f"ARI = {mean_ari:.3f}. "
                f"\\textcolor{{mrvred}}{{\\textbf{{Ordering: FAIL}}}} --- "
                f"Spearman = {mean_sp:.3f}. "
                f"Both labels and risk ordering are unstable for {_tex(name)}."
            )
        else:
            finding = (
                f"\\textcolor{{mrvgreen}}{{\\textbf{{Pass:}}}} "
                f"ARI = {mean_ari:.3f}, Spearman = {mean_sp:.3f} --- "
                f"acceptable stability for {_tex(name)}."
            )

        b = block_tpl
        b = b.replace("{{ASSET_NAME}}", _tex(name))
        b = b.replace("{{N_OBS}}", f"{a['n_obs']:,}")
        b = b.replace("{{N_FACTOR_SETS}}", str(a["n_factor_sets"]))
        b = b.replace("{{MEAN_ARI}}", f"{mean_ari:.4f}")
        b = b.replace("{{MIN_ARI}}", f"{min_ari:.4f}")
        b = b.replace("{{MEAN_SPEARMAN}}", f"{mean_sp:.4f}")
        b = b.replace("{{ARI_COLOR}}", color)
        b = b.replace("{{ARI_TABLE}}", _ari_table(ari_data["labels"], ari_data["values"], ari_threshold))
        b = b.replace("{{HEATMAP_FILE}}", a.get("heatmap_png", ""))
        b = b.replace("{{FINDING_TEXT}}", finding)
        expanded += b

    return text[:match.start()] + expanded + text[match.end():]


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _render(template: str, data: Dict[str, Any]) -> str:
    """Fill template from JSON: conditionals → placeholders → asset blocks."""

    date_str = data.get("generated", "")[:10]
    model = data.get("model", "GMM")
    n_states = data.get("n_states", 3)
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
    }

    # Step 1: Evaluate conditionals
    out = _eval_conditionals(template, flags)

    # Step 2: Factor set list
    factor_items = "\\begin{enumerate}[leftmargin=2em]\n"
    for fs in data.get("factor_sets", []):
        factor_items += f"  \\item \\textbf{{Set {fs['index']}:}} {_tex(fs['label'])}\n"
    factor_items += "\\end{enumerate}"

    # Step 3: Dashboard table (pure data)
    dash_rows = ""
    for name, a in assets.items():
        ac = "mrvred" if a["mean_ari"] < ari_threshold else "mrvgreen"
        sp = a.get("mean_spearman", float("nan"))
        sc = "mrvgreen" if sp >= sp_threshold else "mrvred"
        ps = "\\statuspass" if a.get("partition_pass") else "\\statusfail"
        os_ = "\\statuspass" if a.get("ordering_pass") else "\\statusfail"
        dash_rows += (
            f"{_tex(name)} & "
            f"\\textcolor{{{ac}}}{{\\textbf{{{a['mean_ari']:.3f}}}}} & "
            f"\\textcolor{{{sc}}}{{\\textbf{{{sp:.3f}}}}} & "
            f"{a['n_obs']:,} & {ps} & {os_} \\\\\n"
        )
    oc = "mrvred" if not partition_pass else "mrvgreen"
    sc = "mrvgreen" if ordering_pass else "mrvred"
    overall_ps = "\\statuspass" if partition_pass else "\\statusfail"
    overall_os = "\\statuspass" if ordering_pass else "\\statusfail"
    dash_rows += (
        f"\\midrule\n\\textbf{{Overall}} & "
        f"\\textcolor{{{oc}}}{{\\textbf{{{overall_ari:.3f}}}}} & "
        f"\\textcolor{{{sc}}}{{\\textbf{{{overall_sp:.3f}}}}} & "
        f"--- & {overall_ps} & "
        f"{overall_os} \\\\\n"
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
        "\\caption{Two-layer representation stability dashboard.}\n\\end{table}"
    )

    # Step 4: Simple value replacements
    out = out.replace("{{DATE}}", date_str)
    out = out.replace("{{MODEL}}", model)
    out = out.replace("{{N_STATES}}", str(n_states))
    out = out.replace("{{DATE_RANGE}}", date_range)
    out = out.replace("{{ASSET_LIST}}", ", ".join(_tex(a) for a in assets.keys()))
    out = out.replace("{{FACTOR_SETS}}", factor_items)
    out = out.replace("{{DASHBOARD_TABLE}}", dashboard)
    out = out.replace("{{OVERALL_MEAN_ARI}}", f"{overall_ari:.3f}" if overall_ari else "N/A")
    out = out.replace("{{OVERALL_MEAN_SPEARMAN}}", f"{overall_sp:.3f}" if overall_sp else "N/A")
    out = out.replace("{{ARI_THRESHOLD}}", str(ari_threshold))
    out = out.replace("{{SPEARMAN_THRESHOLD}}", str(sp_threshold))

    # Step 5: Expand per-asset blocks
    out = _expand_assets(out, data)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    json_path: str | Path,
    template: Optional[str | Path] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Generate PDF report from validator JSON + LaTeX template.

    Parameters
    ----------
    json_path : path to result.json
    template : path to .tex template (default: from cfg or templates/template.tex)
    cfg : mrv config dict
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    cfg = cfg or {}
    v_cfg = cfg.get("validator", {})
    tpl = Path(template) if template else Path(v_cfg.get("report_template", "templates/template.tex"))
    if not tpl.exists():
        raise FileNotFoundError(f"Template not found: {tpl}")

    run_dir = json_path.parent
    tex_path = run_dir / f"{run_dir.name}.tex"

    output = _render(tpl.read_text(encoding="utf-8"), data)
    tex_path.write_text(output, encoding="utf-8")
    logger.info("LaTeX -> %s", tex_path)

    return _compile_pdf(tex_path)


def generate_sr11_7_report(
    json_path: "str | Path",
    template: "Optional[str | Path]" = None,
    cfg: Optional[Dict[str, Any]] = None,
    overrides: "Optional[str | Path]" = None,
) -> Optional[Path]:
    """Generate SR 11-7 compliant PDF from validator JSON.

    Parameters
    ----------
    json_path : path to result.json
    template : path to SR 11-7 .tex template
    cfg : mrv config dict
    overrides : path to findings_override.yaml
    """
    from mrv.validator.findings import (
        generate_findings, overall_risk_rating, findings_summary,
    )

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    cfg = cfg or {}
    v_cfg = cfg.get("validator", {})
    tpl_path = (
        Path(template) if template
        else Path(v_cfg.get("sr11_7_template", "templates/sr11_7_template.tex"))
    )
    if not tpl_path.exists():
        raise FileNotFoundError(f"SR 11-7 template not found: {tpl_path}")

    overrides_path = Path(overrides) if overrides else None
    if overrides_path and not overrides_path.exists():
        overrides_path = None

    # Determine validator type
    test_type = data.get("test", "")
    validator_type = "res" if "resolution" in test_type else "rep"

    # Generate findings
    assets = data.get("assets", {})
    findings = generate_findings(assets, validator_type, overrides_path)
    risk_rating = overall_risk_rating(findings)
    summary = findings_summary(findings)

    # Build template
    tpl = tpl_path.read_text(encoding="utf-8")

    # Condition flags
    flags = {
        "RATING_HIGH": risk_rating == "High",
        "RATING_MEDIUM": risk_rating == "Medium",
        "RATING_LOW": risk_rating == "Low",
        "IS_REP": validator_type == "rep",
        "IS_RES": validator_type == "res",
        "HAS_IMPACT": any("impact" in a for a in assets.values()),
        "HAS_HMM": any("hmm_overall_mean_ari" in a for a in assets.values()),
        "HAS_PERMUTATION": any(a.get("pvalue_perm") is not None for a in assets.values()),
        "HAS_ATTRIBUTION": any("attribution" in a for a in assets.values()),
    }
    out = _eval_conditionals(tpl, flags)

    # Expand finding blocks
    finding_match = re.search(r"%% BEGIN_FINDING\s*\n(.*?)%% END_FINDING", out, re.DOTALL)
    if finding_match and findings:
        block_tpl = finding_match.group(1)
        expanded = ""
        for f in findings:
            b = block_tpl
            b = b.replace("{{FINDING_ID}}", _tex(f.id))
            b = b.replace("{{FINDING_TITLE}}", _tex(f.title))
            b = b.replace("{{FINDING_SEVERITY}}", f.severity)
            b = b.replace("{{FINDING_SEVERITY_LC}}", f.severity.lower())
            b = b.replace("{{FINDING_DESCRIPTION}}", _tex(f.description))
            b = b.replace("{{FINDING_EVIDENCE}}", _tex(f.evidence))
            b = b.replace("{{FINDING_RECOMMENDATION}}", _tex(f.recommendation))
            b = b.replace("{{FINDING_OWNER}}", _tex(f.remediation_owner))
            b = b.replace("{{FINDING_DEADLINE}}", _tex(f.deadline))
            b = b.replace("{{FINDING_RESPONSE}}", _tex(f.management_response))
            has_rem = bool(f.remediation_owner or f.deadline)
            b = _eval_conditionals(b, {"HAS_REMEDIATION": has_rem})
            expanded += b
        out = out[:finding_match.start()] + expanded + out[finding_match.end():]
    elif finding_match:
        out = out[:finding_match.start()] + "No findings.\n" + out[finding_match.end():]

    # Expand asset blocks (appendix)
    asset_match = re.search(r"%% BEGIN_ASSET\s*\n(.*?)%% END_ASSET", out, re.DOTALL)
    if asset_match:
        block_tpl = asset_match.group(1)
        expanded = ""
        for name, a in assets.items():
            b = block_tpl
            b = b.replace("{{ASSET_NAME}}", _tex(name))
            ari_data = a.get("ari_matrix", {})
            if ari_data:
                b = b.replace("{{ARI_TABLE}}", _ari_table(
                    ari_data.get("labels", []), ari_data.get("values", []),
                    data.get("ari_threshold", ARI_THRESHOLD)))
            else:
                b = b.replace("{{ARI_TABLE}}", "No ARI data.")
            b = b.replace("{{HEATMAP_PNG}}", a.get("heatmap_png", ""))
            b = b.replace("{{TIMELINE_PNG}}", a.get("timeline_png", ""))
            asset_flags = {
                "ASSET_HAS_HEATMAP": bool(a.get("heatmap_png")),
                "ASSET_HAS_TIMELINE": bool(a.get("timeline_png")),
            }
            b = _eval_conditionals(b, asset_flags)
            expanded += b
        out = out[:asset_match.start()] + expanded + out[asset_match.end():]

    # Global placeholders
    dr = data.get("date_range", {})
    date_range = f"{dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}"
    model = data.get("model", "GMM")
    n_states = data.get("n_components", data.get("n_states", 2))

    replacements = {
        "{{MODEL_NAME}}": f"{model} Regime Model (K={n_states})",
        "{{MODEL_OWNER}}": cfg.get("model_owner", "---"),
        "{{VALIDATOR_NAME}}": "mrv-lib",
        "{{DATE}}": data.get("generated", "")[:10],
        "{{TEST_TYPE}}": "Resolution Invariance" if validator_type == "res" else "Representation Invariance",
        "{{DATE_RANGE}}": date_range,
        "{{VERSION}}": "0.2.0",
        "{{MODEL}}": model,
        "{{N_STATES}}": str(n_states),
        "{{OVERALL_RISK_RATING}}": risk_rating,
        "{{N_CRITICAL}}": str(summary.get("Critical", 0)),
        "{{N_HIGH}}": str(summary.get("High", 0)),
        "{{N_MEDIUM}}": str(summary.get("Medium", 0)),
        "{{N_LOW}}": str(summary.get("Low", 0)),
        "{{N_INFO}}": str(summary.get("Informational", 0)),
        "{{N_TOTAL}}": str(sum(summary.values())),
        "{{MODEL_DESCRIPTION}}": cfg.get("model_description", "Regime classification model for market risk monitoring."),
        "{{PERTURBATION_TYPE}}": "time frequency" if validator_type == "res" else "risk factor representation",
        "{{INPUT_DESCRIPTION}}": "OHLCV market data across multiple frequencies" if validator_type == "res" else "Price series with multiple risk factor sets",
        "{{TEST_DESCRIPTION}}": (
            "Cross-frequency ARI matrix comparing regime labels across 5m/15m/1h/1d"
            if validator_type == "res"
            else "Cross-representation ARI matrix comparing regime labels across factor sets"
        ),
        "{{ASSET_LIST}}": ", ".join(_tex(a) for a in assets.keys()),
        "{{ARI_THRESHOLD}}": str(data.get("ari_threshold", ARI_THRESHOLD)),
        "{{SPEARMAN_THRESHOLD}}": str(SPEARMAN_THRESHOLD),
        "{{GENERATED}}": data.get("generated", ""),
    }

    # Optional fields
    max_impact = 0.0
    worst_impact_pair = "N/A"
    for a in assets.values():
        imp = a.get("impact", {})
        if isinstance(imp, dict) and imp.get("max_delta", 0) > max_impact:
            max_impact = imp["max_delta"]
            worst_impact_pair = str(imp.get("worst_pair", "N/A"))
    replacements["{{MAX_IMPACT_DELTA}}"] = f"{max_impact:.4f}"
    replacements["{{WORST_IMPACT_PAIR}}"] = _tex(worst_impact_pair)

    # HMM / permutation
    hmm_aris = [a.get("hmm_overall_mean_ari") for a in assets.values() if a.get("hmm_overall_mean_ari") is not None]
    replacements["{{HMM_MEAN_ARI}}"] = f"{np.mean(hmm_aris):.3f}" if hmm_aris else "N/A"
    perm_pvals = [a.get("pvalue_perm") for a in assets.values() if a.get("pvalue_perm") is not None]
    replacements["{{PERM_PVALUE}}"] = f"{np.mean(perm_pvals):.4f}" if perm_pvals else "N/A"
    replacements["{{PERM_CI_LOW}}"] = "N/A"
    replacements["{{PERM_CI_HIGH}}"] = "N/A"
    for a in assets.values():
        ci = a.get("null_ci")
        if ci:
            replacements["{{PERM_CI_LOW}}"] = f"{ci[0]:.4f}"
            replacements["{{PERM_CI_HIGH}}"] = f"{ci[1]:.4f}"
            break

    # Attribution summary
    attr_parts = []
    for name, a in assets.items():
        attr = a.get("attribution", {})
        if isinstance(attr, dict) and attr.get("summary"):
            attr_parts.append(f"{name}: {attr['summary']}")
    replacements["{{ATTRIBUTION_SUMMARY}}"] = _tex(" ".join(attr_parts)) if attr_parts else "No attribution data."

    for key, val in replacements.items():
        out = out.replace(key, val)

    # Write and compile
    run_dir = json_path.parent
    tex_path = run_dir / f"{run_dir.name}_sr11_7.tex"
    out_text = out
    # Need numpy for HMM mean
    tex_path.write_text(out_text, encoding="utf-8")
    logger.info("SR 11-7 LaTeX -> %s", tex_path)

    return _compile_pdf(tex_path)


def _compile_pdf(tex_path: Path) -> Optional[Path]:
    """Run pdflatex twice (for TOC) and clean up aux files."""
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        logger.error("pdflatex not found.")
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
