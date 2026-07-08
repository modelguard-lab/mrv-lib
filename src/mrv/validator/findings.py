"""
mrv.validator.findings -- Auto-generate SR 26-2 / OCC Bulletin 2026-13 findings
from validation results. SR 11-7 (Federal Reserve, 2011) was superseded by
SR 26-2 / OCC Bulletin 2026-13 on 2026-04-17; this module targets the new standard.

Severity levels:
- Critical:      overall_mean_ari < 0.1
- High:          overall_mean_ari < ARI_THRESHOLD
- Medium:        any min_pairwise_ari < ARI_THRESHOLD
- Low:           any fallback triggered
- Informational: all pass
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml  # type: ignore[import-untyped]

from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD

logger = logging.getLogger(__name__)

__all__ = [
    "Finding",
    "classify_severity",
    "generate_findings",
    "overall_risk_rating",
    "findings_summary",
]


@dataclass
class Finding:
    """A single validation finding in SR 26-2 / OCC Bulletin 2026-13 format."""
    id: str
    severity: str
    title: str
    description: str
    evidence: str = ""
    recommendation: str = ""
    remediation_owner: str = ""
    deadline: str = ""
    management_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_severity(
    overall_mean_ari: Optional[float],
    min_pairwise_ari: Optional[float] = None,
    any_fallback: bool = False,
) -> str:
    """Classify finding severity based on empirical rules."""
    if overall_mean_ari is None or np.isnan(overall_mean_ari):
        return "High"
    if overall_mean_ari < 0.1:
        return "Critical"
    if overall_mean_ari < ARI_THRESHOLD:
        return "High"
    if min_pairwise_ari is not None and min_pairwise_ari < ARI_THRESHOLD:
        return "Medium"
    if any_fallback:
        return "Low"
    return "Informational"


def generate_findings(
    results: Dict[str, Any],
    validator_type: str,
    overrides_path: Optional[Path] = None,
) -> List[Finding]:
    """Auto-generate findings from validation results.

    Parameters
    ----------
    results : dict
        The ``assets`` dict from a validator run (``{asset_name: result_dict}``).
    validator_type : str
        ``"rep"`` or ``"res"``.
    overrides_path : Path, optional
        YAML file with user-provided overrides (owner, deadline, response).
    """
    overrides = _load_overrides(overrides_path) if overrides_path else {}
    findings: List[Finding] = []
    date_prefix = datetime.now().strftime("%Y")

    for asset_name, r in results.items():
        if validator_type == "rep":
            findings += _findings_rep(asset_name, r, date_prefix)
        elif validator_type == "res":
            findings += _findings_res(asset_name, r, date_prefix)

    # Re-number sequentially
    for i, f in enumerate(findings, 1):
        f.id = f"MRV-{date_prefix}-{i:03d}"

    # Apply user overrides
    for f in findings:
        if f.id in overrides:
            ov = overrides[f.id]
            f.remediation_owner = ov.get("remediation_owner", f.remediation_owner)
            f.deadline = ov.get("deadline", f.deadline)
            f.management_response = ov.get("management_response", f.management_response)
            if "severity" in ov:
                f.severity = ov["severity"]

    return findings


def overall_risk_rating(findings: List[Finding]) -> str:
    """Derive overall model risk rating from findings."""
    severities = [f.severity for f in findings]
    if "Critical" in severities:
        return "High"
    if "High" in severities:
        return "High"
    if "Medium" in severities:
        return "Medium"
    return "Low"


def findings_summary(findings: List[Finding]) -> Dict[str, int]:
    """Count findings by severity."""
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Informational": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return counts


# -- Internal -----------------------------------------------------------------

def _findings_rep(asset: str, r: Dict, date_prefix: str) -> List[Finding]:
    """Generate findings for representation invariance."""
    findings = []
    mean_ari = r.get("mean_ari")
    min_ari = r.get("min_ari")

    sev = classify_severity(mean_ari, min_ari)
    if sev in ("Critical", "High", "Medium"):
        mean_ari_str = f"{mean_ari:.3f}" if mean_ari is not None else "N/A"
        min_ari_str = f"{min_ari:.3f}" if min_ari is not None else "N/A"
        findings.append(Finding(
            id="",
            severity=sev,
            title=f"Representation invariance {sev.upper()} for {asset}",
            description=(
                f"Mean cross-representation ARI = {mean_ari_str} "
                f"(threshold: {ARI_THRESHOLD}). "
                f"Min pairwise ARI = {min_ari_str}. "
                f"Regime labels are sensitive to the choice of risk factors."
            ),
            evidence=f"See {asset}.png (ARI heatmap) and result.json.",
            recommendation=(
                "Review factor set selection. Consider reducing to the most stable "
                "factor combination or switching to a more robust model."
            ),
        ))

    sp = r.get("mean_spearman")
    if (sp is not None and sp < SPEARMAN_THRESHOLD
            and mean_ari is not None and mean_ari >= ARI_THRESHOLD):
        findings.append(Finding(
            id="",
            severity="Medium",
            title=f"Ordering instability for {asset}",
            description=(
                f"Mean Spearman correlation = {sp:.3f} (threshold: {SPEARMAN_THRESHOLD}). "
                f"While partition labels are stable, the risk ordering of states "
                f"differs across representations."
            ),
            evidence="See result.json ordering metrics.",
            recommendation=(
                "Validate that downstream risk decisions are robust to state "
                "reordering."
            ),
        ))

    return findings


def _findings_res(asset: str, r: Dict, date_prefix: str) -> List[Finding]:
    """Generate findings for resolution invariance."""
    findings = []
    mean_ari = r.get("overall_mean_ari")

    sev = classify_severity(mean_ari)
    if sev in ("Critical", "High", "Medium"):
        mean_ari_str = f"{mean_ari:.3f}" if mean_ari is not None else "N/A"
        findings.append(Finding(
            id="",
            severity=sev,
            title=f"Resolution invariance {sev.upper()} for {asset}",
            description=(
                f"Mean cross-frequency ARI = {mean_ari_str} "
                f"(threshold: {ARI_THRESHOLD}). "
                f"Regime labels change significantly across time frequencies "
                f"(5m/15m/1h/1d)."
            ),
            evidence=f"See {asset}_ari_heatmap.png and {asset}_timeline.png.",
            recommendation=(
                "Restrict regime model to a single canonical frequency, or "
                "use frequency-ensemble consensus labels."
            ),
        ))

    return findings


def _load_overrides(path: Path) -> Dict[str, Dict]:
    """Load user-provided finding overrides from YAML."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception as e:
        line_info = ""
        if (isinstance(e, yaml.YAMLError) and hasattr(e, "problem_mark")
                and e.problem_mark is not None):
            line_info = f" (line {e.problem_mark.line + 1}, column {e.problem_mark.column + 1})"
        logger.error("Could not load overrides %s%s: %s", path, line_info, e)
        raise
