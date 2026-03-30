"""
mrv.validator.findings — Auto-generate SR 11-7 findings from validation results.

Severity levels:
- Critical:      overall_mean_ari < 0.1
- High:          overall_mean_ari < ARI_THRESHOLD
- Medium:        any min_pairwise_ari < ARI_THRESHOLD
- Low:           any fallback triggered
- Informational: all pass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from mrv.validator.metrics import ARI_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    """A single validation finding in SR 11-7 format."""
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
    counter = 1
    date_prefix = datetime.now().strftime("%Y")

    for asset_name, r in results.items():
        if validator_type == "rep":
            findings += _findings_rep(asset_name, r, date_prefix, counter)
        elif validator_type == "res":
            findings += _findings_res(asset_name, r, date_prefix, counter)
        counter += len(findings)

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


# ── Internal ─────────────────────────────────────────────────────────────────

def _findings_rep(asset: str, r: Dict, date_prefix: str, start: int) -> List[Finding]:
    """Generate findings for representation invariance."""
    findings = []
    mean_ari = r.get("mean_ari")
    min_ari = r.get("min_ari")

    sev = classify_severity(mean_ari, min_ari)
    if sev in ("Critical", "High", "Medium"):
        findings.append(Finding(
            id="",
            severity=sev,
            title=f"Representation invariance {sev.upper()} for {asset}",
            description=(
                f"Mean cross-representation ARI = {mean_ari:.3f} "
                f"(threshold: {ARI_THRESHOLD}). "
                f"Min pairwise ARI = {min_ari:.3f}. "
                f"Regime labels are sensitive to the choice of risk factors."
            ),
            evidence=f"See {asset}.png (ARI heatmap) and result.json.",
            recommendation=(
                "Review factor set selection. Consider reducing to the most stable "
                "factor combination or switching to a more robust model."
            ),
        ))

    sp = r.get("mean_spearman")
    if sp is not None and sp < 0.85 and mean_ari is not None and mean_ari >= ARI_THRESHOLD:
        findings.append(Finding(
            id="",
            severity="Medium",
            title=f"Ordering instability for {asset}",
            description=(
                f"Mean Spearman correlation = {sp:.3f} (threshold: 0.85). "
                f"While partition labels are stable, the risk ordering of states "
                f"differs across representations."
            ),
            evidence="See result.json ordering metrics.",
            recommendation="Validate that downstream risk decisions are robust to state reordering.",
        ))

    return findings


def _findings_res(asset: str, r: Dict, date_prefix: str, start: int) -> List[Finding]:
    """Generate findings for resolution invariance."""
    findings = []
    mean_ari = r.get("overall_mean_ari")
    fallback_flags = r.get("fallback_flags", {})
    any_fallback = any(fallback_flags.values()) if fallback_flags else False

    sev = classify_severity(mean_ari, any_fallback=any_fallback)
    if sev in ("Critical", "High", "Medium"):
        findings.append(Finding(
            id="",
            severity=sev,
            title=f"Resolution invariance {sev.upper()} for {asset}",
            description=(
                f"Mean cross-frequency ARI = {mean_ari:.3f} "
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

    # Fallback finding
    fallback_freqs = [f for f, v in fallback_flags.items() if v]
    if fallback_freqs:
        findings.append(Finding(
            id="",
            severity="Low",
            title=f"GMM fallback triggered for {asset}",
            description=(
                f"Percentile fallback was triggered at frequencies: "
                f"{', '.join(fallback_freqs)}. "
                f"GMM produced trivial partitions (< 1% or > 99% crisis), "
                f"replaced by 80th percentile thresholding."
            ),
            evidence=f"See {asset}_fallback_triggers.csv.",
            recommendation="Check data quality at these frequencies or increase GMM components.",
        ))

    # TOD seasonality finding
    tod = r.get("tod_crisis_distribution")
    if hasattr(tod, 'empty') and not tod.empty:
        for freq in tod["freq"].unique():
            sub = tod[tod["freq"] == freq]
            if sub["crisis_share"].max() - sub["crisis_share"].min() > 30:
                findings.append(Finding(
                    id="",
                    severity="Low",
                    title=f"TOD seasonality in {asset} at {freq}",
                    description=(
                        f"Crisis share varies from {sub['crisis_share'].min():.0f}% to "
                        f"{sub['crisis_share'].max():.0f}% across hours of day at {freq} "
                        f"frequency, suggesting regime labels are confounded by "
                        f"intraday volatility patterns."
                    ),
                    evidence=f"See {asset}_tod_crisis_distribution.csv.",
                    recommendation="Consider TOD-adjusted volatility for regime fitting.",
                ))
                break  # One finding per asset is enough

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
        logger.warning("Could not load overrides %s: %s", path, e)
        return {}
