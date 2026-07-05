"""
mrv.validator.rep -- Representation Invariance validator.

Validates whether regime labels remain stable across different feature
representations (Paper 1).  Users supply pre-computed labels from their
own models -- this module only measures agreement.

Usage::

    from mrv.validator.rep import RepValidator

    v = RepValidator()
    result = v.validate(labels={
        "SPY": {
            "vol+dd+var": labels_a,    # ndarray of regime labels
            "vol+var+cvar": labels_b,
            "skew+vol+var": labels_c,
        }
    })

    # With ordering consistency (requires a risk proxy per asset)
    result = v.validate(
        labels={"SPY": {"rep_a": la, "rep_b": lb}},
        risk_proxy={"SPY": volatility_array},
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mrv.validator.base import BaseValidator
from mrv.validator.metrics import ARI_THRESHOLD, SPEARMAN_THRESHOLD, ari, ordering_consistency

logger = logging.getLogger(__name__)


class RepValidator(BaseValidator):
    """Representation Invariance validator.

    Measures whether regime labels agree across different feature
    representations.  Users provide their own labels -- no model fitting
    is performed.
    """

    name = "rep"

    def validate(  # type: ignore[override]
        self,
        labels: Dict[str, Dict[str, np.ndarray]],
        risk_proxy: Optional[Dict[str, np.ndarray]] = None,
        prices: Optional[Dict[str, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """
        Run representation invariance test on user-provided labels.

        Parameters
        ----------
        labels : dict
            ``{asset_name: {spec_label: ndarray}}``.
            Each asset must have >= 2 specifications.
            Each ndarray is a 1-D integer array of regime labels.
        risk_proxy : dict, optional
            ``{asset_name: ndarray}``.  A 1-D risk measure (e.g. rolling
            volatility) used for ordering consistency (Spearman).
            If not provided, ordering consistency is skipped.
        prices : dict, optional
            ``{asset_name: price_series}``.  Only used for business
            impact computation (if ``impact_fn`` was set).
        """
        if not labels:
            raise ValueError("labels dict is empty -- provide at least one asset")

        for asset, specs in labels.items():
            if len(specs) < 2:
                raise ValueError(
                    f"Asset '{asset}' has {len(specs)} specification(s), need >= 2"
                )

        rep_cfg = self.test_cfg
        run_dir = self._make_run_dir()
        set_labels_all: set[str] = set()
        for specs in labels.values():
            set_labels_all.update(specs.keys())
        set_labels = sorted(set_labels_all)

        logger.info("=== Representation Invariance ===")
        logger.info("Assets: %s, Specs per asset: %s",
                     list(labels.keys()),
                     {a: len(s) for a, s in labels.items()})

        all_results: Dict[str, Dict] = {}

        for asset_name, asset_labels in labels.items():
            logger.info("--- %s ---", asset_name)

            present = list(asset_labels.keys())
            n = len(present)
            present_idx = {name: i for i, name in enumerate(present)}

            # ARI matrix
            ari_mat = pd.DataFrame(np.eye(n), index=present, columns=present)
            sp_mat = pd.DataFrame(np.eye(n), index=present, columns=present)

            # Risk proxy for ordering consistency
            rp = None
            if risk_proxy and asset_name in risk_proxy:
                rp = risk_proxy[asset_name]

            for (la, a), (lb, b) in combinations(asset_labels.items(), 2):
                ari_val = ari(a, b)
                ari_mat.loc[la, lb] = ari_mat.loc[lb, la] = ari_val

                if rp is not None:
                    nc = min(len(a), len(b), len(rp))
                    sp_val = ordering_consistency(a[:nc], b[:nc], rp[:nc])
                else:
                    sp_val = float("nan")
                sp_mat.loc[la, lb] = sp_mat.loc[lb, la] = sp_val

                logger.info("  [%d] vs [%d]: ARI=%.3f Spearman=%.3f",
                            present_idx[la], present_idx[lb], ari_val, sp_val)

            offdiag = ari_mat.values[np.triu_indices(n, k=1)]
            sp_offdiag = sp_mat.values[np.triu_indices(n, k=1)]
            mean_ari = float(np.nanmean(offdiag)) if len(offdiag) else float("nan")
            min_ari = float(np.nanmin(offdiag)) if len(offdiag) else float("nan")
            mean_sp = float(np.nanmean(sp_offdiag)) if len(sp_offdiag) else float("nan")

            first_arr = next(iter(asset_labels.values()))
            n_obs = int(len(first_arr)) if first_arr is not None else 0
            asset_result = {
                "ari_matrix": ari_mat,
                "mean_ari": mean_ari, "min_ari": min_ari,
                "mean_spearman": mean_sp,
                "n_specs": n,
                "n_obs": n_obs,
            }

            # Attribution (if enabled and >= 3 specs)
            if rep_cfg.get("attribution", False) and n >= 3:
                from mrv.validator.attribution import loo_factor_attribution
                attr = loo_factor_attribution(asset_labels, mean_ari)
                asset_result["attribution"] = attr
                logger.info("  Attribution: worst=%s delta=%s",
                            attr["worst_contributor"],
                            attr["scores"].get(attr["worst_contributor"], "N/A"))

            # Business impact (if impact_fn and prices provided)
            if prices and asset_name in prices:
                impact = self._compute_impact_matrix(asset_labels, prices[asset_name])
                if impact is not None:
                    asset_result["impact"] = impact
                    impact["delta_matrix"].to_csv(run_dir / f"{asset_name}_impact_matrix.csv")
                    logger.info("  Impact: max_delta=%.4f worst_pair=%s",
                                impact["max_delta"], impact["worst_pair"])

            all_results[asset_name] = asset_result

            # Plot
            try:
                from mrv.validator.plots import plot_ari_heatmap
                plot_ari_heatmap(ari_mat, asset_name, run_dir / f"{asset_name}.png",
                               title_prefix="Cross-Representation")
            except ImportError:
                logger.debug("matplotlib not available, skipping heatmap")

        # Save JSON
        json_path = run_dir / "result.json"
        json_data = self._build_json(all_results, set_labels, rep_cfg)
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.json_path = json_path
        logger.info("JSON -> %s", json_path)

        # Save text summary
        _write_text_report(run_dir / "summary.txt", all_results, rep_cfg)

        logger.info("=== Output: %s ===", run_dir)
        self.results = all_results
        return {"run_dir": str(run_dir), "json_path": str(json_path), "assets": all_results}

    def _build_json(self, results, set_labels, rep_cfg):
        all_ari = [r["mean_ari"] for r in results.values() if not np.isnan(r["mean_ari"])]
        all_sp = [r["mean_spearman"] for r in results.values() if not np.isnan(r["mean_spearman"])]
        overall_ari = float(np.mean(all_ari)) if all_ari else None
        overall_sp = float(np.mean(all_sp)) if all_sp else None

        assets_json = {}
        for name, r in results.items():
            ari_df = r["ari_matrix"]
            assets_json[name] = {
                "n_specs": r["n_specs"],
                "n_factor_sets": r["n_specs"],  # alias: n_factor_sets == n_specs for report.py
                "n_obs": r.get("n_obs", 0),
                "mean_ari": round(r["mean_ari"], 6), "min_ari": round(r["min_ari"], 6),
                "mean_spearman": round(r["mean_spearman"], 6),
                "partition_pass": r["mean_ari"] >= ARI_THRESHOLD,
                "ordering_pass": r["mean_spearman"] >= SPEARMAN_THRESHOLD,
                "ari_matrix": {
                    "labels": list(ari_df.columns),
                    "values": [[round(v, 6) for v in row] for row in ari_df.values.tolist()],
                },
                "heatmap_png": f"{name}.png",
            }
            if "impact" in r:
                imp = r["impact"]
                assets_json[name]["impact"] = {
                    "values": {k: round(v, 6) for k, v in imp["impacts"].items()},
                    "max_delta": round(imp["max_delta"], 6),
                    "mean_delta": round(imp["mean_delta"], 6),
                    "worst_pair": imp["worst_pair"],
                }
        return {
            "test": "representation_invariance",
            "generated": datetime.now().isoformat(),
            "model": rep_cfg.get("model", "user_supplied"),
            "n_states": rep_cfg.get("n_states", None),
            "date_range": {"start": rep_cfg.get("start"), "end": rep_cfg.get("end")},
            "spec_labels": set_labels,
            "ari_threshold": ARI_THRESHOLD, "spearman_threshold": SPEARMAN_THRESHOLD,
            "overall_mean_ari": round(overall_ari, 6) if overall_ari is not None else None,
            "overall_mean_spearman": round(overall_sp, 6) if overall_sp is not None else None,
            "partition_pass": overall_ari is not None and overall_ari >= ARI_THRESHOLD,
            "ordering_pass": overall_sp is not None and overall_sp >= SPEARMAN_THRESHOLD,
            "assets": assets_json,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_text_report(path, results, rep_cfg):
    lines = [
        "=" * 60, "MRV Representation Invariance Report", "=" * 60, "",
        f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Period:  {rep_cfg.get('start', '?')} -> {rep_cfg.get('end', '?')}", "",
    ]
    for asset, r in results.items():
        status_p = "PASS" if r["mean_ari"] >= ARI_THRESHOLD else "FAIL"
        status_o = "PASS" if r["mean_spearman"] >= SPEARMAN_THRESHOLD else "FAIL"
        lines += [
            f"--- {asset} ---",
            f"  Specs: {r['n_specs']}",
            f"  Partition: ARI={r['mean_ari']:.3f} [{status_p}]",
            f"  Ordering:  Spearman={r['mean_spearman']:.3f} [{status_o}]", "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
