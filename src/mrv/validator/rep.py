"""
mrv.validator.rep — Representation Invariance test (Paper 1).

One asset, multiple factor sets → fit regime model on each → compare via ARI.
Two-layer verdict: partition stability (ARI) + ordering stability (Spearman).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mrv.validator.base import BaseValidator
from mrv.validator.metrics import ari, ordering_consistency, ARI_THRESHOLD, SPEARMAN_THRESHOLD
from mrv.data.normalize import normalize
from mrv.data.factors import build_factors, resolve_name, log_returns, volatility
from mrv.models import fit as fit_model
import matplotlib
if matplotlib.get_backend().lower() == "qtagg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class RepValidator(BaseValidator):
    """Representation Invariance validator."""

    name = "rep"

    def validate(
        self,
        prices: Optional[Dict[str, pd.Series]] = None,
        labels: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """
        Run representation invariance test.

        Parameters
        ----------
        prices : dict, optional
            ``{asset: price_series}``. If None, loaded from config paths.
        labels : dict, optional
            ``{asset: {set_label: ndarray}}``. If None, computed internally.
        """
        rep_cfg = self.test_cfg
        factor_sets_raw = rep_cfg.get("factors", [])
        if len(factor_sets_raw) < 2:
            raise ValueError("Need >= 2 factor sets")

        model_name = rep_cfg.get("model", "gmm")
        n_states = rep_cfg.get("n_states", 3)
        set_labels = [", ".join(fs) for fs in factor_sets_raw]
        resolved_sets = [[resolve_name(f) for f in fs] for fs in factor_sets_raw]

        run_dir = self._make_run_dir()

        # Step 1: Load data (if not provided)
        if prices is None:
            from mrv.pipeline import load_data
            prices = load_data(self.cfg, self.name)

        if not prices:
            raise ValueError("No price data available")

        logger.info("=== Representation Invariance ===")
        logger.info("Assets: %s, Factor sets: %d, Model: %s (K=%d)",
                     list(prices.keys()), len(resolved_sets), model_name, n_states)

        all_results: Dict[str, Dict] = {}

        for asset_name, price in prices.items():
            logger.info("--- %s ---", asset_name)

            # Step 2+3: Factors → labels (if not provided)
            if labels and asset_name in labels:
                asset_labels = labels[asset_name]
            else:
                asset_labels = {}
                for i, (factors, label) in enumerate(zip(resolved_sets, set_labels)):
                    norm_df = normalize(build_factors(price, factors=factors, cfg=self.cfg), cfg=self.cfg)
                    valid = norm_df.dropna()
                    if len(valid) < 50:
                        continue
                    result = fit_model(valid, model=model_name, n_states=n_states)
                    if result is not None:
                        asset_labels[label] = result
                        logger.info("  [%d] %s: %d obs", i, label, len(result))

            # Step 4: Metrics
            risk_proxy = volatility(log_returns(price), window=20, annualize=False).dropna().values

            present = list(asset_labels.keys())
            n = len(present)
            ari_mat = pd.DataFrame(np.eye(n), index=present, columns=present)
            sp_mat = pd.DataFrame(np.eye(n), index=present, columns=present)

            for (la, a), (lb, b) in combinations(asset_labels.items(), 2):
                ari_val = ari(a, b)
                ari_mat.loc[la, lb] = ari_mat.loc[lb, la] = ari_val
                nc = min(len(a), len(b), len(risk_proxy))
                sp_val = ordering_consistency(a[:nc], b[:nc], risk_proxy[:nc])
                sp_mat.loc[la, lb] = sp_mat.loc[lb, la] = sp_val
                logger.info("  [%d] vs [%d]: ARI=%.3f Spearman=%.3f",
                            present.index(la), present.index(lb), ari_val, sp_val)

            offdiag = ari_mat.values[np.triu_indices(n, k=1)]
            sp_offdiag = sp_mat.values[np.triu_indices(n, k=1)]
            mean_ari = float(np.nanmean(offdiag)) if len(offdiag) else float("nan")
            min_ari = float(np.nanmin(offdiag)) if len(offdiag) else float("nan")
            mean_sp = float(np.nanmean(sp_offdiag)) if len(sp_offdiag) else float("nan")

            all_results[asset_name] = {
                "ari_matrix": ari_mat,
                "mean_ari": mean_ari, "min_ari": min_ari,
                "mean_spearman": mean_sp,
                "n_factor_sets": n, "n_obs": len(price),
            }

            _plot_ari_heatmap(ari_mat, asset_name, run_dir / f"{asset_name}.png")

        # Save JSON
        json_path = run_dir / "result.json"
        json_data = self._build_json(all_results, set_labels, model_name, n_states, rep_cfg)
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.json_path = json_path
        logger.info("JSON -> %s", json_path)

        # Save text summary
        _write_text_report(run_dir / "summary.txt", all_results, set_labels, model_name, n_states, rep_cfg)

        logger.info("=== Output: %s ===", run_dir)
        self.results = all_results
        return {"run_dir": str(run_dir), "json_path": str(json_path), "assets": all_results}

    def _build_json(self, results, set_labels, model, n_states, rep_cfg):
        all_ari = [r["mean_ari"] for r in results.values() if not np.isnan(r["mean_ari"])]
        all_sp = [r["mean_spearman"] for r in results.values() if not np.isnan(r["mean_spearman"])]
        overall_ari = float(np.mean(all_ari)) if all_ari else None
        overall_sp = float(np.mean(all_sp)) if all_sp else None

        assets_json = {}
        for name, r in results.items():
            ari_df = r["ari_matrix"]
            assets_json[name] = {
                "n_obs": r["n_obs"], "n_factor_sets": r["n_factor_sets"],
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
        return {
            "test": "representation_invariance",
            "generated": datetime.now().isoformat(),
            "model": model.upper(), "n_states": n_states,
            "date_range": {"start": rep_cfg.get("start"), "end": rep_cfg.get("end")},
            "factor_sets": [{"index": idx, "label": lbl} for idx, lbl in enumerate(set_labels)],
            "ari_threshold": ARI_THRESHOLD, "spearman_threshold": SPEARMAN_THRESHOLD,
            "overall_mean_ari": round(overall_ari, 6) if overall_ari else None,
            "overall_mean_spearman": round(overall_sp, 6) if overall_sp else None,
            "partition_pass": overall_ari is not None and overall_ari >= ARI_THRESHOLD,
            "ordering_pass": overall_sp is not None and overall_sp >= SPEARMAN_THRESHOLD,
            "assets": assets_json,
        }


# ---------------------------------------------------------------------------
# Helpers (kept as module-level functions, used by RepValidator)
# ---------------------------------------------------------------------------

def _plot_ari_heatmap(ari_matrix: pd.DataFrame, asset_name: str, out_path: Path) -> None:
    n = len(ari_matrix)
    fig, ax = plt.subplots(figsize=(5 + n * 0.5, 4 + n * 0.4))
    data = ari_matrix.values.astype(float)
    im = ax.imshow(data, vmin=-0.1, vmax=1.0, cmap="RdYlGn", aspect="auto")
    labels = [f"Set {i}" for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=10,
                    fontweight="bold" if i != j else "normal",
                    color="white" if v < 0.4 else "black")
    ax.set_title(f"{asset_name} — Cross-Representation ARI\n(threshold = {ARI_THRESHOLD})",
                 fontsize=12, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ARI", fontsize=10)
    cbar.ax.axhline(y=ARI_THRESHOLD, color="black", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _write_text_report(path, results, set_labels, model, n_states, rep_cfg):
    lines = [
        "=" * 60, "MRV Representation Invariance Report", "=" * 60, "",
        f"Date:    {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Model:   {model.upper()} (K={n_states})",
        f"Period:  {rep_cfg.get('start', '?')} -> {rep_cfg.get('end', '?')}", "",
    ]
    for asset, r in results.items():
        status_p = "PASS" if r["mean_ari"] >= ARI_THRESHOLD else "FAIL"
        status_o = "PASS" if r["mean_spearman"] >= SPEARMAN_THRESHOLD else "FAIL"
        lines += [
            f"--- {asset} ---",
            f"  Obs: {r['n_obs']}  Sets: {r['n_factor_sets']}",
            f"  Partition: ARI={r['mean_ari']:.3f} [{status_p}]",
            f"  Ordering:  Spearman={r['mean_spearman']:.3f} [{status_o}]", "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
