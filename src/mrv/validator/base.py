"""mrv.validator.base — Abstract base class for validators."""

from __future__ import annotations

import abc
import logging
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ImpactFn = Callable[[np.ndarray, pd.Series], float]


class BaseValidator(abc.ABC):
    """Base class for all validators."""

    name: str = ""

    def __init__(self, cfg: Dict[str, Any], impact_fn: Optional[ImpactFn] = None):
        self.cfg = cfg
        self.v_cfg = cfg.get("validator", {})
        self.test_cfg = self.v_cfg.get(self.name, {})
        self.results: Dict[str, Any] = {}
        self.run_dir: Optional[Path] = None
        self.json_path: Optional[Path] = None
        self.impact_fn = impact_fn

    @abc.abstractmethod
    def validate(self, prices=None, labels=None) -> Dict[str, Any]:
        """Run the validation test. Accepts optional pre-computed data."""

    def _make_run_dir(self) -> Path:
        base_dir = Path(self.v_cfg.get("report_dir", "reports"))
        report_name = self.v_cfg.get("report_name", "mrv_report_{date}")
        date_str = datetime.now().strftime("%Y%m%d")
        run_name = report_name.replace("{date}", date_str) + f"_{self.name}"
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        return run_dir

    def _compute_impact_matrix(
        self,
        labels_dict: Dict[str, np.ndarray],
        prices: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Compute business impact matrix using ``self.impact_fn``.

        For each label set, calls ``impact_fn(labels, prices)`` to get a scalar
        business metric (e.g. VaR).  Then builds a pairwise delta matrix.

        Returns None if ``impact_fn`` is not set.
        """
        if self.impact_fn is None:
            return None

        keys = list(labels_dict.keys())
        impacts: Dict[str, float] = {}
        for key in keys:
            try:
                impacts[key] = float(self.impact_fn(labels_dict[key], prices))
            except Exception as e:
                logger.warning("impact_fn failed for %s: %s", key, e)
                impacts[key] = float("nan")

        n = len(keys)
        delta = pd.DataFrame(np.zeros((n, n)), index=keys, columns=keys)
        for (ka, va), (kb, vb) in combinations(impacts.items(), 2):
            d = abs(va - vb)
            delta.loc[ka, kb] = delta.loc[kb, ka] = d

        offdiag = delta.values[np.triu_indices(n, k=1)]
        max_delta = float(np.nanmax(offdiag)) if len(offdiag) else 0.0
        worst_pair = None
        if len(offdiag) and max_delta > 0:
            idx = int(np.nanargmax(offdiag))
            pairs = [(keys[i], keys[j]) for i in range(n) for j in range(i + 1, n)]
            worst_pair = list(pairs[idx]) if idx < len(pairs) else None

        return {
            "impacts": impacts,
            "delta_matrix": delta,
            "max_delta": max_delta,
            "mean_delta": float(np.nanmean(offdiag)) if len(offdiag) else 0.0,
            "worst_pair": worst_pair,
        }
