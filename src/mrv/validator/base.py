"""mrv.validator.base — Abstract base class for validators."""

from __future__ import annotations

import abc
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseValidator(abc.ABC):
    """Base class for all validators."""

    name: str = ""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.v_cfg = cfg.get("validator", {})
        self.test_cfg = self.v_cfg.get(self.name, {})
        self.results: Dict[str, Any] = {}
        self.run_dir: Optional[Path] = None
        self.json_path: Optional[Path] = None

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
