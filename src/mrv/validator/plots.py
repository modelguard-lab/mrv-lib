"""
mrv.validator.plots -- Shared visualization helpers for validators.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

if matplotlib.get_backend().lower() == "qtagg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mrv.validator.metrics import ARI_THRESHOLD

__all__ = ["plot_ari_heatmap"]


def plot_ari_heatmap(
    ari_matrix: pd.DataFrame,
    asset_name: str,
    out_path: Path,
    title_prefix: str = "Cross-Representation",
) -> None:
    """ARI heatmap shared by rep and res validators.

    Parameters
    ----------
    ari_matrix : DataFrame
        Square ARI matrix.
    asset_name : str
        Asset name for the title.
    out_path : Path
        Output PNG path.
    title_prefix : str
        E.g. ``"Cross-Representation"`` or ``"Cross-Frequency"``.
    """
    n = len(ari_matrix)
    fig, ax = plt.subplots(figsize=(5 + n * 0.5, 4 + n * 0.4))
    data = ari_matrix.values.astype(float)
    im = ax.imshow(data, vmin=-0.1, vmax=1.0, cmap="RdYlGn", aspect="auto")
    labels = (
        list(ari_matrix.columns)
        if title_prefix == "Cross-Frequency"
        else [f"Set {i}" for i in range(n)]
    )
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
    ax.set_title(f"{asset_name}: {title_prefix} ARI\n(threshold = {ARI_THRESHOLD})",
                 fontsize=12, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("ARI", fontsize=10)
    cbar.ax.axhline(y=ARI_THRESHOLD, color="black", linewidth=1.5, linestyle="--")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
