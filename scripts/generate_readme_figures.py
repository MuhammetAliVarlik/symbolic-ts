"""Regenerate the README figures under docs/figures/.

Not part of the library or its test suite -- a documentation utility only.
Uses a fixed random seed so the figures are exactly reproducible, and calls
the real project()/SigmaBinner code (not standalone plotting logic), so the
pictures never drift out of sync with what the library actually does.

Run: .venv/bin/python scripts/generate_readme_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from symbolic_ts.binning import SigmaBinner
from symbolic_ts.projection import FINANCE_ADAPTER, project
from symbolic_ts.vocabulary import NEUTRAL_CHANGE_LABELS

FIGURES_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
RNG = np.random.default_rng(seed=7)


def _synthetic_price_series(n: int = 400) -> pd.DataFrame:
    """A geometric random walk standing in for a real 'Adj Close' column --
    good enough to demonstrate what project() does without committing a real
    dataset to a library repo that has no business owning one."""
    daily_log_returns = RNG.normal(loc=0.0003, scale=0.012, size=n)
    price = 100.0 * np.exp(np.cumsum(daily_log_returns))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Adj Close": price}, index=dates)


def plot_projection_example(raw: pd.DataFrame, projected: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(raw.index, raw["Adj Close"], color="#1f77b4")
    axes[0].set_title("Input: 1 raw channel (Adj Close)")
    axes[0].set_ylabel("price")

    axes[1].plot(projected.index, projected["change"], color="#d62728", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("Output channel 1: change (log return)")
    axes[1].set_ylabel("log return")

    axes[2].plot(projected.index, projected["volatility"], color="#2ca02c")
    axes[2].set_title("Output channel 2: volatility (rolling std of change, window=20)")
    axes[2].set_ylabel("rolling std")

    fig.suptitle("project(): N input channels -> 2 summary channels", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "projection_example.png", dpi=150)
    plt.close(fig)


def plot_sigma_binning_example(projected: pd.DataFrame) -> None:
    split = int(len(projected) * 0.7)
    train_change = projected["change"].iloc[:split]

    binner = SigmaBinner(labels=NEUTRAL_CHANGE_LABELS).fit(train_change)
    edges_in_data_units = [binner.mean_ + e * binner.std_ for e in binner.sigma_edges]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_change, bins=60, color="#9ecae1", edgecolor="none")

    for edge in edges_in_data_units:
        ax.axvline(edge, color="black", linestyle="--", linewidth=1)

    boundaries = [train_change.min(), *edges_in_data_units, train_change.max()]
    y_top = ax.get_ylim()[1]
    for label, lo, hi in zip(binner.labels, boundaries[:-1], boundaries[1:]):
        ax.text((lo + hi) / 2, y_top * 0.95, label, ha="center", va="top", fontweight="bold")

    ax.set_title("SigmaBinner fit on the training split: 5 buckets at +/-0.5σ, +/-1.5σ")
    ax.set_xlabel("change (log return)")
    ax.set_ylabel("count (training rows)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sigma_binning_example.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = _synthetic_price_series()
    projected = project(raw, FINANCE_ADAPTER)
    plot_projection_example(raw, projected)
    plot_sigma_binning_example(projected)
    print(f"wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
