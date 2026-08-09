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
from symbolic_ts.encoding import compare_encoders
from symbolic_ts.events import ETTEventAdapter
from symbolic_ts.projection import FINANCE_ADAPTER, project
from symbolic_ts.splits import walk_forward_split
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


def plot_event_calendar_example() -> None:
    timestamps = pd.date_range("2024-02-25", periods=24 * 10, freq="h")
    adapter = ETTEventAdapter(
        holiday_dates=frozenset({pd.Timestamp("2024-03-01").date()}),
        peak_hours=frozenset({18, 19, 20}),
    )
    event_names = ["E_WEEKEND", "E_HOLIDAY", "E_HOUR_PEAK", "E_SEASON_CHANGE"]
    grid = np.zeros((len(event_names), len(timestamps)))
    for col, ts in enumerate(timestamps):
        fired = adapter.events_for(ts.to_pydatetime())
        for row, name in enumerate(event_names):
            grid[row, col] = 1.0 if name in fired else 0.0

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.imshow(grid, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(event_names)))
    ax.set_yticklabels(event_names)
    tick_positions = range(0, len(timestamps), 24)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([timestamps[k].strftime("%b %d") for k in tick_positions], rotation=45, ha="right")
    ax.set_title("ETTEventAdapter.events_for() over 10 hourly-sampled days")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "event_calendar_example.png", dpi=150)
    plt.close(fig)


class _CharTokenizer:
    """Counts one 'token' per character -- NOT a real subword tokeniser. Used
    here only to get an honest, reproducible, offline number out of
    compare_encoders() for the figure; the actual subword-token figures
    (~6.7 tokens/symbol semantic vs. ~1 token/symbol single-character) are
    measured separately in symbolic-ts-research's F0-09 notes under a real
    tokeniser and are not recomputed here."""

    def encode(self, text: str) -> list[str]:
        return list(text)


def plot_encoding_comparison_example() -> None:
    tokens = ["C0_V0", "C1_V2", "C2_V1", "C3_V3", "C4_V4", "C2_V2", "C1_V1", "C3_V0"]
    reports = compare_encoders(tokens, _CharTokenizer())
    names = [r.encoder_name for r in reports]
    char_counts = [r.char_count for r in reports]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, char_counts, color=["#1f77b4", "#d62728", "#2ca02c"])
    for bar, count in zip(bars, char_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(count), ha="center")
    ax.set_ylabel("prompt length (characters)")
    ax.set_title(f"Same {len(tokens)}-symbol sequence, three encoders")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "encoding_comparison_example.png", dpi=150)
    plt.close(fig)


def plot_walk_forward_splits_example() -> None:
    n = 220
    context_len, embargo, n_folds = 20, 5, 3
    folds = walk_forward_split(np.zeros(n), n_folds=n_folds, context_len=context_len, embargo=embargo)

    fig, ax = plt.subplots(figsize=(10, 2.2 + 0.4 * n_folds))
    for row, fold in enumerate(folds):
        ax.barh(row, fold.train_indices.max() + 1, left=0, color="#1f77b4", label="train" if row == 0 else None)
        gap_start = fold.train_indices.max() + 1
        gap_width = fold.validation_indices.min() - gap_start
        ax.barh(row, gap_width, left=gap_start, color="#dddddd", label="purge + embargo" if row == 0 else None)
        ax.barh(
            row,
            fold.validation_indices.max() - fold.validation_indices.min() + 1,
            left=fold.validation_indices.min(),
            color="#d62728",
            label="validation" if row == 0 else None,
        )

    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"fold {k}" for k in range(n_folds)])
    ax.set_xlabel("index")
    ax.set_title(f"walk_forward_split(n_folds={n_folds}, context_len={context_len}, embargo={embargo})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "walk_forward_split_example.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = _synthetic_price_series()
    projected = project(raw, FINANCE_ADAPTER)
    plot_projection_example(raw, projected)
    plot_sigma_binning_example(projected)
    plot_event_calendar_example()
    plot_encoding_comparison_example()
    plot_walk_forward_splits_example()
    print(f"wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
