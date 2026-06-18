"""Helpers for saving metric exports and creating visualizations.

This module centralizes CSV/JSON export and plotting so `main.py` can
focus on orchestration only.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(df: pd.DataFrame, folder_pred: str) -> None:
    """Create a visualization from the metrics DataFrame."""
    _, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        df.index,
        df["cer_raw"],
        marker="o",
        linewidth=2,
        label="CER raw",
        color="#FF6B6B",
    )
    ax.plot(
        df.index,
        df["wer_raw"],
        marker="s",
        linewidth=2,
        label="WER raw",
        color="#4ECDC4",
    )
    ax.plot(
        df.index,
        df["cer_jiwer_normalized"],
        marker="^",
        linewidth=2,
        label="CER jiwer normalized",
        color="#FFE66D",
    )
    ax.plot(
        df.index,
        df["wer_jiwer_normalized"],
        marker="d",
        linewidth=2,
        label="WER jiwer normalized",
        color="#95E1D3",
    )

    ax.set_xlabel("Page Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "CER/WER per Page - Raw vs jiwer Normalized",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(df)))

    chart_output_path = os.path.join(folder_pred, "../metrics_visualization.png")
    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {chart_output_path}")
    plt.show()
