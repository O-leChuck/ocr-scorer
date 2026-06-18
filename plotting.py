"""Helpers for saving metric exports and creating visualizations.

This module centralizes CSV/JSON export and plotting so `main.py` can
focus on orchestration only.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def save_and_plot_metrics(page_metrics: list[dict], folder_pred: str) -> None:
    """Save per-page metrics to CSV/JSON and create a visualization PNG.

    Args:
        page_metrics: list of dicts with per-page metrics (same keys as used
            in the original script).
        folder_pred: path to the predictions folder; used to build output
            paths relative to the predictions directory.
    """
    df = pd.DataFrame(page_metrics)

    # Save metrics to CSV
    csv_output_path = os.path.join(folder_pred, "../metrics_pagewise.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"\nMetrics saved to CSV: {csv_output_path}")

    # Save metrics to JSON
    json_output_path = os.path.join(folder_pred, "../metrics_pagewise.json")
    df.to_json(json_output_path, orient="records", indent=2)
    print(f"Metrics saved to JSON: {json_output_path}")

    # Create visualization
    _, ax = plt.subplots(figsize=(14, 7))

    # Plot the raw and normalized metrics
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

    # Set labels and title
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

    # Save figure
    chart_output_path = os.path.join(folder_pred, "../metrics_visualization.png")
    plt.tight_layout()
    plt.savefig(chart_output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {chart_output_path}")
    plt.show()
