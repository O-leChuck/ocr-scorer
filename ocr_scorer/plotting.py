"""Helpers for saving metric exports and creating visualizations.

This module centralizes CSV/JSON export and plotting so evaluate.py
can focus on orchestration only.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

# Force a non-interactive backend before importing pyplot. This module
# only ever saves figures to disk (PNG/PDF) and never displays them, so
# it must not depend on a display being available - relevant both when
# run headless (e.g. a server/CI pipeline importing ocr_scorer as a
# library) and to avoid flaky, environment-dependent backend
# auto-selection.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_METRIC_SPECS = [
    ("cer_raw", "o", "CER raw", "#FF6B6B"),
    ("wer_raw", "s", "WER raw", "#4ECDC4"),
    ("cer_jiwer_normalized", "^", "CER jiwer normalized", "#FFE66D"),
    ("wer_jiwer_normalized", "d", "WER jiwer normalized", "#95E1D3"),
]


def build_metrics_chart(df: pd.DataFrame) -> plt.Figure:
    """Build a matplotlib figure for the CER/WER pagewise chart.

    A page whose value is +inf (an empty-reference page the OCR
    hallucinated text onto - see README, "Empty-reference pages") can't
    be drawn at its true height, so it's left as a gap in the line and
    flagged with an explicit "undefined" marker at the top of the chart
    instead of silently plotting off-scale and disappearing.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    undefined_positions = set()
    for column, marker, label, color in _METRIC_SPECS:
        undefined_positions.update(df.index[np.isposinf(df[column])])
        values = df[column].replace([np.inf, -np.inf], np.nan)
        ax.plot(
            df.index,
            values,
            marker=marker,
            linewidth=2,
            label=label,
            color=color,
        )

    ax.set_xlabel("Page Number", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "CER/WER per Page - Raw vs jiwer Normalized",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(df)))

    if undefined_positions:
        _, ylim_top = ax.get_ylim()
        ax.scatter(
            sorted(undefined_positions),
            [ylim_top] * len(undefined_positions),
            marker="^",
            color="red",
            s=120,
            zorder=5,
            clip_on=False,
            label="undefined (empty reference)",
        )

    ax.legend(loc="best", fontsize=10)

    return fig


def plot_metrics(
    df: pd.DataFrame, output_dir: str, *, verbose: bool = True
) -> None:
    """Create a visualization from the metrics DataFrame and save it as PNG."""
    fig = build_metrics_chart(df)
    chart_output_path = os.path.join(output_dir, "metrics_visualization.png")
    plt.tight_layout()
    fig.savefig(chart_output_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"Visualization saved to: {chart_output_path}")
    plt.close(fig)


def create_pdf_report(
    df: pd.DataFrame,
    document_metrics: dict,
    output_dir: str,
    *,
    verbose: bool = True,
) -> None:
    """Create a PDF report with summary text and the metrics chart."""
    report_path = os.path.join(output_dir, "evaluation_report.pdf")

    with PdfPages(report_path) as pdf:
        # Page 1: summary and top-letter tables
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")

        summary = document_metrics.get("summary", {})
        lines = [
            "OCR Evaluation Report",
            "====================",
            f"Date: {summary.get('evaluation_date', '')}",
            "",
            "Summary Metrics:",
            f"  CER raw: {summary.get('cer_raw', 0):.2f}%",
            f"  WER raw: {summary.get('wer_raw', 0):.2f}%",
            f"  CER normalized: {summary.get('cer_normalized', 0):.2f}%",
            f"  WER normalized: {summary.get('wer_normalized', 0):.2f}%",
            f"  Page count: {summary.get('page_count', '')}",
            "",
            "Top 10 raw character errors:",
        ]

        for item in document_metrics.get("top_error_chars_raw", []):
            lines.append(
                f"  {item['rank']:2d}. '{item['character']}' — {item['count']} errors ({item['percent']:.2f}%)"
            )

        lines.append("")
        lines.append("Top 10 normalized character errors:")
        for item in document_metrics.get("top_error_chars_normalized", []):
            lines.append(
                f"  {item['rank']:2d}. '{item['character']}' — {item['count']} errors ({item['percent']:.2f}%)"
            )

        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            fontsize=10,
            va="top",
            family="monospace",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: chart visualization
        fig2 = build_metrics_chart(df)
        plt.tight_layout()
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    if os.path.isfile(report_path):
        if verbose:
            print(f"PDF report saved to: {report_path}")
    else:
        raise FileNotFoundError(f"PDF report was not created: {report_path}")
