"""Helpers for saving metric exports and creating visualizations.

This module centralizes CSV/JSON export and plotting so `main.py` can
focus on orchestration only.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_metrics_chart(df: pd.DataFrame) -> plt.Figure:
    """Build a matplotlib figure for the CER/WER pagewise chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

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

    return fig


def plot_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """Create a visualization from the metrics DataFrame and save it as PNG."""
    fig = build_metrics_chart(df)
    chart_output_path = os.path.join(output_dir, "metrics_visualization.png")
    plt.tight_layout()
    fig.savefig(chart_output_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {chart_output_path}")
    plt.close(fig)


def create_pdf_report(
    df: pd.DataFrame, document_metrics: dict, output_dir: str
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
        print(f"PDF report saved to: {report_path}")
    else:
        raise FileNotFoundError(f"PDF report was not created: {report_path}")
