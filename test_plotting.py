"""
Unit tests for plotting.py.

These are smoke tests: they verify that chart/PDF generation runs
without error and produces the expected output files/structure for
typical and edge-case (single-page, no character errors) inputs. They
do not assert on pixel-level rendering.
"""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed for tests

import matplotlib.pyplot as plt
import pandas as pd

from ocr_scorer.plotting import build_metrics_chart, create_pdf_report, plot_metrics


def _make_df(num_pages: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "page": f"page-{i:02d}",
                "cer_raw": float(i),
                "wer_raw": float(i) * 2,
                "cer_jiwer_normalized": float(i) * 0.5,
                "wer_jiwer_normalized": float(i) * 1.5,
            }
            for i in range(1, num_pages + 1)
        ]
    )


class TestBuildMetricsChart(unittest.TestCase):
    def test_returns_figure_with_one_line_per_metric(self):
        df = _make_df(3)

        fig = build_metrics_chart(df)

        ax = fig.axes[0]
        self.assertEqual(len(ax.lines), 4)
        plt.close(fig)

    def test_handles_single_page_without_error(self):
        df = _make_df(1)

        fig = build_metrics_chart(df)

        self.assertEqual(len(fig.axes[0].lines), 4)
        plt.close(fig)

    def test_infinite_value_gets_undefined_marker_and_legend_entry(self):
        # an empty-reference page the OCR hallucinated text onto: CER raw
        # is +inf and must not be silently invisible in the chart
        df = _make_df(3)
        df.loc[1, "cer_raw"] = float("inf")

        fig = build_metrics_chart(df)
        ax = fig.axes[0]

        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertIn("undefined (empty reference)", legend_labels)
        # exactly one marker collection for the undefined-value flag
        self.assertEqual(len(ax.collections), 1)
        # the line itself must not error out or plot an infinite y-value
        for line in ax.lines:
            y_data = line.get_ydata()
            self.assertTrue(all(v != float("inf") for v in y_data))
        plt.close(fig)

    def test_nan_value_does_not_add_a_spurious_undefined_marker(self):
        # a both-empty page (true 0/0): a plain gap in the line, not
        # flagged the same way as a hallucination
        df = _make_df(3)
        df.loc[1, "cer_raw"] = float("nan")

        fig = build_metrics_chart(df)
        ax = fig.axes[0]

        self.assertEqual(len(ax.collections), 0)
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertNotIn("undefined (empty reference)", legend_labels)
        plt.close(fig)

    def test_multiple_infinite_pages_share_one_marker_series(self):
        df = _make_df(4)
        df.loc[1, "cer_raw"] = float("inf")
        df.loc[3, "wer_raw"] = float("inf")

        fig = build_metrics_chart(df)
        ax = fig.axes[0]

        self.assertEqual(len(ax.collections), 1)
        marker_x = sorted(ax.collections[0].get_offsets()[:, 0])
        self.assertEqual(marker_x, [1, 3])
        plt.close(fig)


class TestPlotMetrics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_png_file(self):
        df = _make_df(3)

        plot_metrics(df, self.output_dir)

        png_path = os.path.join(self.output_dir, "metrics_visualization.png")
        self.assertTrue(os.path.isfile(png_path))
        self.assertGreater(os.path.getsize(png_path), 0)


class TestCreatePdfReport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_writes_pdf_file(self):
        df = _make_df(3)
        document_metrics = {
            "summary": {
                "evaluation_date": "2026-01-01",
                "cer_raw": 1.23,
                "wer_raw": 4.56,
                "cer_normalized": 0.5,
                "wer_normalized": 2.0,
                "page_count": 3,
            },
            "top_error_chars_raw": [
                {"rank": 1, "character": "e", "count": 10, "percent": 50.0}
            ],
            "top_error_chars_normalized": [
                {"rank": 1, "character": "e", "count": 8, "percent": 40.0}
            ],
        }

        create_pdf_report(df, document_metrics, self.output_dir)

        pdf_path = os.path.join(self.output_dir, "evaluation_report.pdf")
        self.assertTrue(os.path.isfile(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 0)

    def test_handles_empty_top_error_lists(self):
        # a corpus with zero attributable character errors (e.g. identical
        # texts) should not crash report generation
        df = _make_df(1)
        document_metrics = {
            "summary": {
                "evaluation_date": "2026-01-01",
                "cer_raw": 0.0,
                "wer_raw": 0.0,
                "cer_normalized": 0.0,
                "wer_normalized": 0.0,
                "page_count": 1,
            },
            "top_error_chars_raw": [],
            "top_error_chars_normalized": [],
        }

        create_pdf_report(df, document_metrics, self.output_dir)

        pdf_path = os.path.join(self.output_dir, "evaluation_report.pdf")
        self.assertTrue(os.path.isfile(pdf_path))


if __name__ == "__main__":
    unittest.main()
