"""ocr-scorer: compute CER/WER between OCR predictions and ground truth.

Public API for using this as a dependency of another pipeline:

    from ocr_scorer import run_evaluation
    output_dir, document_metrics = run_evaluation(gt_path, pred_path)

This top-level import deliberately pulls in only run_evaluation()'s own
dependencies (io_helpers, metrics, plotting) - never ocr_scorer.cli or
ocr_scorer.dialogs, which need tkinter. `import ocr_scorer` must work
in headless environments where tkinter isn't installed at all (a very
common state for minimal/slim Python Docker images used in pipelines
and CI), even though tkinter is a normal part of the Python standard
library on a typical desktop install.
"""

from .evaluate import run_evaluation

__all__ = ["run_evaluation"]
