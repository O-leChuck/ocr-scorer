"""Compatibility entry point.

The actual implementation now lives in the ocr_scorer package
(ocr_scorer/cli.py for the interactive/CLI entry point,
ocr_scorer/evaluate.py for run_evaluation()). This file is kept only so
`python main.py` continues to work for anyone running this project
directly from a clone without installing it - for anything else,
prefer `from ocr_scorer import run_evaluation` or the installed
`ocr-scorer` command.
"""

from ocr_scorer.cli import main

if __name__ == "__main__":
    main()
