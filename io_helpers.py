"""Utility helpers for folder selection and metric export.

This module provides input/output helpers for the OCR evaluation
workflow, including selecting folders via dialog, locating relevant
folders on disk, and saving metric data to CSV/JSON.
"""

import os
from tkinter import Tk
from tkinter.filedialog import askdirectory
import pandas as pd


def select_folder(initialdirectory, title):
    """Open a dialog to select a folder and return the selected path."""

    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    folder_selected = askdirectory(
        initialdir=initialdirectory, title=title
    )  # show an "Open" dialog box and return the path to the selected folder
    return folder_selected


def find_folder(target_name, start_path="/home/"):
    """Search for a folder with the specified name starting from the given path."""

    for root, dirs, _ in os.walk(start_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


def save_metrics(page_metrics: list[dict], folder_pred: str) -> pd.DataFrame:
    """Save per-page metrics to CSV/JSON and return a DataFrame."""
    df = pd.DataFrame(page_metrics)

    csv_output_path = os.path.join(folder_pred, "../metrics_pagewise.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"\nMetrics saved to CSV: {csv_output_path}")

    json_output_path = os.path.join(folder_pred, "../metrics_pagewise.json")
    df.to_json(json_output_path, orient="records", indent=2)
    print(f"Metrics saved to JSON: {json_output_path}")

    return df
