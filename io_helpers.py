"""Utility helpers for folder selection and metric export.

This module provides input/output helpers for the OCR evaluation
workflow, including selecting folders via dialog, locating relevant
folders on disk, and saving metric data to CSV/JSON.
"""

import os
import glob
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.messagebox import showerror
import pandas as pd


def select_folder(initialdirectory, title):
    """Open a dialog to select a folder and return the selected path."""

    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    folder_selected = askdirectory(
        initialdir=initialdirectory, title=title
    )  # show an "Open" dialog box and return the path to the selected folder
    return folder_selected


def validate_and_select_folders(
    initial_dir_gt: str,
    initial_dir_pred: str,
    title_gt: str,
    title_pred: str,
) -> tuple[str, str] | None:
    """Repeatedly prompt user to select GT and prediction folders until
    file counts match. Allows user to abort at any time.

    Args:
        initial_dir_gt: Starting directory for GT folder selection
        initial_dir_pred: Starting directory for prediction folder selection
        title_gt: Title for GT folder selection dialog
        title_pred: Title for prediction folder selection dialog

    Returns:
        Tuple of (folder_gt, folder_pred) if valid, None if user aborts.
    """
    while True:
        # Select GT folder
        folder_gt = select_folder(initial_dir_gt, title_gt)
        if not folder_gt:  # User clicked Cancel
            print("Folder selection cancelled by user.")
            return None

        # Select prediction folder
        folder_pred = select_folder(initial_dir_pred, title_pred)
        if not folder_pred:  # User clicked Cancel
            print("Folder selection cancelled by user.")
            return None

        # Check if file counts match
        files_gt = sorted(glob.glob(os.path.join(folder_gt, "*.txt")))
        files_pred = sorted(glob.glob(os.path.join(folder_pred, "*.txt")))

        if len(files_gt) == len(files_pred):
            print(f"✓ Matched: {len(files_gt)} files in each folder")
            return (folder_gt, folder_pred)
        else:
            error_message = (
                f"File count mismatch!\n\n"
                f"Ground Truth folder: {len(files_gt)} files\n"
                f"Prediction folder: {len(files_pred)} files\n\n"
                f"Please ensure both folders contain the same dataset "
                f"and that the prediction process completed successfully.\n\n"
                f"Click OK to select the folders again, or Cancel to abort."
            )
            print(
                f"✗ Mismatch: GT has {len(files_gt)} files, "
                f"Prediction has {len(files_pred)} files."
            )
            # Show error dialog to user
            root = Tk()
            root.withdraw()
            showerror("Folder Validation Error", error_message)
            root.destroy()

            # Keep the last selected directories as defaults for next attempt
            initial_dir_gt = folder_gt
            initial_dir_pred = folder_pred
            initial_dir_pred = folder_pred


def find_folder(target_name, start_path="/home/"):
    """Search for a folder with the specified name starting from the given path."""

    for root, dirs, _ in os.walk(start_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


def save_metrics(page_metrics: list[dict], folder_pred: str) -> pd.DataFrame:
    """Save per-page metrics to CSV/JSON and return a DataFrame.

    Outputs are written to the parent directory of the selected prediction folder.
    """
    df = pd.DataFrame(page_metrics)

    # Write exports next to the supplied prediction folder.
    csv_output_path = os.path.join(folder_pred, "../metrics_pagewise.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"\nMetrics saved to CSV: {csv_output_path}")

    json_output_path = os.path.join(folder_pred, "../metrics_pagewise.json")
    df.to_json(json_output_path, orient="records", indent=2)
    print(f"Metrics saved to JSON: {json_output_path}")

    return df
