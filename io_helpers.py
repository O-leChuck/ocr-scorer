"""Utility helpers for folder selection and metric export.

This module provides input/output helpers for the OCR evaluation
workflow, including selecting folders via dialog, locating relevant
folders on disk, and saving metric data to CSV/JSON.
"""

import json
import math
import os
import re
import glob
from datetime import date
from typing import Any
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import pandas as pd

# Matches a page/sequence number marked with a "p"/"page" prefix, e.g.
# "p0004_..." -> 4, "..._p0142.txt" -> 142. Longer alternative first so
# "page" isn't shadowed by "p" during matching.
_PAGE_MARKER_PATTERN = re.compile(r"(?:page|p)(\d+)", re.IGNORECASE)


def select_folder(initialdirectory, title):
    """Open a dialog to select a folder and return the selected path.

    This asks the user to pick any one file *inside* the target folder,
    rather than the folder itself, and returns that file's parent
    directory. tkinter's directory-only picker (askdirectory) hides
    files by design on every platform, since its purpose is choosing a
    container rather than a file - which made it easy to end up in the
    wrong folder (e.g. an unrelated same-named folder elsewhere) without
    ever seeing that it didn't contain the expected files. A file picker
    shows the folder's contents while browsing, so a wrong/empty folder
    is visible immediately instead of surfacing later as a confusing
    file-count mismatch.

    The dialog is restricted to .txt files only (no "all files" escape
    hatch), since that's the only format this tool processes - letting
    someone pick a different file type would defeat the point above, by
    allowing a folder with no usable files to be "selected" anyway. A
    folder with no .txt files will simply appear empty in the dialog.
    """

    print(
        f"{title} - only .txt files are shown, since this tool only "
        "processes plain-text .txt files."
    )

    # we don't want a full GUI, so keep the root window from appearing
    root = Tk()
    root.withdraw()
    try:
        file_selected = askopenfilename(
            initialdir=initialdirectory,
            title=title,
            filetypes=[("Text files", "*.txt")],
        )
    finally:
        root.destroy()
    return os.path.dirname(file_selected) if file_selected else ""


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
        print(f"Selected Goldstandard folder: {folder_gt}")

        # Select prediction folder
        folder_pred = select_folder(initial_dir_pred, title_pred)
        if not folder_pred:  # User clicked Cancel
            print("Folder selection cancelled by user.")
            return None
        print(f"Selected prediction folder: {folder_pred}")

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


def find_folder(target_name, start_path="/home/"):
    """
    Search for a folder with the specified name starting from the given
    path.
    """

    for root, dirs, _ in os.walk(start_path):
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


def extract_page_number(filename: str) -> int | None:
    """Best-effort extraction of a page number from a filename.

    Prefers a digit run immediately preceded by "p" or "page"
    (case-insensitive), e.g. "p0004_...txt" -> 4. Falls back to the
    filename's only digit run if there is exactly one. Returns None if no
    page number can be identified unambiguously (multiple "p"/"page"
    markers, or multiple digit runs with no marker to disambiguate them).
    """
    basename = os.path.basename(filename)
    marker_matches = _PAGE_MARKER_PATTERN.findall(basename)
    if len(marker_matches) == 1:
        return int(marker_matches[0])
    if len(marker_matches) > 1:
        return None

    digit_runs = re.findall(r"\d+", basename)
    if len(digit_runs) == 1:
        return int(digit_runs[0])
    return None


def check_page_number_alignment(
    files_gt: list[str], files_pred: list[str]
) -> tuple[bool, list[str]]:
    """Best-effort sanity check for the sort-order file pairing used by
    main.py.

    Files are otherwise paired purely by their position in each sorted
    list, since prediction filenames are not guaranteed to resemble GT
    filenames at all. This check extracts a page number from every
    filename (see extract_page_number) and compares them position by
    position, as a heuristic warning - not a guarantee.

    Returns (used, messages):
      - used=True if a page number was extracted for every file in both
        lists, so the comparison could be performed. messages then
        contains one line per position where the extracted GT/prediction
        page numbers disagree (empty if all positions matched).
      - used=False if a page number could not be reliably extracted for
        at least one file, so no comparison was made. messages is empty
        in this case - silence is intentional to avoid false alarms from
        an unreliable heuristic.
    """
    gt_numbers = [extract_page_number(f) for f in files_gt]
    pred_numbers = [extract_page_number(f) for f in files_pred]

    if any(n is None for n in gt_numbers) or any(
        n is None for n in pred_numbers
    ):
        return False, []

    messages = []
    for position, (gt_file, pred_file, gt_num, pred_num) in enumerate(
        zip(files_gt, files_pred, gt_numbers, pred_numbers), start=1
    ):
        if gt_num != pred_num:
            messages.append(
                f"Position {position}: GT file '{os.path.basename(gt_file)}' "
                f"(page {gt_num}) is paired with prediction file "
                f"'{os.path.basename(pred_file)}' (page {pred_num}) - "
                "these do not look like the same page."
            )
    return True, messages


def _make_json_safe(value: Any) -> Any:
    """Recursively make a value safe for strict JSON export.

    JSON has no token for IEEE-754 infinity or NaN. +inf/-inf are
    converted to the strings "Infinity"/"-Infinity" - a deliberate choice
    so a value that must not be silently mistaken for "no data" (e.g. an
    empty-reference page where the OCR hallucinated text) stays visible
    and unmistakable to anyone reading the JSON, even though it means
    that field is no longer purely numeric. NaN (a true 0/0 - nothing to
    measure) becomes None/null, which is the standard "no value" token.
    """
    if isinstance(value, float):
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _make_json_safe(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    return value


def save_metrics(
    page_metrics: list[dict],
    output_dir: str,
    document_metrics: dict | None = None,
) -> pd.DataFrame:
    """Save per-page metrics to CSV/JSON and optionally document-level metrics.

    Outputs are written to the provided evaluation output directory.
    """
    df = pd.DataFrame(page_metrics)

    csv_output_path = os.path.join(output_dir, "metrics_pagewise.csv")
    df.to_csv(csv_output_path, index=False)
    print(f"\nMetrics saved to CSV: {csv_output_path}")

    json_output_path = os.path.join(output_dir, "metrics_pagewise.json")
    json_records = _make_json_safe(df.to_dict(orient="records"))
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(json_records, json_file, indent=2, ensure_ascii=False)
    print(f"Metrics saved to JSON: {json_output_path}")

    if document_metrics is not None:
        save_document_metrics(document_metrics, output_dir)

    return df


def save_document_metrics(document_metrics: dict, output_dir: str) -> None:
    """Save document-level metrics and top-error metadata to CSV/JSON."""

    # Write a clean JSON summary (nested structure)
    json_output_path = os.path.join(output_dir, "metrics_document.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(
            _make_json_safe(document_metrics),
            json_file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Document metrics saved to JSON: {json_output_path}")

    # Export human-friendly CSV files: summary and two top-letter tables
    summary = document_metrics.get("summary", {})
    summary_rows = [{"metric": k, "value": v} for k, v in summary.items()]
    summary_csv_path = os.path.join(output_dir, "metrics_document_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    print(f"Document summary saved to CSV: {summary_csv_path}")

    def _write_top_table(section_name: str, filename: str):
        items = document_metrics.get(section_name, [])
        if not items:
            return
        table_rows = [
            {
                "rank": it.get("rank"),
                "character": it.get("character"),
                "count": it.get("count"),
                "percent": it.get("percent"),
            }
            for it in items
        ]
        out_path = os.path.join(output_dir, filename)
        pd.DataFrame(table_rows).to_csv(out_path, index=False)
        print(f"{section_name} saved to CSV: {out_path}")

    _write_top_table("top_error_chars_raw", "metrics_document_top_raw.csv")
    _write_top_table(
        "top_error_chars_normalized", "metrics_document_top_normalized.csv"
    )


def make_evaluation_output_folder(folder_pred: str) -> str:
    """
    Create a dated evaluation folder next to the prediction folder.
    """

    parent_dir = os.path.dirname(folder_pred)
    evaluation_folder = os.path.join(
        parent_dir, f"evaluation_{date.today().isoformat()}"
    )
    os.makedirs(evaluation_folder, exist_ok=True)
    return evaluation_folder


def format_page_number_check_report(
    used: bool, messages: list[str]
) -> list[str]:
    """
    Format the result of check_page_number_alignment as report lines,
    for printing to the terminal and/or writing to the evaluation log.
    """
    if not used:
        return [
            "Page-number check: could not reliably extract a page number "
            "from every filename (ambiguous or missing numbering); this "
            "sanity check was skipped."
        ]
    if not messages:
        return [
            "Page-number check: page numbers were extracted from every "
            "filename, and all files appear to be correctly paired."
        ]
    lines = [
        "Page-number check: page numbers were extracted from every "
        f"filename, but {len(messages)} potential mismatch(es) were found:"
    ]
    lines.extend(f"  - {message}" for message in messages)
    return lines


def save_evaluation_log(
    folder_gt: str,
    folder_pred: str,
    output_dir: str,
    evaluation_date: str | None = None,
    page_number_check: tuple[bool, list[str]] | None = None,
) -> None:
    """
    Save an evaluation log containing folder metadata and evaluation
    date.
    """

    log_date = evaluation_date if evaluation_date else date.today().isoformat()
    gold_files = sorted(glob.glob(os.path.join(folder_gt, "*.txt")))
    pred_files = sorted(glob.glob(os.path.join(folder_pred, "*.txt")))
    txt_output_path = os.path.join(output_dir, "evaluation_log.txt")
    with open(txt_output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("Evaluation Log\n")
        txt_file.write("===============\n")
        txt_file.write(f"Evaluation date: {log_date}\n")
        txt_file.write(f"Goldstandard folder: {folder_gt}\n")
        txt_file.write(f"Prediction folder: {folder_pred}\n")
        txt_file.write(f"Export folder: {output_dir}\n")
        txt_file.write(f"Page count: {len(gold_files)}\n")
        if page_number_check is not None:
            used, messages = page_number_check
            txt_file.write("\n")
            for line in format_page_number_check_report(used, messages):
                txt_file.write(f"{line}\n")
        txt_file.write("\nGround truth files:\n")
        for file_name in [os.path.basename(path) for path in gold_files]:
            txt_file.write(f"  - {file_name}\n")
        txt_file.write("\nPrediction files:\n")
        for file_name in [os.path.basename(path) for path in pred_files]:
            txt_file.write(f"  - {file_name}\n")
    print(f"Evaluation log saved to: {txt_output_path}")
