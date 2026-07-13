"""Interactive folder-selection dialogs (tkinter).

This module is intentionally the *only* place in ocr_scorer that
imports tkinter. It is used by ocr_scorer.cli (the interactive/CLI
entry point) and must never be imported from ocr_scorer.evaluate or
ocr_scorer/__init__.py: tkinter's Tk/Tcl bindings are a separate OS
package on many systems (e.g. not included in minimal/slim Python
Docker images) and are not needed at all when this project is used as
a library dependency in another pipeline - `from ocr_scorer import
run_evaluation` must work even where tkinter isn't installed.
"""

import glob
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror


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
