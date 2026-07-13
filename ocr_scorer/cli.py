"""Interactive / command-line entry point.

This module (and ocr_scorer.dialogs, which it uses) is the only part
of ocr_scorer that touches tkinter or argparse. For programmatic use
from another Python script, import run_evaluation() from ocr_scorer
directly instead of anything in this module.
"""

import argparse
from pathlib import Path

from .config import load_default_paths
from .dialogs import validate_and_select_folders
from .evaluate import run_evaluation


def _resolve_initial_directory(configured_path, label):
    """Pick the starting directory for a folder-selection dialog and
    report which source was used, so a stale/unexpected default is
    visible upfront rather than silently steering folder selection.

    Priority: config.ini > the user's home directory. The home
    directory (via pathlib.Path.home(), which resolves correctly on
    Windows/macOS/Linux) is used as the last resort specifically
    because it always exists and makes no assumption whatsoever about
    this machine's folder structure, username, or OS - unlike a
    hardcoded path, which by definition only ever matches the one
    machine it was hardcoded for.
    """
    if configured_path:
        print(f"Using configured default for {label}: {configured_path}")
        return configured_path
    fallback_path = str(Path.home())
    print(f"Using home directory as default for {label}: {fallback_path}")
    return fallback_path


def main(argv=None):
    """CLI/interactive entry point.

    Runs interactively (folder-picker dialogs) unless both --gt and
    --pred are given, in which case those paths are used directly and
    no dialog is shown - suitable for shell-level pipeline use. For use
    from other Python code, prefer importing run_evaluation() directly
    instead of going through this function at all.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute CER/WER between OCR predictions and ground truth "
            "text files."
        )
    )
    parser.add_argument("--gt", help="Path to the ground truth folder")
    parser.add_argument("--pred", help="Path to the prediction folder")
    args = parser.parse_args(argv)

    if bool(args.gt) != bool(args.pred):
        print("Error: --gt and --pred must both be given together.")
        return

    if args.gt and args.pred:
        folder_gt, folder_pred = args.gt, args.pred
    else:
        configured_gt, configured_pred = load_default_paths()

        initial_directory_gt = _resolve_initial_directory(
            configured_gt, "ground truth folder"
        )
        initial_directory_pred = _resolve_initial_directory(
            configured_pred, "prediction folder"
        )

        title_gt_selection = (
            "Select ANY ONE .txt file - the whole ground truth folder "
            "will be used"
        )
        title_pred_selection = (
            "Select ANY ONE .txt file - the whole predictions folder "
            "will be used"
        )

        # Validate and select folders with retry logic if counts don't match
        result = validate_and_select_folders(
            initial_directory_gt,
            initial_directory_pred,
            title_gt_selection,
            title_pred_selection,
        )

        if result is None:
            print("Operation aborted by user.")
            return

        folder_gt, folder_pred = result

    try:
        run_evaluation(folder_gt, folder_pred)
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
