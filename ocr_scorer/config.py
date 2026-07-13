"""Loads user-configurable default paths from config.ini, if present.

config.ini is intentionally left out of version control (see
.gitignore), since it holds machine-specific paths that are only
meaningful on the machine that created it. config.template.ini
documents the expected format and is the file tracked in git - copy it
to config.ini and fill in your own paths to use this feature.
"""

import configparser
import os

CONFIG_FILENAME = "config.ini"
# config.ini lives at the project root (alongside config.template.ini),
# not inside the ocr_scorer/ package - so this must go up one level from
# this file's own directory, not just os.path.dirname(__file__).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, CONFIG_FILENAME)


def load_default_paths() -> tuple[str | None, str | None]:
    """Read the configured default ground truth/prediction folders.

    Returns (ground_truth_folder, prediction_folder). Either value is
    None if config.ini doesn't exist, can't be parsed, has no value for
    that path, or the configured path doesn't exist on disk - callers
    should treat None as "no configured default" and fall back to their
    own default-detection logic in all of these cases, rather than
    erroring out.
    """
    if not os.path.isfile(_CONFIG_PATH):
        return None, None

    parser = configparser.ConfigParser()
    try:
        parser.read(_CONFIG_PATH, encoding="utf-8")
    except configparser.Error as exc:
        print(f"Warning: could not parse {CONFIG_FILENAME}: {exc}")
        return None, None

    gt_path = parser.get("paths", "ground_truth_folder", fallback="").strip()
    pred_path = parser.get("paths", "prediction_folder", fallback="").strip()

    if gt_path and not os.path.isdir(gt_path):
        print(
            f"Warning: {CONFIG_FILENAME}'s ground_truth_folder "
            f"'{gt_path}' does not exist - ignoring it."
        )
        gt_path = ""
    if pred_path and not os.path.isdir(pred_path):
        print(
            f"Warning: {CONFIG_FILENAME}'s prediction_folder "
            f"'{pred_path}' does not exist - ignoring it."
        )
        pred_path = ""

    return gt_path or None, pred_path or None
