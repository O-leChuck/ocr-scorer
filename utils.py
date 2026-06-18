import os
from tkinter import Tk
from tkinter.filedialog import askdirectory


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
