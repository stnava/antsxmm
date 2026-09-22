from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

DATA_PATH = os.path.expanduser("~/.antspymm/")


def get_data(
    name: str | None = None,
    force_download: bool = False,
    version: int = 26,
    target_extension: str = ".csv",
) -> str | list[str] | None:
    """Get ANTsPyMM data filename.

    The first time this is called, it will download data to ~/.antspymm.
    After, it will just read data from disk.

    Arguments
    ---------
    name : string
        name of data tag to retrieve. Options: None, 'all', or specific tag stem.
    force_download : boolean
        Whether to re-download even if files exist.
    version : integer
        Version of data to download.
    target_extension : string
        Target file extension to filter by (default: '.csv').

    Returns
    -------
    string or list of strings or None
        Filepath or list of filepaths of selected data.
    """
    os.makedirs(DATA_PATH, exist_ok=True)

    def _mv_subfolder_files(folder: str, verbose: bool = False) -> None:
        for root, dirs, files in os.walk(folder):
            if verbose:
                print(f"Processing directory: {root}")
            for file in files:
                if root != folder:
                    src = os.path.join(root, file)
                    dst = os.path.join(folder, file)
                    if not os.path.exists(dst):
                        shutil.move(src, folder)
            for d in dirs:
                if root != folder:
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)

    def _download_data(ver: int) -> None:
        try:
            import tensorflow as tf
            url = f"https://ndownloader.figshare.com/articles/16912366/versions/{ver}"
            target_file_name = "16912366.zip"
            tf.keras.utils.get_file(
                target_file_name,
                url,
                cache_subdir=DATA_PATH,
                extract=True,
            )
            _mv_subfolder_files(DATA_PATH, False)
            zip_path = os.path.join(DATA_PATH, target_file_name)
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception as err:
            raise RuntimeError(f"Failed to download ANTsPyMM reference data: {err}") from err

    if force_download:
        _download_data(version)

    files: list[str] = []
    if os.path.exists(DATA_PATH):
        for fname in os.listdir(DATA_PATH):
            if fname.endswith(target_extension):
                files.append(os.path.join(DATA_PATH, fname))

    if len(files) == 0 and not force_download:
        _download_data(version)
        if os.path.exists(DATA_PATH):
            for fname in os.listdir(DATA_PATH):
                if fname.endswith(target_extension):
                    files.append(os.path.join(DATA_PATH, fname))

    if name == "all":
        return files

    if name is None:
        return None

    for fname in os.listdir(DATA_PATH):
        p = Path(fname)
        stem = p.resolve().stem
        while "." in stem:
            stem = Path(stem).stem
        if name == stem and fname.endswith(target_extension):
            return os.path.join(DATA_PATH, fname)

    return None


def get_models(version: int = 3, force_download: bool = True) -> None:
    """Download and cache ANTsPyMM deep learning models."""
    os.makedirs(DATA_PATH, exist_ok=True)
    import tensorflow as tf

    url = f"https://ndownloader.figshare.com/articles/21718412/versions/{version}"
    target_file_name = "21718412.zip"
    tf.keras.utils.get_file(
        target_file_name,
        url,
        cache_subdir=DATA_PATH,
        extract=True,
    )
    zip_path = os.path.join(DATA_PATH, target_file_name)
    if os.path.exists(zip_path):
        os.remove(zip_path)
