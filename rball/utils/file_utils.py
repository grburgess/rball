import os
import shutil
import tempfile
from builtins import str
from contextlib import contextmanager
from pathlib import Path


def sanitize_filename(filename, abspath: bool = False) -> Path:

    path: Path = Path(filename)

    sanitized = path.expanduser()

    if abspath:

        return sanitized.absolute()

    else:

        return sanitized


def file_existing_and_readable(filename) -> bool:

    sanitized_filename: Path = sanitize_filename(filename)

    return sanitized_filename.is_file()


def path_exists_and_is_directory(path) -> bool:

    sanitized_path: Path = sanitize_filename(path, abspath=True)

    return sanitized_path.is_dir()
