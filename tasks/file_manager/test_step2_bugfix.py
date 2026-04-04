"""Step 2: Path normalization and directory deletion."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pytest

import app
from app import create_file, read_file, delete_dir


def test_path_normalization():
    """Duplicate slashes should be normalized."""
    app.files = {}
    create_file("a//b//c.txt", "content")
    assert read_file("a/b/c.txt") == "content"


def test_auto_create_parent_dirs():
    """Creating a file should auto-create parent directories."""
    app.files = {}
    create_file("/deep/nested/file.txt", "data")
    content = read_file("/deep/nested/file.txt")
    assert content == "data"


def test_delete_nonempty_dir_raises():
    """Deleting a non-empty directory without recursive should raise OSError."""
    app.files = {}
    create_file("/mydir/file.txt", "data")
    with pytest.raises(OSError, match="not empty"):
        delete_dir("/mydir/")


def test_delete_dir_recursive():
    """delete_dir with recursive=True should delete everything inside."""
    app.files = {}
    create_file("/mydir/file1.txt", "a")
    create_file("/mydir/file2.txt", "b")
    delete_dir("/mydir/", recursive=True)
    with pytest.raises(FileNotFoundError):
        read_file("/mydir/file1.txt")
