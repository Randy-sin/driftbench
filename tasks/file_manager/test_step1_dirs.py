"""Step 1: Directory support."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pytest

import app
from app import create_file, create_dir, list_dir, is_dir


def test_create_dir():
    """create_dir should create a directory."""
    app.files = {}
    create_dir("/docs/")
    assert is_dir("/docs/")


def test_create_dir_exists_raises():
    """Creating an existing directory should raise FileExistsError."""
    app.files = {}
    create_dir("/docs/")
    with pytest.raises(FileExistsError):
        create_dir("/docs/")


def test_list_dir():
    """list_dir should list contents of a directory."""
    app.files = {}
    create_dir("/docs/")
    create_file("/docs/readme.txt", "hello")
    contents = list_dir("/docs/")
    assert "readme.txt" in contents or "/docs/readme.txt" in contents


def test_list_dir_not_found():
    """Listing a non-existent directory should raise FileNotFoundError."""
    app.files = {}
    with pytest.raises(FileNotFoundError):
        list_dir("/nonexistent/")
