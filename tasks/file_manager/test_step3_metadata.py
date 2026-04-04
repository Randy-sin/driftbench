"""Step 3: File metadata support."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import time

import app
from app import create_file, get_info, update_file


def test_get_info_basic():
    """get_info should return file metadata."""
    app.files = {}
    create_file("/test.txt", "hello")
    info = get_info("/test.txt")
    assert info["path"] == "/test.txt"
    assert info["size"] == 5
    assert "created_at" in info
    assert "modified_at" in info
    assert info["is_dir"] is False


def test_update_file():
    """update_file should overwrite content and update modified_at."""
    app.files = {}
    create_file("/test.txt", "old")
    info1 = get_info("/test.txt")
    time.sleep(0.01)
    update_file("/test.txt", "new content")
    info2 = get_info("/test.txt")
    assert info2["size"] == len("new content")
    assert info2["modified_at"] >= info1["modified_at"]


def test_metadata_size():
    """Size should reflect content length."""
    app.files = {}
    create_file("/big.txt", "x" * 1000)
    info = get_info("/big.txt")
    assert info["size"] == 1000
