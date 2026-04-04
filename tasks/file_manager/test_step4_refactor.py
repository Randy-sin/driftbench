"""Step 4: Refactor to FileSystem class."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def test_filesystem_class_exists():
    """FileSystem class should exist with all methods."""
    from app import FileSystem
    fs = FileSystem()
    assert hasattr(fs, 'create_file')
    assert hasattr(fs, 'read_file')
    assert hasattr(fs, 'delete_file')
    assert hasattr(fs, 'list_files')
    assert hasattr(fs, 'create_dir')
    assert hasattr(fs, 'get_info')


def test_filesystem_isolation():
    """Two FileSystem instances should have independent state."""
    from app import FileSystem
    fs1 = FileSystem()
    fs2 = FileSystem()
    fs1.create_file("/test.txt", "data")
    assert len(fs1.list_files()) >= 1
    assert len(fs2.list_files()) == 0


def test_filesystem_preserves_features():
    """FileSystem should support all previously added features."""
    from app import FileSystem
    fs = FileSystem()
    fs.create_file("/docs/readme.txt", "hello")
    info = fs.get_info("/docs/readme.txt")
    assert info["size"] == 5
    assert info["is_dir"] is False
    import pytest
    with pytest.raises(FileExistsError):
        fs.create_file("/docs/readme.txt", "duplicate")
