"""Step 5: File search with glob patterns."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def test_search_wildcard():
    """search('*.txt') should match .txt files in root."""
    from app import FileSystem
    fs = FileSystem()
    fs.create_file("/readme.txt", "a")
    fs.create_file("/notes.txt", "b")
    fs.create_file("/script.py", "c")
    results = fs.search("*.txt")
    assert len(results) == 2
    assert all(r.endswith(".txt") for r in results)


def test_search_recursive():
    """search('**/*.py') should match .py files recursively."""
    from app import FileSystem
    fs = FileSystem()
    fs.create_file("/src/main.py", "a")
    fs.create_file("/src/utils/helpers.py", "b")
    fs.create_file("/readme.txt", "c")
    results = fs.search("**/*.py")
    assert len(results) == 2
    assert all(r.endswith(".py") for r in results)


def test_search_returns_sorted():
    """Search results should be sorted."""
    from app import FileSystem
    fs = FileSystem()
    fs.create_file("/z.txt", "a")
    fs.create_file("/a.txt", "b")
    fs.create_file("/m.txt", "c")
    results = fs.search("*.txt")
    assert results == sorted(results)
