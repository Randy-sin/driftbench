"""
A minimal in-memory file manager — seed project for DriftBench.
Simulates a simple file system with basic operations.
"""

files = {}  # path -> content mapping


def create_file(path: str, content: str = "") -> dict:
    """Create a new file. Raises FileExistsError if file already exists."""
    if path in files:
        raise FileExistsError(f"File already exists: {path}")
    files[path] = content
    return {"path": path, "size": len(content)}


def read_file(path: str) -> str:
    """Read file content. Raises FileNotFoundError if not found."""
    if path not in files:
        raise FileNotFoundError(f"File not found: {path}")
    return files[path]


def delete_file(path: str) -> bool:
    """Delete a file. Returns True if deleted, raises FileNotFoundError if not found."""
    if path not in files:
        raise FileNotFoundError(f"File not found: {path}")
    del files[path]
    return True


def list_files() -> list:
    """List all file paths."""
    return sorted(files.keys())
