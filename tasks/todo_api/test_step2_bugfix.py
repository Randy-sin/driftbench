"""Step 2: Fix the bug where completing a non-existent todo raises no error."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import add_todo, complete_todo


def test_complete_nonexistent_todo_raises():
    """Completing a todo that doesn't exist should raise a ValueError."""
    import app
    app.todos = []
    app.next_id = 1

    import pytest
    with pytest.raises(ValueError, match="not found"):
        complete_todo(999)


def test_complete_already_done_raises():
    """Completing an already-done todo should raise a ValueError."""
    import app
    app.todos = []
    app.next_id = 1

    todo = add_todo("Test task", priority="medium")
    complete_todo(todo["id"])

    import pytest
    with pytest.raises(ValueError, match="already"):
        complete_todo(todo["id"])
