"""Step 1: Add priority support to todos."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import add_todo, get_todos, get_todo


def test_add_todo_with_priority():
    """Todo items should support an optional priority field (low/medium/high)."""
    # Reset state
    import app
    app.todos = []
    app.next_id = 1

    todo = add_todo("Buy milk", priority="high")
    assert todo["priority"] == "high"

    todo2 = add_todo("Read book")  # default priority
    assert todo2["priority"] == "medium"  # default should be medium


def test_get_todos_sorted_by_priority():
    """get_todos should support sorting by priority (high > medium > low)."""
    import app
    app.todos = []
    app.next_id = 1

    add_todo("Low task", priority="low")
    add_todo("High task", priority="high")
    add_todo("Medium task", priority="medium")

    sorted_todos = get_todos(sort_by_priority=True)
    assert sorted_todos[0]["title"] == "High task"
    assert sorted_todos[1]["title"] == "Medium task"
    assert sorted_todos[2]["title"] == "Low task"
