"""Step 3: Add a tagging system to todos."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import add_todo, get_todos, get_todo


def test_add_todo_with_tags():
    """Todos should support an optional list of string tags."""
    import app
    app.todos = []
    app.next_id = 1

    todo = add_todo("Deploy app", priority="high", tags=["work", "urgent"])
    assert "tags" in todo
    assert set(todo["tags"]) == {"work", "urgent"}


def test_add_todo_default_empty_tags():
    """Todos without tags should have an empty list."""
    import app
    app.todos = []
    app.next_id = 1

    todo = add_todo("Relax")
    assert todo["tags"] == []


def test_filter_todos_by_tag():
    """get_todos should support filtering by tag."""
    import app
    app.todos = []
    app.next_id = 1

    add_todo("Work task 1", tags=["work"])
    add_todo("Personal task", tags=["personal"])
    add_todo("Work task 2", tags=["work", "urgent"])

    work_todos = get_todos(filter_tag="work")
    assert len(work_todos) == 2
    assert all("work" in t["tags"] for t in work_todos)
