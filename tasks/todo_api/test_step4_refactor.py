"""Step 4: Refactor — extract a TodoStore class to replace global state."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def test_todo_store_class_exists():
    """A TodoStore class should exist and encapsulate all state."""
    from app import TodoStore
    store = TodoStore()
    assert hasattr(store, 'add_todo')
    assert hasattr(store, 'get_todos')
    assert hasattr(store, 'get_todo')
    assert hasattr(store, 'complete_todo')


def test_todo_store_isolation():
    """Two TodoStore instances should have independent state."""
    from app import TodoStore
    store1 = TodoStore()
    store2 = TodoStore()

    store1.add_todo("Task in store 1")
    assert len(store1.get_todos()) == 1
    assert len(store2.get_todos()) == 0


def test_todo_store_preserves_features():
    """TodoStore should support all previously added features (priority, tags)."""
    from app import TodoStore
    store = TodoStore()

    todo = store.add_todo("Important", priority="high", tags=["work"])
    assert todo["priority"] == "high"
    assert todo["tags"] == ["work"]

    sorted_todos = store.get_todos(sort_by_priority=True)
    assert len(sorted_todos) == 1

    filtered = store.get_todos(filter_tag="work")
    assert len(filtered) == 1
