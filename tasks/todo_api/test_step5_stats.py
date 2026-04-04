"""Step 5: Add statistics/analytics to the TodoStore."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def test_stats_basic():
    """TodoStore should have a get_stats() method returning summary statistics."""
    from app import TodoStore
    store = TodoStore()

    store.add_todo("Task 1", priority="high", tags=["work"])
    store.add_todo("Task 2", priority="low", tags=["personal"])
    store.add_todo("Task 3", priority="high", tags=["work"])
    store.complete_todo(1)

    stats = store.get_stats()
    assert stats["total"] == 3
    assert stats["completed"] == 1
    assert stats["pending"] == 2


def test_stats_by_priority():
    """Stats should include a breakdown by priority."""
    from app import TodoStore
    store = TodoStore()

    store.add_todo("High 1", priority="high")
    store.add_todo("High 2", priority="high")
    store.add_todo("Low 1", priority="low")

    stats = store.get_stats()
    assert stats["by_priority"]["high"] == 2
    assert stats["by_priority"]["low"] == 1


def test_stats_by_tag():
    """Stats should include a breakdown by tag."""
    from app import TodoStore
    store = TodoStore()

    store.add_todo("T1", tags=["work", "urgent"])
    store.add_todo("T2", tags=["work"])
    store.add_todo("T3", tags=["personal"])

    stats = store.get_stats()
    assert stats["by_tag"]["work"] == 2
    assert stats["by_tag"]["urgent"] == 1
    assert stats["by_tag"]["personal"] == 1


def test_completion_rate():
    """Stats should include a completion rate percentage."""
    from app import TodoStore
    store = TodoStore()

    store.add_todo("T1")
    store.add_todo("T2")
    store.complete_todo(1)

    stats = store.get_stats()
    assert stats["completion_rate"] == 50.0
