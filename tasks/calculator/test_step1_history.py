"""Step 1: Add operation history tracking."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import add, subtract, multiply, divide, get_history, clear_history


def test_history_records_operations():
    """Each operation should be recorded in history."""
    clear_history()
    add(2, 3)
    subtract(10, 4)
    h = get_history()
    assert len(h) == 2
    assert h[0]["operation"] == "add"
    assert h[0]["operands"] == [2, 3]
    assert h[0]["result"] == 5


def test_history_clear():
    """clear_history should empty the history list."""
    clear_history()
    add(1, 1)
    assert len(get_history()) == 1
    clear_history()
    assert len(get_history()) == 0


def test_history_all_operations():
    """All four basic operations should be tracked."""
    clear_history()
    add(1, 2)
    subtract(5, 3)
    multiply(2, 4)
    divide(10, 2)
    h = get_history()
    assert len(h) == 4
    ops = [e["operation"] for e in h]
    assert ops == ["add", "subtract", "multiply", "divide"]
