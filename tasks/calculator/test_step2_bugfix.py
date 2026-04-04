"""Step 2: Fix divide edge cases and history error handling."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pytest

from app import divide, get_history, clear_history


def test_divide_zero_by_zero():
    """divide(0, 0) should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        divide(0, 0)


def test_divide_returns_float():
    """divide should always return a float."""
    clear_history()
    result = divide(10, 3)
    assert isinstance(result, float)


def test_failed_operations_not_in_history():
    """Failed operations (exceptions) should NOT be recorded in history."""
    clear_history()
    divide(10, 2)  # should succeed
    try:
        divide(1, 0)  # should fail
    except ZeroDivisionError:
        pass
    h = get_history()
    assert len(h) == 1  # only the successful one
