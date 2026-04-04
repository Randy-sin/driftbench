"""Step 3: Advanced math operations."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pytest
import math

from app import power, sqrt, modulo, get_history, clear_history


def test_power():
    """power(a, b) should return a ** b."""
    assert power(2, 3) == 8
    assert power(5, 0) == 1


def test_sqrt():
    """sqrt(a) should return the square root."""
    assert sqrt(16) == 4.0
    assert abs(sqrt(2) - math.sqrt(2)) < 1e-10


def test_sqrt_negative_raises():
    """sqrt of negative number should raise ValueError."""
    with pytest.raises(ValueError, match="negative"):
        sqrt(-1)


def test_modulo():
    """modulo(a, b) should return a % b."""
    assert modulo(10, 3) == 1
    assert modulo(7, 2) == 1


def test_modulo_zero_raises():
    """modulo by zero should raise ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        modulo(5, 0)


def test_advanced_ops_in_history():
    """Advanced operations should be recorded in history."""
    clear_history()
    power(2, 3)
    sqrt(9)
    modulo(10, 3)
    h = get_history()
    assert len(h) == 3
    ops = [e["operation"] for e in h]
    assert "power" in ops
    assert "sqrt" in ops
    assert "modulo" in ops
