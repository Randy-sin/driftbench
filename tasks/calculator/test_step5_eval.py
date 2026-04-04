"""Step 5: Expression evaluator."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pytest


def test_evaluate_simple():
    """evaluate should handle simple arithmetic."""
    from app import Calculator
    calc = Calculator()
    assert calc.evaluate("2 + 3") == 5.0
    assert calc.evaluate("10 - 4") == 6.0


def test_evaluate_order_of_operations():
    """evaluate should respect operator precedence."""
    from app import Calculator
    calc = Calculator()
    assert calc.evaluate("2 + 3 * 4") == 14.0


def test_evaluate_parentheses():
    """evaluate should handle parentheses."""
    from app import Calculator
    calc = Calculator()
    assert calc.evaluate("(2 + 3) * 4") == 20.0


def test_evaluate_in_history():
    """evaluate should record in history."""
    from app import Calculator
    calc = Calculator()
    calc.clear_history()
    calc.evaluate("1 + 1")
    h = calc.get_history()
    assert len(h) == 1
    assert h[0]["operation"] == "evaluate"


def test_evaluate_invalid():
    """Invalid expressions should raise ValueError."""
    from app import Calculator
    calc = Calculator()
    with pytest.raises(ValueError, match="invalid"):
        calc.evaluate("2 +* 3")
