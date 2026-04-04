"""Step 4: Refactor to Calculator class."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def test_calculator_class_exists():
    """A Calculator class should exist with all methods."""
    from app import Calculator
    calc = Calculator()
    assert hasattr(calc, 'add')
    assert hasattr(calc, 'subtract')
    assert hasattr(calc, 'multiply')
    assert hasattr(calc, 'divide')
    assert hasattr(calc, 'power')
    assert hasattr(calc, 'sqrt')
    assert hasattr(calc, 'modulo')
    assert hasattr(calc, 'get_history')
    assert hasattr(calc, 'clear_history')


def test_calculator_isolation():
    """Two Calculator instances should have independent history."""
    from app import Calculator
    c1 = Calculator()
    c2 = Calculator()
    c1.add(1, 2)
    assert len(c1.get_history()) == 1
    assert len(c2.get_history()) == 0


def test_calculator_preserves_features():
    """Calculator should support all previously added features."""
    from app import Calculator
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.power(2, 3) == 8
    assert calc.sqrt(16) == 4.0
    assert calc.modulo(10, 3) == 1
    import pytest
    with pytest.raises(ValueError, match="negative"):
        calc.sqrt(-1)
