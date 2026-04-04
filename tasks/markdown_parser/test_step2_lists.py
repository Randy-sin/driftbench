"""Step 2: Unordered list support."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import render


def test_basic_list():
    """Consecutive '- ' lines should form a <ul> block."""
    md = "- Item 1\n- Item 2\n- Item 3"
    result = render(md)
    assert "<ul>" in result
    assert "<li>Item 1</li>" in result
    assert "<li>Item 2</li>" in result
    assert "<li>Item 3</li>" in result
    assert "</ul>" in result


def test_list_with_inline_formatting():
    """List items should support bold and italic."""
    md = "- **bold item**\n- *italic item*"
    result = render(md)
    assert "<strong>bold item</strong>" in result
    assert "<em>italic item</em>" in result


def test_list_ends_at_non_list_line():
    """Non-list content should end the list block."""
    md = "- Item 1\n- Item 2\nA paragraph"
    result = render(md)
    assert "</ul>" in result
    assert "<p>A paragraph</p>" in result
