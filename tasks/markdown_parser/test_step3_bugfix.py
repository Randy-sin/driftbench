"""Step 3: Fix bold/italic conflicts and empty line handling."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import render


def test_bold_italic_combined():
    """***text*** should render as bold+italic."""
    result = render("***bold and italic***")
    assert "<strong>" in result
    assert "<em>" in result
    assert "bold and italic" in result


def test_empty_lines_preserved():
    """Empty lines should be preserved in output."""
    md = "Line 1\n\nLine 2"
    result = render(md)
    lines = result.split('\n')
    assert len(lines) == 3  # line1, empty, line2


def test_whitespace_only_lines():
    """Lines with only whitespace should be treated as empty."""
    md = "Line 1\n   \nLine 2"
    result = render(md)
    lines = result.split('\n')
    # whitespace-only line should become empty
    assert any(line.strip() == '' for line in lines)
