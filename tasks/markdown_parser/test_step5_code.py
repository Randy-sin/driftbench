"""Step 5: Code block support."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def test_inline_code():
    """Inline `code` should become <code>code</code>."""
    from app import render
    result = render("Use `print()` function")
    assert "<code>print()</code>" in result


def test_fenced_code_block():
    """Fenced code blocks should become <pre><code>...</code></pre>."""
    from app import render
    md = "```\nprint('hello')\nprint('world')\n```"
    result = render(md)
    assert "<pre><code>" in result
    assert "print('hello')" in result
    assert "</code></pre>" in result


def test_no_formatting_in_code():
    """Content inside code should not have Markdown formatting applied."""
    from app import render
    result = render("`**not bold**`")
    assert "<strong>" not in result
    assert "<code>**not bold**</code>" in result
