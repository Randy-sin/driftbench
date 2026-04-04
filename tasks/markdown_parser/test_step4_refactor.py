"""Step 4: Refactor to MarkdownParser class with custom patterns."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def test_parser_class_exists():
    """MarkdownParser class should exist with render method."""
    from app import MarkdownParser
    parser = MarkdownParser()
    assert hasattr(parser, 'render')
    assert hasattr(parser, 'register_pattern')


def test_parser_basic_rendering():
    """MarkdownParser should handle all existing features."""
    from app import MarkdownParser
    parser = MarkdownParser()
    result = parser.render("# Hello\n\n**bold** text")
    assert "<h1>Hello</h1>" in result
    assert "<strong>bold</strong>" in result


def test_custom_pattern():
    """register_pattern should add new inline transformations."""
    from app import MarkdownParser
    parser = MarkdownParser()
    parser.register_pattern("strikethrough", r'~~(.+?)~~', r'<del>\1</del>')
    result = parser.render("~~deleted~~")
    assert "<del>deleted</del>" in result


def test_backward_compat():
    """Module-level render() should still work."""
    from app import render
    result = render("# Test")
    assert "<h1>Test</h1>" in result
