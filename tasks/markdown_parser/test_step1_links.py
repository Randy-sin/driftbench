"""Step 1: Add link and image support."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app import render


def test_link():
    """[text](url) should become <a href='url'>text</a>."""
    result = render("[Click here](https://example.com)")
    assert '<a href="https://example.com">Click here</a>' in result


def test_image():
    """![alt](src) should become <img src='src' alt='alt' />."""
    result = render("![Logo](logo.png)")
    assert '<img src="logo.png" alt="Logo" />' in result


def test_link_with_bold():
    """Links should work alongside bold text."""
    result = render("**bold** and [link](url)")
    assert "<strong>bold</strong>" in result
    assert '<a href="url">link</a>' in result
