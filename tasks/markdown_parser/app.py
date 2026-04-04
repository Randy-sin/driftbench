"""
A minimal Markdown parser — seed project for DriftBench.
Converts basic Markdown syntax to HTML.
"""


def parse_heading(line: str) -> str:
    """Convert a Markdown heading to HTML."""
    level = 0
    for ch in line:
        if ch == '#':
            level += 1
        else:
            break
    if 1 <= level <= 6:
        text = line[level:].strip()
        return f"<h{level}>{text}</h{level}>"
    return line


def parse_bold(text: str) -> str:
    """Convert **bold** to <strong>bold</strong>."""
    import re
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)


def parse_italic(text: str) -> str:
    """Convert *italic* to <em>italic</em>."""
    import re
    return re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)


def render(markdown: str) -> str:
    """Render Markdown text to HTML."""
    lines = markdown.split('\n')
    html_lines = []
    for line in lines:
        if line.startswith('#'):
            html_lines.append(parse_heading(line))
        else:
            line = parse_bold(line)
            line = parse_italic(line)
            if line.strip():
                html_lines.append(f"<p>{line}</p>")
    return '\n'.join(html_lines)
