"""
A minimal TODO API — the seed project for DriftBench's demo task chain.
This is intentionally simple so the benchmark focuses on how the agent
EVOLVES the codebase over multiple steps, not on initial complexity.
"""

todos = []
next_id = 1


def add_todo(title: str) -> dict:
    """Add a new todo item."""
    global next_id
    todo = {"id": next_id, "title": title, "done": False}
    todos.append(todo)
    next_id += 1
    return todo


def get_todos() -> list:
    """Get all todo items."""
    return todos


def get_todo(todo_id: int) -> dict | None:
    """Get a single todo by ID."""
    for todo in todos:
        if todo["id"] == todo_id:
            return todo
    return None


def complete_todo(todo_id: int) -> dict | None:
    """Mark a todo as complete."""
    todo = get_todo(todo_id)
    if todo:
        todo["done"] = True
    return todo
