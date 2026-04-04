"""
DriftBench Agents — Agent adapters for running evaluations.
Includes a simulated "naive" agent and a real LLM-powered agent.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class NaiveAppendAgent:
    """
    A deliberately naive agent that just appends code to the end of files.
    Used as a BASELINE to show what "bad" looks like on DriftBench.
    It demonstrates the kind of entropy-increasing behavior we want to detect.
    """

    NAIVE_SOLUTIONS = {
        1: '''
# Step 1: Priority support (naive append)
_original_add_todo = add_todo
def add_todo(title: str, priority: str = "medium", **kwargs) -> dict:
    global next_id
    todo = {"id": next_id, "title": title, "done": False, "priority": priority}
    todos.append(todo)
    next_id += 1
    return todo

_original_get_todos = get_todos
def get_todos(sort_by_priority: bool = False, **kwargs) -> list:
    if sort_by_priority:
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(todos, key=lambda t: priority_order.get(t.get("priority", "medium"), 1))
    return todos
''',
        2: '''
# Step 2: Bug fix (naive append)
_original_complete = complete_todo
def complete_todo(todo_id: int) -> dict:
    todo = get_todo(todo_id)
    if todo is None:
        raise ValueError(f"Todo {todo_id} not found")
    if todo["done"]:
        raise ValueError(f"Todo {todo_id} is already completed")
    todo["done"] = True
    return todo
''',
        3: '''
# Step 3: Tags support (naive append - duplicates add_todo again!)
_prev_add_todo = add_todo
def add_todo(title: str, priority: str = "medium", tags: list = None, **kwargs) -> dict:
    global next_id
    if tags is None:
        tags = []
    todo = {"id": next_id, "title": title, "done": False, "priority": priority, "tags": tags}
    todos.append(todo)
    next_id += 1
    return todo

_prev_get_todos = get_todos
def get_todos(sort_by_priority: bool = False, filter_tag: str = None, **kwargs) -> list:
    result = todos
    if filter_tag:
        result = [t for t in result if filter_tag in t.get("tags", [])]
    if sort_by_priority:
        priority_order = {"high": 0, "medium": 1, "low": 2}
        result = sorted(result, key=lambda t: priority_order.get(t.get("priority", "medium"), 1))
    return result
''',
        4: '''
# Step 4: Refactor to class (naive - just wraps everything)
class TodoStore:
    def __init__(self):
        self._todos = []
        self._next_id = 1

    def add_todo(self, title, priority="medium", tags=None):
        if tags is None:
            tags = []
        todo = {"id": self._next_id, "title": title, "done": False, "priority": priority, "tags": tags}
        self._todos.append(todo)
        self._next_id += 1
        return todo

    def get_todos(self, sort_by_priority=False, filter_tag=None):
        result = self._todos
        if filter_tag:
            result = [t for t in result if filter_tag in t.get("tags", [])]
        if sort_by_priority:
            priority_order = {"high": 0, "medium": 1, "low": 2}
            result = sorted(result, key=lambda t: priority_order.get(t.get("priority", "medium"), 1))
        return result

    def get_todo(self, todo_id):
        for todo in self._todos:
            if todo["id"] == todo_id:
                return todo
        return None

    def complete_todo(self, todo_id):
        todo = self.get_todo(todo_id)
        if todo is None:
            raise ValueError(f"Todo {todo_id} not found")
        if todo["done"]:
            raise ValueError(f"Todo {todo_id} is already completed")
        todo["done"] = True
        return todo
''',
        5: '''
# Step 5: Stats (appended to TodoStore - but outside the class, causing issues)
def _get_stats(self):
    from collections import Counter
    total = len(self._todos)
    completed = sum(1 for t in self._todos if t["done"])
    pending = total - completed
    completion_rate = (completed / total * 100) if total > 0 else 0.0

    by_priority = Counter(t.get("priority", "medium") for t in self._todos)
    by_tag = Counter()
    for t in self._todos:
        for tag in t.get("tags", []):
            by_tag[tag] += 1

    return {
        "total": total,
        "completed": completed,
        "pending": pending,
        "completion_rate": completion_rate,
        "by_priority": dict(by_priority),
        "by_tag": dict(by_tag),
    }

TodoStore.get_stats = _get_stats
'''
    }

    def __call__(self, project_dir: str, instruction: str) -> tuple[str, int, int]:
        """Apply naive solution by appending to app.py."""
        app_path = Path(project_dir) / "app.py"

        # Determine which step based on instruction keywords
        step = 0
        if "priority" in instruction.lower() and "tag" not in instruction.lower():
            step = 1
        elif "bug" in instruction.lower() or "fix" in instruction.lower() or "ValueError" in instruction:
            step = 2
        elif "tag" in instruction.lower():
            step = 3
        elif "refactor" in instruction.lower() or "TodoStore" in instruction:
            step = 4
        elif "stats" in instruction.lower() or "statistics" in instruction.lower():
            step = 5

        if step in self.NAIVE_SOLUTIONS:
            with open(app_path, "a") as f:
                f.write("\n" + self.NAIVE_SOLUTIONS[step])
            return f"Applied naive solution for step {step}", 0, 1
        return "No matching solution found", 0, 0


class LLMCodingAgent:
    """
    A real LLM-powered coding agent that reads the codebase,
    understands the instruction, and modifies files accordingly.
    """

    SYSTEM_PROMPT = """You are an expert Python developer. You are given a codebase and an instruction to modify it.

RULES:
1. Read the existing code carefully before making changes.
2. Modify the code to satisfy the instruction.
3. Maintain backward compatibility with existing functionality.
4. Follow the existing code style and patterns.
5. Do NOT add unnecessary complexity.

You must respond with the COMPLETE new content of app.py (the entire file, not just the changes).
Wrap your code in ```python ... ``` markers.
Do NOT include test files in your response. Only modify app.py."""

    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        if HAS_OPENAI:
            self.client = OpenAI()
        else:
            raise RuntimeError("OpenAI package not installed")

    def __call__(self, project_dir: str, instruction: str) -> tuple[str, int, int]:
        """Use LLM to modify the codebase."""
        app_path = Path(project_dir) / "app.py"
        current_code = app_path.read_text()

        prompt = f"""Here is the current app.py:

```python
{current_code}
```

INSTRUCTION: {instruction}

Please provide the complete updated app.py that satisfies this instruction while maintaining all existing functionality."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        reply = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0

        # Extract code from response
        code = self._extract_code(reply)
        if code:
            app_path.write_text(code)
            return reply[:500], tokens, 1
        else:
            return f"Failed to extract code from response: {reply[:200]}", tokens, 1

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        """Extract Python code from markdown code blocks."""
        if "```python" in text:
            parts = text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) > 2:
                code = parts[1]
                if code.startswith("\n"):
                    code = code[1:]
                return code.strip()
        return None
