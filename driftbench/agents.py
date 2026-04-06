"""
DriftBench Agents — Agent adapters for running evaluations.

Key Features:
- ReAct-style multi-turn agent with tool use (read, write, run_tests, grep)
- Error feedback loop: agent receives test output and iterates
- Conversation history tracking for context management
- Configurable max_turns to cap interaction budget
- Structured tool-call parsing with fallback
- Token and action counting across turns
- Backward-compatible single-shot mode
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ── Naive Baseline Agent ─────────────────────────────────────────

class NaiveAppendAgent:
    """
    A deliberately naive agent that just appends code to the end of files.
    Used as a BASELINE to show what "bad" looks like on DriftBench.
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


# ── Single-Shot LLM Agent  ─────────────────────

class LLMCodingAgent:
    """
    Single-shot LLM agent: reads code, generates full replacement.
    Kept for backward compatibility and as a baseline.
    """

    SYSTEM_PROMPT = """You are an expert Python developer. You are given a codebase and an instruction to modify it.

RULES:
1. Read the existing code carefully before making changes.
2. Modify the code to satisfy the instruction.
3. Maintain backward compatibility with existing functionality.
4. Follow the existing code style and patterns.
5. Do NOT add unnecessary complexity.
6. When refactoring, ensure ALL existing functionality continues to work.
7. Prefer clean, well-structured code over quick patches.

You must respond with the COMPLETE new content of app.py (the entire file, not just the changes).
Wrap your code in ```python ... ``` markers.
Do NOT include test files in your response. Only modify app.py."""

    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        if HAS_OPENAI:
            self.client = OpenAI()
        else:
            raise RuntimeError("OpenAI package not installed")

    def __call__(self, project_dir: str, instruction: str) -> tuple[str, int, int]:
        """Use LLM to modify the codebase (single-shot)."""
        app_path = Path(project_dir) / "app.py"
        current_code = app_path.read_text()

        prompt = f"""Here is the current app.py:

```python
{current_code}
```

INSTRUCTION: {instruction}

Please provide the complete updated app.py that satisfies this instruction while maintaining all existing functionality."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )

            reply = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0

            code = self._extract_code(reply)
            if code:
                app_path.write_text(code)
                return reply[:500], tokens, 1
            else:
                return f"Failed to extract code from response: {reply[:200]}", tokens, 1
        except Exception as e:
            return f"LLM API error: {str(e)}", 0, 1

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


# ── ReAct Multi-Turn Agent ───────────────────────────

class ReActCodingAgent:
    """
    A ReAct-style multi-turn coding agent with tool use capabilities.

    Unlike the single-shot LLMCodingAgent, this agent can:
    1. Read files to understand the codebase
    2. Write/modify files incrementally
    3. Run Python scripts to verify behavior
    4. Search code with grep
    5. Iterate based on execution results

    This more closely simulates how real coding agents (Devin, Cursor, etc.)
    operate, making the benchmark more realistic.
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a file in the project directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to read (e.g., 'app.py')"
                        }
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write complete content to a file. This overwrites the entire file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to write (e.g., 'app.py')"
                        },
                        "content": {
                            "type": "string",
                            "description": "The complete file content to write"
                        }
                    },
                    "required": ["filename", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Run a Python command or script and return stdout/stderr. Use this to test your changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Python code or command to run (e.g., 'python3 -c \"import app; print(app.add_todo(\\\"test\\\"))\"')"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep_code",
                "description": "Search for a pattern in all Python files in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for"
                        }
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Signal that you have completed the task. Call this when you are confident your changes are correct.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of what you changed and why"
                        }
                    },
                    "required": ["summary"]
                }
            }
        }
    ]

    SYSTEM_PROMPT = """You are an expert Python developer working on a coding task. You have access to tools to read files, write files, run Python code, and search the codebase.

WORKFLOW:
1. First, read the existing code to understand the current state.
2. Plan your changes carefully, considering backward compatibility.
3. Write the modified code.
4. Optionally run a quick sanity check.
5. Call 'done' when you are confident.

RULES:
- ALWAYS read the file before modifying it.
- Maintain backward compatibility with ALL existing functionality.
- When refactoring, ensure module-level functions still work.
- Follow the existing code style.
- Do NOT add unnecessary complexity.
- Do NOT create new files unless absolutely necessary.
- When you write a file, provide the COMPLETE content (not just changes).
- If you encounter errors, analyze them and fix your code."""

    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.2,
                 max_turns: int = 10):
        self.model = model
        self.temperature = temperature
        self.max_turns = max_turns
        if HAS_OPENAI:
            self.client = OpenAI()
        else:
            raise RuntimeError("OpenAI package not installed")

    def _execute_tool(self, project_dir: str, tool_name: str, args: dict) -> str:
        """Execute a tool call and return the result string."""
        project_path = Path(project_dir)

        if tool_name == "read_file":
            filename = args.get("filename", "app.py")
            filepath = project_path / filename
            if not filepath.exists():
                return f"Error: File '{filename}' not found. Available files: {[f.name for f in project_path.glob('*.py') if 'test_' not in f.name]}"
            try:
                content = filepath.read_text()
                return f"Content of {filename}:\n```python\n{content}\n```"
            except Exception as e:
                return f"Error reading {filename}: {str(e)}"

        elif tool_name == "write_file":
            filename = args.get("filename", "app.py")
            content = args.get("content", "")
            filepath = project_path / filename
            # Security: only allow writing to project directory
            if ".." in filename or filename.startswith("/"):
                return "Error: Invalid filename. Must be a relative path within the project."
            # Only allow .py files
            if not filename.endswith(".py"):
                return "Error: Only .py files can be written."
            try:
                filepath.write_text(content)
                return f"Successfully wrote {len(content)} characters to {filename}."
            except Exception as e:
                return f"Error writing {filename}: {str(e)}"

        elif tool_name == "run_python":
            command = args.get("command", "")
            try:
                result = subprocess.run(
                    command, shell=True,
                    capture_output=True, text=True, timeout=30,
                    cwd=str(project_path)
                )
                output = ""
                if result.stdout:
                    output += f"STDOUT:\n{result.stdout[-1000:]}\n"
                if result.stderr:
                    output += f"STDERR:\n{result.stderr[-1000:]}\n"
                output += f"Exit code: {result.returncode}"
                return output if output.strip() else "Command completed with no output."
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 30 seconds."
            except Exception as e:
                return f"Error running command: {str(e)}"

        elif tool_name == "grep_code":
            pattern = args.get("pattern", "")
            try:
                result = subprocess.run(
                    ["grep", "-rn", pattern, "--include=*.py", "."],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(project_path)
                )
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    # Filter out test files
                    lines = [l for l in lines if 'test_' not in l]
                    return "\n".join(lines[:30]) if lines else "No matches found."
                return "No matches found."
            except Exception as e:
                return f"Error searching: {str(e)}"

        elif tool_name == "done":
            return "TASK_COMPLETE"

        return f"Unknown tool: {tool_name}"

    def __call__(self, project_dir: str, instruction: str) -> tuple[str, int, int]:
        """
        Multi-turn ReAct agent execution.
        Returns (summary, total_tokens, total_actions).
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"TASK: {instruction}\n\nStart by reading the current code, then make the necessary changes."}
        ]

        total_tokens = 0
        total_actions = 0
        summary = ""

        for turn in range(self.max_turns):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.TOOLS,
                    tool_choice="auto" if turn < self.max_turns - 1 else "none",
                    temperature=self.temperature,
                )

                if response.usage:
                    total_tokens += response.usage.total_tokens

                choice = response.choices[0]
                assistant_msg = choice.message

                # Add assistant message to history
                messages.append(assistant_msg)

                # Check if the model wants to call tools
                if assistant_msg.tool_calls:
                    for tool_call in assistant_msg.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            tool_args = {}

                        total_actions += 1

                        if tool_name == "done":
                            summary = tool_args.get("summary", "Task completed")
                            return summary, total_tokens, total_actions

                        result = self._execute_tool(project_dir, tool_name, tool_args)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                else:
                    # No tool calls — model is done or providing final response
                    if assistant_msg.content:
                        summary = assistant_msg.content[:500]

                        # Check if the model included code in its response
                        # (fallback for models that don't use tools properly)
                        code = LLMCodingAgent._extract_code(assistant_msg.content)
                        if code:
                            app_path = Path(project_dir) / "app.py"
                            app_path.write_text(code)
                            total_actions += 1

                    break

            except Exception as e:
                summary = f"Agent error on turn {turn}: {str(e)}"
                break

        return summary, total_tokens, total_actions


# ── Agent Factory ────────────────────────────────────────────────

def create_agent(agent_type: str, model: str = "gpt-4.1-mini",
                 temperature: float = 0.2, max_turns: int = 10):
    """
    Factory function to create agents by type.

    Args:
        agent_type: One of "naive", "single-shot", "react"
        model: LLM model name
        temperature: Sampling temperature
        max_turns: Max interaction turns (for react agent)

    Returns:
        Callable agent function
    """
    if agent_type == "naive":
        return NaiveAppendAgent()
    elif agent_type == "single-shot":
        return LLMCodingAgent(model=model, temperature=temperature)
    elif agent_type == "react":
        return ReActCodingAgent(model=model, temperature=temperature,
                                max_turns=max_turns)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. "
                         f"Choose from: naive, single-shot, react")
