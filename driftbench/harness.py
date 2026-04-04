"""
DriftBench Harness — The core evaluation harness that orchestrates
task chain execution and environment isolation for coding agents.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional


@dataclass
class TaskStep:
    """A single step in a task chain."""
    step_id: int
    instruction: str
    task_type: str  # "feature", "bugfix", "refactor", "evolution"
    test_file: str  # path to the test file for this step
    timeout: int = 120  # seconds


@dataclass
class StepResult:
    """Result of executing a single task step."""
    step_id: int
    task_type: str
    passed: bool
    regression_failures: int  # how many previous tests broke
    total_previous_tests: int
    new_test_passed: bool
    agent_output: str
    duration_seconds: float
    token_count: int = 0
    action_count: int = 0


@dataclass
class TrialResult:
    """Aggregated result of an entire task chain trial."""
    agent_name: str
    task_chain_name: str
    step_results: list = field(default_factory=list)
    # Metrics
    pass_rate: float = 0.0
    regression_rate: float = 0.0
    entropy_delta: dict = field(default_factory=dict)
    consistency_score: float = 0.0
    total_tokens: int = 0
    total_actions: int = 0
    total_duration: float = 0.0

    def compute_metrics(self):
        if not self.step_results:
            return
        passed = sum(1 for s in self.step_results if s.passed)
        self.pass_rate = passed / len(self.step_results)

        total_prev = sum(s.total_previous_tests for s in self.step_results)
        total_reg = sum(s.regression_failures for s in self.step_results)
        self.regression_rate = total_reg / max(total_prev, 1)

        self.total_tokens = sum(s.token_count for s in self.step_results)
        self.total_actions = sum(s.action_count for s in self.step_results)
        self.total_duration = sum(s.duration_seconds for s in self.step_results)


class DriftBenchHarness:
    """
    The main harness that:
    1. Sets up an isolated sandbox for each trial
    2. Feeds task steps sequentially to the agent
    3. Runs tests after each step (new + all previous)
    4. Collects metrics (regression, entropy, etc.)
    """

    def __init__(self, base_project_dir: str, task_chain: list[TaskStep],
                 agent_fn: Callable[[str, str], tuple[str, int, int]],
                 agent_name: str = "unknown"):
        """
        Args:
            base_project_dir: Path to the seed project (will be copied into sandbox)
            task_chain: Ordered list of TaskStep objects
            agent_fn: A callable that takes (project_dir, instruction) and returns
                       (patch_or_output, token_count, action_count).
                       The agent should modify files in project_dir in-place.
            agent_name: Name of the agent being evaluated
        """
        self.base_project_dir = Path(base_project_dir)
        self.task_chain = task_chain
        self.agent_fn = agent_fn
        self.agent_name = agent_name
        self.sandbox_dir: Optional[Path] = None

    def _create_sandbox(self) -> Path:
        """Create an isolated copy of the base project."""
        sandbox = Path(tempfile.mkdtemp(prefix="driftbench_"))
        project_copy = sandbox / "project"
        shutil.copytree(self.base_project_dir, project_copy)
        return sandbox

    def _run_tests(self, project_dir: Path, test_files: list[str]) -> dict:
        """
        Run a list of test files and return results.
        Returns: {"passed": int, "failed": int, "errors": list[str]}
        """
        passed = 0
        failed = 0
        errors = []

        for tf in test_files:
            test_path = project_dir / tf
            if not test_path.exists():
                errors.append(f"Test file not found: {tf}")
                failed += 1
                continue

            try:
                result = subprocess.run(
                    ["python3", "-m", "pytest", str(test_path), "-v", "--tb=short"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(project_dir)
                )
                if result.returncode == 0:
                    passed += 1
                else:
                    failed += 1
                    errors.append(f"{tf}: {result.stdout[-500:]}")
            except subprocess.TimeoutExpired:
                failed += 1
                errors.append(f"{tf}: TIMEOUT")
            except Exception as e:
                failed += 1
                errors.append(f"{tf}: {str(e)}")

        return {"passed": passed, "failed": failed, "errors": errors}

    def _compute_complexity(self, project_dir: Path) -> dict:
        """
        Compute code complexity metrics using radon.
        Returns: {"avg_cc": float, "total_loc": int, "num_functions": int}
        """
        try:
            result = subprocess.run(
                ["python3", "-m", "radon", "cc", str(project_dir), "-a", "-j"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                total_cc = 0
                num_funcs = 0
                for filepath, blocks in data.items():
                    if isinstance(blocks, list):
                        for block in blocks:
                            if isinstance(block, dict) and "complexity" in block:
                                total_cc += block["complexity"]
                                num_funcs += 1
                avg_cc = total_cc / max(num_funcs, 1)
                return {"avg_cc": round(avg_cc, 2), "num_functions": num_funcs}
        except Exception:
            pass
        return {"avg_cc": 0, "num_functions": 0}

    def _count_duplicate_lines(self, project_dir: Path) -> float:
        """Simple duplicate line detection ratio."""
        all_lines = []
        for py_file in project_dir.rglob("*.py"):
            if "test_" in py_file.name or "__pycache__" in str(py_file):
                continue
            try:
                lines = py_file.read_text().splitlines()
                # Only count non-trivial lines
                meaningful = [l.strip() for l in lines
                              if l.strip() and not l.strip().startswith("#")
                              and len(l.strip()) > 10]
                all_lines.extend(meaningful)
            except Exception:
                continue

        if not all_lines:
            return 0.0

        from collections import Counter
        counts = Counter(all_lines)
        duplicated = sum(c - 1 for c in counts.values() if c > 1)
        return round(duplicated / max(len(all_lines), 1), 4)

    def run_trial(self) -> TrialResult:
        """Execute the full task chain and collect results."""
        sandbox = self._create_sandbox()
        self.sandbox_dir = sandbox
        project_dir = sandbox / "project"

        # Compute baseline complexity
        baseline_complexity = self._compute_complexity(project_dir)
        baseline_duplication = self._count_duplicate_lines(project_dir)

        trial = TrialResult(
            agent_name=self.agent_name,
            task_chain_name=self.task_chain[0].instruction[:50] if self.task_chain else "empty"
        )

        completed_test_files: list[str] = []

        for step in self.task_chain:
            print(f"\n{'='*60}")
            print(f"  Step {step.step_id}: [{step.task_type.upper()}]")
            print(f"  {step.instruction[:80]}...")
            print(f"{'='*60}")

            # --- Execute agent ---
            start_time = time.time()
            try:
                agent_output, tokens, actions = self.agent_fn(
                    str(project_dir), step.instruction
                )
            except Exception as e:
                agent_output = f"AGENT ERROR: {str(e)}"
                tokens, actions = 0, 0
            duration = time.time() - start_time

            # --- Run new test ---
            new_test_result = self._run_tests(project_dir, [step.test_file])
            new_test_passed = new_test_result["passed"] > 0

            # --- Run all previous tests (regression check) ---
            regression_failures = 0
            total_previous = len(completed_test_files)
            if completed_test_files:
                reg_result = self._run_tests(project_dir, completed_test_files)
                regression_failures = reg_result["failed"]

            step_passed = new_test_passed and regression_failures == 0

            step_result = StepResult(
                step_id=step.step_id,
                task_type=step.task_type,
                passed=step_passed,
                regression_failures=regression_failures,
                total_previous_tests=total_previous,
                new_test_passed=new_test_passed,
                agent_output=agent_output[:1000],
                duration_seconds=round(duration, 2),
                token_count=tokens,
                action_count=actions
            )
            trial.step_results.append(step_result)

            status = "PASS" if step_passed else "FAIL"
            reg_info = f"(regressions: {regression_failures}/{total_previous})" if total_previous > 0 else ""
            print(f"  Result: {status} {reg_info}")

            if new_test_passed:
                completed_test_files.append(step.test_file)

        # --- Compute final metrics ---
        final_complexity = self._compute_complexity(project_dir)
        final_duplication = self._count_duplicate_lines(project_dir)

        trial.entropy_delta = {
            "cc_before": baseline_complexity.get("avg_cc", 0),
            "cc_after": final_complexity.get("avg_cc", 0),
            "cc_delta": round(final_complexity.get("avg_cc", 0) - baseline_complexity.get("avg_cc", 0), 2),
            "duplication_before": baseline_duplication,
            "duplication_after": final_duplication,
            "duplication_delta": round(final_duplication - baseline_duplication, 4),
        }

        trial.compute_metrics()
        return trial

    def cleanup(self):
        """Remove sandbox directory."""
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
