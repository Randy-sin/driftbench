"""
DriftBench Harness — The core evaluation harness that orchestrates
task chain execution and environment isolation for coding agents.

v2.0 Improvements:
- Per-test-case regression counting (not per-file)
- Test suite isolation (hidden from agent sandbox)
- Per-step complexity snapshots (entropy trajectory)
- Structural Erosion metric (SlopCodeBench-inspired)
- pass@k / pass^k multi-trial support
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
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
class ComplexitySnapshot:
    """Code quality metrics captured at a specific step."""
    step_id: int
    avg_cc: float = 0.0
    max_cc: float = 0.0
    num_functions: int = 0
    total_loc: int = 0
    duplication_ratio: float = 0.0
    structural_erosion: float = 0.0  # fraction of CC mass in top-heavy functions


@dataclass
class StepResult:
    """Result of executing a single task step."""
    step_id: int
    task_type: str
    passed: bool
    regression_failures: int  # how many previous TEST CASES broke
    total_previous_tests: int  # total previous test CASES (not files)
    new_tests_passed: int  # number of new test cases passed
    new_tests_total: int  # total new test cases
    new_test_passed: bool  # backward compat: did all new tests pass?
    agent_output: str
    duration_seconds: float
    token_count: int = 0
    action_count: int = 0
    complexity_snapshot: Optional[ComplexitySnapshot] = None


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
    entropy_trajectory: list = field(default_factory=list)  # per-step snapshots
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

        # Build entropy trajectory
        self.entropy_trajectory = []
        for sr in self.step_results:
            if sr.complexity_snapshot:
                self.entropy_trajectory.append({
                    "step_id": sr.step_id,
                    "avg_cc": sr.complexity_snapshot.avg_cc,
                    "max_cc": sr.complexity_snapshot.max_cc,
                    "structural_erosion": sr.complexity_snapshot.structural_erosion,
                    "duplication_ratio": sr.complexity_snapshot.duplication_ratio,
                    "total_loc": sr.complexity_snapshot.total_loc,
                })


class DriftBenchHarness:
    """
    The main harness that:
    1. Sets up an isolated sandbox for each trial
    2. Hides test files from the agent (test isolation)
    3. Feeds task steps sequentially to the agent
    4. Runs tests after each step with per-case granularity
    5. Collects per-step complexity snapshots (entropy trajectory)
    """

    def __init__(self, base_project_dir: str, task_chain: list[TaskStep],
                 agent_fn: Callable[[str, str], tuple[str, int, int]],
                 agent_name: str = "unknown"):
        self.base_project_dir = Path(base_project_dir)
        self.task_chain = task_chain
        self.agent_fn = agent_fn
        self.agent_name = agent_name
        self.sandbox_dir: Optional[Path] = None
        self._test_vault: Optional[Path] = None  # hidden test storage

    def _create_sandbox(self) -> Path:
        """Create an isolated sandbox. Test files are stored separately."""
        sandbox = Path(tempfile.mkdtemp(prefix="driftbench_"))
        project_copy = sandbox / "project"
        test_vault = sandbox / "_test_vault"

        # Copy project files (excluding tests)
        shutil.copytree(self.base_project_dir, project_copy)

        # Move test files to vault (hidden from agent)
        test_vault.mkdir(parents=True, exist_ok=True)
        for test_file in project_copy.glob("test_*.py"):
            shutil.move(str(test_file), str(test_vault / test_file.name))

        self._test_vault = test_vault
        return sandbox

    def _run_tests_granular(self, project_dir: Path, test_files: list[str]) -> dict:
        """
        Run tests with per-test-case granularity using pytest JSON output.
        Returns: {
            "cases_passed": int,
            "cases_failed": int,
            "cases_total": int,
            "failed_cases": list[str],
            "errors": list[str]
        }
        """
        cases_passed = 0
        cases_failed = 0
        failed_cases = []
        errors = []

        for tf in test_files:
            # Copy test file from vault to project temporarily
            vault_path = self._test_vault / tf
            project_test_path = project_dir / tf

            if not vault_path.exists():
                errors.append(f"Test file not found in vault: {tf}")
                cases_failed += 1
                continue

            # Temporarily inject test file
            shutil.copy2(str(vault_path), str(project_test_path))

            try:
                result = subprocess.run(
                    ["python3", "-m", "pytest", str(project_test_path),
                     "-v", "--tb=short", "-q"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(project_dir)
                )

                # Parse pytest verbose output for per-case results
                stdout = result.stdout
                # Count PASSED and FAILED from pytest -v output
                passed_in_file = len(re.findall(r'PASSED', stdout))
                failed_in_file = len(re.findall(r'FAILED', stdout))

                # If no PASSED/FAILED markers found, fall back to return code
                if passed_in_file == 0 and failed_in_file == 0:
                    if result.returncode == 0:
                        passed_in_file = 1
                    else:
                        failed_in_file = 1

                cases_passed += passed_in_file
                cases_failed += failed_in_file

                if failed_in_file > 0:
                    # Extract failed test names
                    for line in stdout.split('\n'):
                        if 'FAILED' in line:
                            failed_cases.append(line.strip())

            except subprocess.TimeoutExpired:
                cases_failed += 1
                errors.append(f"{tf}: TIMEOUT")
            except Exception as e:
                cases_failed += 1
                errors.append(f"{tf}: {str(e)}")
            finally:
                # Remove test file from agent sandbox
                if project_test_path.exists():
                    project_test_path.unlink()

        return {
            "cases_passed": cases_passed,
            "cases_failed": cases_failed,
            "cases_total": cases_passed + cases_failed,
            "failed_cases": failed_cases,
            "errors": errors,
        }

    def _compute_complexity(self, project_dir: Path) -> ComplexitySnapshot:
        """
        Compute comprehensive code complexity metrics.
        Includes Structural Erosion (inspired by SlopCodeBench).
        """
        snapshot = ComplexitySnapshot(step_id=0)

        try:
            result = subprocess.run(
                ["python3", "-m", "radon", "cc", str(project_dir), "-a", "-j"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                cc_values = []
                for filepath, blocks in data.items():
                    if isinstance(blocks, list):
                        for block in blocks:
                            if isinstance(block, dict) and "complexity" in block:
                                cc_values.append(block["complexity"])

                if cc_values:
                    snapshot.avg_cc = round(sum(cc_values) / len(cc_values), 2)
                    snapshot.max_cc = max(cc_values)
                    snapshot.num_functions = len(cc_values)

                    # Structural Erosion: fraction of total CC mass
                    # concentrated in functions with CC > median
                    if len(cc_values) >= 2:
                        sorted_cc = sorted(cc_values, reverse=True)
                        total_mass = sum(sorted_cc)
                        median_cc = sorted_cc[len(sorted_cc) // 2]
                        heavy_mass = sum(c for c in sorted_cc if c > median_cc)
                        snapshot.structural_erosion = round(
                            heavy_mass / max(total_mass, 1), 4
                        )
        except Exception:
            pass

        # Count LOC
        try:
            total_loc = 0
            for py_file in project_dir.rglob("*.py"):
                if "test_" in py_file.name or "__pycache__" in str(py_file):
                    continue
                total_loc += len(py_file.read_text().splitlines())
            snapshot.total_loc = total_loc
        except Exception:
            pass

        # Duplication ratio
        snapshot.duplication_ratio = self._count_duplicate_lines(project_dir)

        return snapshot

    def _count_duplicate_lines(self, project_dir: Path) -> float:
        """Duplicate line detection ratio."""
        all_lines = []
        for py_file in project_dir.rglob("*.py"):
            if "test_" in py_file.name or "__pycache__" in str(py_file):
                continue
            try:
                lines = py_file.read_text().splitlines()
                meaningful = [l.strip() for l in lines
                              if l.strip() and not l.strip().startswith("#")
                              and len(l.strip()) > 10]
                all_lines.extend(meaningful)
            except Exception:
                continue

        if not all_lines:
            return 0.0

        counts = Counter(all_lines)
        duplicated = sum(c - 1 for c in counts.values() if c > 1)
        return round(duplicated / max(len(all_lines), 1), 4)

    def run_trial(self) -> TrialResult:
        """Execute the full task chain and collect results."""
        sandbox = self._create_sandbox()
        self.sandbox_dir = sandbox
        project_dir = sandbox / "project"

        # Compute baseline complexity
        baseline = self._compute_complexity(project_dir)
        baseline.step_id = 0

        trial = TrialResult(
            agent_name=self.agent_name,
            task_chain_name=self.task_chain[0].instruction[:50] if self.task_chain else "empty"
        )

        completed_test_files: list[str] = []
        # Track total previous test cases (not files)
        previous_case_count = 0

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

            # --- Run new test (granular) ---
            new_result = self._run_tests_granular(project_dir, [step.test_file])
            new_tests_passed = new_result["cases_passed"]
            new_tests_total = new_result["cases_total"]
            new_test_passed = new_result["cases_failed"] == 0 and new_tests_passed > 0

            # --- Run all previous tests (regression check, granular) ---
            regression_failures = 0
            total_previous = previous_case_count
            if completed_test_files:
                reg_result = self._run_tests_granular(project_dir, completed_test_files)
                regression_failures = reg_result["cases_failed"]
                # Update actual previous case count
                total_previous = reg_result["cases_total"]

            step_passed = new_test_passed and regression_failures == 0

            # --- Compute per-step complexity snapshot ---
            step_snapshot = self._compute_complexity(project_dir)
            step_snapshot.step_id = step.step_id

            step_result = StepResult(
                step_id=step.step_id,
                task_type=step.task_type,
                passed=step_passed,
                regression_failures=regression_failures,
                total_previous_tests=total_previous,
                new_tests_passed=new_tests_passed,
                new_tests_total=new_tests_total,
                new_test_passed=new_test_passed,
                agent_output=agent_output[:1000],
                duration_seconds=round(duration, 2),
                token_count=tokens,
                action_count=actions,
                complexity_snapshot=step_snapshot,
            )
            trial.step_results.append(step_result)

            status = "PASS" if step_passed else "FAIL"
            reg_info = f"(regressions: {regression_failures}/{total_previous} cases)" if total_previous > 0 else ""
            new_info = f"(new: {new_tests_passed}/{new_tests_total} cases)"
            cc_info = f"[CC={step_snapshot.avg_cc}, erosion={step_snapshot.structural_erosion}]"
            print(f"  Result: {status} {new_info} {reg_info} {cc_info}")

            if new_test_passed:
                completed_test_files.append(step.test_file)
                previous_case_count += new_tests_passed

        # --- Compute final metrics ---
        final = self._compute_complexity(project_dir)
        final.step_id = len(self.task_chain)

        trial.entropy_delta = {
            "cc_before": baseline.avg_cc,
            "cc_after": final.avg_cc,
            "cc_delta": round(final.avg_cc - baseline.avg_cc, 2),
            "max_cc_before": baseline.max_cc,
            "max_cc_after": final.max_cc,
            "erosion_before": baseline.structural_erosion,
            "erosion_after": final.structural_erosion,
            "erosion_delta": round(final.structural_erosion - baseline.structural_erosion, 4),
            "duplication_before": baseline.duplication_ratio,
            "duplication_after": final.duplication_ratio,
            "duplication_delta": round(final.duplication_ratio - baseline.duplication_ratio, 4),
            "loc_before": baseline.total_loc,
            "loc_after": final.total_loc,
        }

        trial.compute_metrics()
        return trial

    def cleanup(self):
        """Remove sandbox directory."""
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
