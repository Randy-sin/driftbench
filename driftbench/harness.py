"""
DriftBench Harness — The core evaluation harness that orchestrates
task chain execution and environment isolation for coding agents.

Key Features:
- Syntax validation: detects unparseable code and penalizes instead of rewarding
- Multi-trial support with pass@k / pass^k computation
- Error feedback loop: agents receive test failure info for retry attempts
- Per-step code snapshots for diff-based grading
- Robust complexity metrics with validity checks
- Effective regression rate (avoids "do nothing" loophole)
"""

import ast
import json
import math
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


# ── Data Classes ─────────────────────────────────────────────────

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
    structural_erosion: float = 0.0
    syntax_valid: bool = True       # NEW: whether code is syntactically valid
    avg_function_length: float = 0.0  # NEW: average lines per function


@dataclass
class StepResult:
    """Result of executing a single task step."""
    step_id: int
    task_type: str
    passed: bool
    regression_failures: int
    total_previous_tests: int
    new_tests_passed: int
    new_tests_total: int
    new_test_passed: bool
    agent_output: str
    duration_seconds: float
    token_count: int = 0
    action_count: int = 0
    complexity_snapshot: Optional[ComplexitySnapshot] = None
    code_snapshot: str = ""          # NEW: full code at this step for diff grading
    test_error_output: str = ""      # NEW: test failure details for error feedback
    retry_count: int = 0             # NEW: how many retries were used


@dataclass
class TrialResult:
    """Aggregated result of an entire task chain trial."""
    agent_name: str
    task_chain_name: str
    trial_id: int = 0
    step_results: list = field(default_factory=list)
    # Metrics
    pass_rate: float = 0.0
    regression_rate: float = 0.0
    effective_regression_rate: float = 0.0  # NEW: avoids "do nothing" loophole
    entropy_delta: dict = field(default_factory=dict)
    entropy_trajectory: list = field(default_factory=list)
    consistency_score: float = 0.0
    total_tokens: int = 0
    total_actions: int = 0
    total_duration: float = 0.0
    syntax_error_count: int = 0      # NEW: steps with syntax errors
    code_snapshots: list = field(default_factory=list)  # NEW: per-step code

    def compute_metrics(self):
        if not self.step_results:
            return
        passed = sum(1 for s in self.step_results if s.passed)
        self.pass_rate = passed / len(self.step_results)

        # Standard regression rate
        total_prev = sum(s.total_previous_tests for s in self.step_results)
        total_reg = sum(s.regression_failures for s in self.step_results)
        self.regression_rate = total_reg / max(total_prev, 1)

        # Effective regression rate: only count steps where agent had
        # accumulated at least 1 previous test (avoids "do nothing" loophole)
        effective_prev = 0
        effective_reg = 0
        for s in self.step_results:
            if s.total_previous_tests > 0:
                effective_prev += s.total_previous_tests
                effective_reg += s.regression_failures
        self.effective_regression_rate = effective_reg / max(effective_prev, 1)

        self.total_tokens = sum(s.token_count for s in self.step_results)
        self.total_actions = sum(s.action_count for s in self.step_results)
        self.total_duration = sum(s.duration_seconds for s in self.step_results)

        # Count syntax errors
        self.syntax_error_count = sum(
            1 for s in self.step_results
            if s.complexity_snapshot and not s.complexity_snapshot.syntax_valid
        )

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
                    "syntax_valid": sr.complexity_snapshot.syntax_valid,
                    "num_functions": sr.complexity_snapshot.num_functions,
                    "avg_function_length": sr.complexity_snapshot.avg_function_length,
                })

        # Collect code snapshots
        self.code_snapshots = [sr.code_snapshot for sr in self.step_results]


@dataclass
class MultiTrialResult:
    """Aggregated result across multiple trials for pass@k computation."""
    agent_name: str
    task_chain_name: str
    trials: list = field(default_factory=list)
    num_trials: int = 0

    # pass@k metrics
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0

    # Aggregated metrics (mean +/- std)
    mean_pass_rate: float = 0.0
    std_pass_rate: float = 0.0
    mean_regression_rate: float = 0.0
    std_regression_rate: float = 0.0

    # Best trial
    best_trial: Optional[TrialResult] = None

    def compute_aggregate(self):
        """Compute aggregate statistics across all trials."""
        if not self.trials:
            return
        self.num_trials = len(self.trials)

        pass_rates = [t.pass_rate for t in self.trials]
        reg_rates = [t.regression_rate for t in self.trials]

        self.mean_pass_rate = sum(pass_rates) / len(pass_rates)
        self.mean_regression_rate = sum(reg_rates) / len(reg_rates)

        if len(pass_rates) > 1:
            mean_pr = self.mean_pass_rate
            self.std_pass_rate = math.sqrt(
                sum((p - mean_pr) ** 2 for p in pass_rates) / (len(pass_rates) - 1)
            )
            mean_rr = self.mean_regression_rate
            self.std_regression_rate = math.sqrt(
                sum((r - mean_rr) ** 2 for r in reg_rates) / (len(reg_rates) - 1)
            )

        # pass@k: unbiased estimator (Chen et al. 2021, Codex paper)
        n = self.num_trials
        c = sum(1 for t in self.trials if t.pass_rate == 1.0)
        self.pass_at_1 = c / n if n > 0 else 0.0
        self.pass_at_k = (1.0 - _comb_ratio(n - c, n, min(n, n))) if c > 0 else 0.0

        # Best trial by pass_rate, then lowest regression_rate
        self.best_trial = max(
            self.trials,
            key=lambda t: (t.pass_rate, -t.regression_rate)
        )


def _comb_ratio(n_minus_c: int, n: int, k: int) -> float:
    """Compute C(n-c, k) / C(n, k) safely for pass@k."""
    if n_minus_c < k:
        return 0.0
    result = 1.0
    for i in range(k):
        result *= (n_minus_c - i) / (n - i)
    return result


# ── Main Harness ─────────────────────────────────────────────────

class DriftBenchHarness:
    """
    The main harness that:
    1. Sets up an isolated sandbox for each trial
    2. Hides test files from the agent (test isolation)
    3. Feeds task steps sequentially to the agent
    4. Runs tests after each step with per-case granularity
    5. Collects per-step complexity snapshots (entropy trajectory)
    6. Validates code syntax before computing metrics
    7. Provides error feedback for agent retries
    8. Captures per-step code snapshots for diff grading
    """

    def __init__(self, base_project_dir: str, task_chain: list[TaskStep],
                 agent_fn: Callable,
                 agent_name: str = "unknown",
                 max_retries: int = 0,
                 seed: int = 0):
        self.base_project_dir = Path(base_project_dir)
        self.task_chain = task_chain
        self.agent_fn = agent_fn
        self.agent_name = agent_name
        self.max_retries = max_retries
        self.seed = seed
        self.sandbox_dir: Optional[Path] = None
        self._test_vault: Optional[Path] = None

    def _create_sandbox(self) -> Path:
        """Create an isolated sandbox. Test files are stored separately."""
        sandbox = Path(tempfile.mkdtemp(prefix=f"driftbench_s{self.seed}_"))
        project_copy = sandbox / "project"
        test_vault = sandbox / "_test_vault"

        shutil.copytree(self.base_project_dir, project_copy)

        test_vault.mkdir(parents=True, exist_ok=True)
        for test_file in project_copy.glob("test_*.py"):
            shutil.move(str(test_file), str(test_vault / test_file.name))

        self._test_vault = test_vault
        return sandbox

    # ── Syntax Validation ────────────────────────────────────────

    def _validate_syntax(self, project_dir: Path) -> tuple[bool, str]:
        """
        Validate that all Python files in the project are syntactically valid.
        Returns (is_valid, error_message).
        """
        for py_file in project_dir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
            try:
                source = py_file.read_text()
                ast.parse(source, filename=str(py_file))
            except SyntaxError as e:
                return False, f"SyntaxError in {py_file.name} line {e.lineno}: {e.msg}"
            except Exception as e:
                return False, f"Parse error in {py_file.name}: {str(e)}"
        return True, ""

    # ── Test Runner ──────────────────────────────────────────────

    def _run_tests_granular(self, project_dir: Path, test_files: list[str]) -> dict:
        """
        Run tests with per-test-case granularity using pytest verbose output.
        Returns detailed results including error output for feedback.
        """
        cases_passed = 0
        cases_failed = 0
        failed_cases = []
        errors = []
        error_output = []

        for tf in test_files:
            vault_path = self._test_vault / tf
            project_test_path = project_dir / tf

            if not vault_path.exists():
                errors.append(f"Test file not found in vault: {tf}")
                cases_failed += 1
                continue

            shutil.copy2(str(vault_path), str(project_test_path))

            try:
                result = subprocess.run(
                    ["python3", "-m", "pytest", str(project_test_path),
                     "-v", "--tb=short", "-q"],
                    capture_output=True, text=True, timeout=60,
                    cwd=str(project_dir)
                )

                stdout = result.stdout
                stderr = result.stderr

                passed_in_file = len(re.findall(r'PASSED', stdout))
                failed_in_file = len(re.findall(r'FAILED', stdout))

                if passed_in_file == 0 and failed_in_file == 0:
                    if result.returncode == 0:
                        passed_in_file = 1
                    else:
                        failed_in_file = 1
                        error_output.append(
                            f"[{tf}] Exit code {result.returncode}:\n"
                            f"{stdout[-500:]}\n{stderr[-500:]}"
                        )

                cases_passed += passed_in_file
                cases_failed += failed_in_file

                if failed_in_file > 0:
                    for line in stdout.split('\n'):
                        if 'FAILED' in line:
                            failed_cases.append(line.strip())
                    error_output.append(
                        f"[{tf}] {failed_in_file} test(s) failed:\n"
                        f"{stdout[-800:]}"
                    )

            except subprocess.TimeoutExpired:
                cases_failed += 1
                errors.append(f"{tf}: TIMEOUT (>60s)")
                error_output.append(f"[{tf}] Test execution timed out after 60 seconds")
            except Exception as e:
                cases_failed += 1
                errors.append(f"{tf}: {str(e)}")
                error_output.append(f"[{tf}] Exception: {str(e)}")
            finally:
                if project_test_path.exists():
                    project_test_path.unlink()

        return {
            "cases_passed": cases_passed,
            "cases_failed": cases_failed,
            "cases_total": cases_passed + cases_failed,
            "failed_cases": failed_cases,
            "errors": errors,
            "error_output": "\n".join(error_output),
        }

    # ── Complexity Metrics ───────────────────────────────────────

    def _compute_complexity(self, project_dir: Path, step_id: int = 0) -> ComplexitySnapshot:
        """
        Compute comprehensive code complexity metrics.
        Validates syntax first; if code is unparseable, marks as invalid
        with penalty values instead of rewarding with 0.0.
        """
        snapshot = ComplexitySnapshot(step_id=step_id)

        # Validate syntax first
        is_valid, error_msg = self._validate_syntax(project_dir)
        snapshot.syntax_valid = is_valid

        if not is_valid:
            # Code has syntax errors — use penalty sentinel values
            # avg_cc = -1.0 signals "invalid" to the grader
            snapshot.avg_cc = -1.0
            snapshot.max_cc = -1.0
            snapshot.structural_erosion = 1.0  # worst possible
            # Still count LOC
            try:
                total_loc = 0
                for py_file in project_dir.rglob("*.py"):
                    if "test_" in py_file.name or "__pycache__" in str(py_file):
                        continue
                    total_loc += len(py_file.read_text().splitlines())
                snapshot.total_loc = total_loc
            except Exception:
                pass
            return snapshot

        # Compute CC with radon
        try:
            result = subprocess.run(
                ["python3", "-m", "radon", "cc", str(project_dir), "-a", "-j"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                cc_values = []
                function_lengths = []
                for filepath, blocks in data.items():
                    if isinstance(blocks, list):
                        for block in blocks:
                            if isinstance(block, dict) and "complexity" in block:
                                cc_values.append(block["complexity"])
                                end_line = block.get("endline", block.get("lineno", 0))
                                start_line = block.get("lineno", 0)
                                if end_line and start_line:
                                    function_lengths.append(end_line - start_line + 1)

                if cc_values:
                    snapshot.avg_cc = round(sum(cc_values) / len(cc_values), 2)
                    snapshot.max_cc = max(cc_values)
                    snapshot.num_functions = len(cc_values)

                    if function_lengths:
                        snapshot.avg_function_length = round(
                            sum(function_lengths) / len(function_lengths), 1
                        )

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
                pass

        if not all_lines:
            return 0.0

        counts = Counter(all_lines)
        duplicated = sum(c - 1 for c in counts.values() if c > 1)
        return round(duplicated / max(len(all_lines), 1), 4)

    # ── Code Snapshot ────────────────────────────────────────────

    def _capture_code_snapshot(self, project_dir: Path) -> str:
        """Capture the full code state at a given step for diff grading."""
        parts = []
        for py_file in sorted(project_dir.rglob("*.py")):
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
            try:
                content = py_file.read_text()
                rel_path = py_file.relative_to(project_dir)
                parts.append(f"### {rel_path}\n```python\n{content}\n```")
            except Exception:
                continue
        return "\n\n".join(parts)

    # ── Step Execution with Retries ──────────────────────────────

    def _execute_step_with_retries(self, project_dir: Path, step: TaskStep,
                                   completed_test_files: list[str],
                                   previous_case_count: int) -> StepResult:
        """
        Execute a step with optional retries.
        On failure, provides error feedback to the agent for the next attempt.
        """
        best_result = None
        total_tokens = 0
        total_actions = 0
        total_duration = 0.0
        backup_dir = project_dir.parent / "_code_backup"

        for attempt in range(1 + self.max_retries):
            instruction = step.instruction
            if attempt > 0 and best_result and best_result.test_error_output:
                instruction = (
                    f"{step.instruction}\n\n"
                    f"--- PREVIOUS ATTEMPT FAILED ---\n"
                    f"Your previous code modification failed the following tests. "
                    f"Please fix the issues:\n\n"
                    f"{best_result.test_error_output[:2000]}"
                )

            if attempt > 0:
                # Restore from backup before retry
                if backup_dir.exists():
                    for f in project_dir.glob("*.py"):
                        if "test_" not in f.name:
                            f.unlink()
                    for f in backup_dir.glob("*.py"):
                        shutil.copy2(str(f), str(project_dir / f.name))
            else:
                # Create backup before first attempt
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                backup_dir.mkdir()
                for f in project_dir.glob("*.py"):
                    if "test_" not in f.name:
                        shutil.copy2(str(f), str(backup_dir / f.name))

            # Execute agent
            start_time = time.time()
            try:
                agent_output, tokens, actions = self.agent_fn(
                    str(project_dir), instruction
                )
            except Exception as e:
                agent_output = f"AGENT ERROR: {str(e)}"
                tokens, actions = 0, 0
            duration = time.time() - start_time

            total_tokens += tokens
            total_actions += actions
            total_duration += duration

            # Run new test
            new_result = self._run_tests_granular(project_dir, [step.test_file])
            new_tests_passed = new_result["cases_passed"]
            new_tests_total = new_result["cases_total"]
            new_test_passed = new_result["cases_failed"] == 0 and new_tests_passed > 0

            # Run regression tests
            regression_failures = 0
            total_previous = previous_case_count
            if completed_test_files:
                reg_result = self._run_tests_granular(project_dir, completed_test_files)
                regression_failures = reg_result["cases_failed"]
                total_previous = reg_result["cases_total"]

            step_passed = new_test_passed and regression_failures == 0

            # Compute complexity
            step_snapshot = self._compute_complexity(project_dir, step.step_id)

            # Capture code snapshot
            code_snapshot = self._capture_code_snapshot(project_dir)

            current_result = StepResult(
                step_id=step.step_id,
                task_type=step.task_type,
                passed=step_passed,
                regression_failures=regression_failures,
                total_previous_tests=total_previous,
                new_tests_passed=new_tests_passed,
                new_tests_total=new_tests_total,
                new_test_passed=new_test_passed,
                agent_output=agent_output[:1000],
                duration_seconds=round(total_duration, 2),
                token_count=total_tokens,
                action_count=total_actions,
                complexity_snapshot=step_snapshot,
                code_snapshot=code_snapshot,
                test_error_output=new_result.get("error_output", ""),
                retry_count=attempt,
            )

            if step_passed:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                return current_result

            best_result = current_result

            if attempt < self.max_retries:
                print(f"    Attempt {attempt + 1} failed, retrying with error feedback...")

        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        return best_result

    # ── Trial Execution ──────────────────────────────────────────

    def run_trial(self, trial_id: int = 0) -> TrialResult:
        """Execute the full task chain and collect results."""
        sandbox = self._create_sandbox()
        self.sandbox_dir = sandbox
        project_dir = sandbox / "project"

        baseline = self._compute_complexity(project_dir, step_id=0)

        trial = TrialResult(
            agent_name=self.agent_name,
            task_chain_name=self.task_chain[0].instruction[:50] if self.task_chain else "empty",
            trial_id=trial_id,
        )

        completed_test_files: list[str] = []
        previous_case_count = 0

        for step in self.task_chain:
            print(f"\n{'='*60}")
            print(f"  Step {step.step_id}: [{step.task_type.upper()}]")
            print(f"  {step.instruction[:80]}...")
            if self.max_retries > 0:
                print(f"  (max retries: {self.max_retries})")
            print(f"{'='*60}")

            step_result = self._execute_step_with_retries(
                project_dir, step, completed_test_files, previous_case_count
            )

            trial.step_results.append(step_result)

            status = "PASS" if step_result.passed else "FAIL"
            reg_info = (f"(regressions: {step_result.regression_failures}/"
                        f"{step_result.total_previous_tests} cases)"
                        if step_result.total_previous_tests > 0 else "")
            new_info = (f"(new: {step_result.new_tests_passed}/"
                        f"{step_result.new_tests_total} cases)")
            cc = step_result.complexity_snapshot
            syntax_flag = " [SYNTAX ERROR]" if not cc.syntax_valid else ""
            cc_info = f"[CC={cc.avg_cc}, erosion={cc.structural_erosion}{syntax_flag}]"
            retry_info = f" [retry={step_result.retry_count}]" if step_result.retry_count > 0 else ""
            print(f"  Result: {status} {new_info} {reg_info} {cc_info}{retry_info}")

            if step_result.new_test_passed:
                completed_test_files.append(step.test_file)
                previous_case_count += step_result.new_tests_passed

        # Compute final metrics
        final = self._compute_complexity(project_dir, step_id=len(self.task_chain))

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
            "syntax_valid_before": baseline.syntax_valid,
            "syntax_valid_after": final.syntax_valid,
        }

        trial.compute_metrics()
        return trial

    def run_multi_trial(self, num_trials: int = 3) -> MultiTrialResult:
        """Run multiple trials and compute pass@k statistics."""
        multi = MultiTrialResult(
            agent_name=self.agent_name,
            task_chain_name=self.task_chain[0].instruction[:50] if self.task_chain else "empty",
        )

        for i in range(num_trials):
            print(f"\n{'#'*60}")
            print(f"  TRIAL {i+1}/{num_trials} -- {self.agent_name}")
            print(f"{'#'*60}")

            self.seed = i
            trial = self.run_trial(trial_id=i)
            multi.trials.append(trial)
            self.cleanup()

        multi.compute_aggregate()
        return multi

    def cleanup(self):
        """Remove sandbox directory."""
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
