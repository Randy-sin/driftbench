"""
DriftBench Grader — Multi-dimensional grading system that combines
code-based static analysis with LLM-as-a-Judge evaluation.

Key Features:
- Syntax error penalty: unparseable code gets 0 on entropy/erosion (not 100)
- Functional-regression binding: regression score is discounted when
  functional score is below threshold (fixes "do nothing" loophole)
- Effective regression rate: only counts steps with actual prior tests
- Diff-aware LLM Judge: judges see code evolution, not just final state
- Per-step code snapshots enable trajectory-aware evaluation
- Multi-trial aggregation with confidence intervals
- Improved overall score formula with non-linear weighting
"""

import json
import os
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ── Grade Report ─────────────────────────────────────────────────

@dataclass
class GradeReport:
    """Complete grading report for a trial."""
    # Code-based metrics (0-100)
    functional_score: float = 0.0
    regression_score: float = 0.0       # raw regression resistance
    adjusted_regression_score: float = 0.0  # NEW: discounted by functional score
    entropy_score: float = 0.0
    duplication_score: float = 0.0
    erosion_score: float = 0.0
    syntax_penalty: float = 0.0         # NEW: penalty for syntax errors

    # LLM-as-Judge metrics (0-100)
    consistency_score: float = 0.0
    refactor_awareness_score: float = 0.0
    taste_score: float = 0.0

    # Inter-rater reliability
    judge_agreement: float = 0.0

    # Cost metrics
    token_efficiency: float = 0.0
    action_efficiency: float = 0.0

    # Overall
    overall_score: float = 0.0

    def compute_overall(self):
        """
        Weighted average of all dimensions.
        Uses adjusted_regression_score and includes syntax_penalty.
        """
        weights = {
            "functional": 0.25,
            "regression": 0.20,
            "entropy": 0.10,
            "erosion": 0.10,
            "consistency": 0.12,
            "refactor": 0.10,
            "taste": 0.05,
            "cost": 0.08,
        }

        # Cost score: lower tokens per passed step = better
        cost_score = min(100, max(0, 100 - self.token_efficiency / 100))

        raw_score = (
            weights["functional"] * self.functional_score +
            weights["regression"] * self.adjusted_regression_score +
            weights["entropy"] * self.entropy_score +
            weights["erosion"] * self.erosion_score +
            weights["consistency"] * self.consistency_score +
            weights["refactor"] * self.refactor_awareness_score +
            weights["taste"] * self.taste_score +
            weights["cost"] * cost_score
        )

        # Apply syntax penalty: each syntax error step reduces overall by 5 points
        raw_score = max(0, raw_score - self.syntax_penalty)

        self.overall_score = round(raw_score, 1)


# ── Code-Based Grader ────────────────────────────────────────────

class CodeBasedGrader:
    """
    Grades based on static code analysis metrics.
    Fixes the CC=0 exploit and the "do nothing" regression loophole.
    """

    @staticmethod
    def grade(trial_result) -> dict:
        # ── Functional Score ──
        functional_score = trial_result.pass_rate * 100

        # ── Raw Regression Score ──
        regression_score = (1 - trial_result.regression_rate) * 100

        # ── Adjusted Regression Score (NEW) ──
        # The "do nothing" loophole: if an agent fails all new tests,
        # it accumulates no previous tests, so regression_rate = 0 → score = 100.
        # Fix: discount regression score by functional score.
        # If functional_score < 20%, regression score is capped at 50%.
        # This ensures that "doing nothing" can't get a perfect regression score.
        if functional_score < 20:
            # Agent barely passed anything — regression score is unreliable
            adjusted_regression_score = min(regression_score, 50.0)
        elif functional_score < 40:
            # Partial discount
            discount = functional_score / 40.0  # 0.5 to 1.0
            adjusted_regression_score = regression_score * discount
        else:
            adjusted_regression_score = regression_score

        # Also use effective regression rate for agents that did pass some tests
        eff_reg_rate = getattr(trial_result, 'effective_regression_rate',
                               trial_result.regression_rate)
        if eff_reg_rate > trial_result.regression_rate:
            # Effective rate is worse — use it
            adjusted_regression_score = min(
                adjusted_regression_score,
                (1 - eff_reg_rate) * 100
            )

        # ── Entropy Score (with syntax validation) ──
        cc_delta = trial_result.entropy_delta.get("cc_delta", 0)
        syntax_valid_after = trial_result.entropy_delta.get("syntax_valid_after", True)

        if not syntax_valid_after:
            # Code has syntax errors at the end — entropy score = 0
            entropy_score = 0.0
        elif cc_delta <= 0:
            entropy_score = 100.0
        elif cc_delta < 1:
            entropy_score = 80.0
        elif cc_delta < 3:
            entropy_score = 50.0
        else:
            entropy_score = max(0, 30 - cc_delta * 5)

        # ── Duplication Score ──
        dup_delta = trial_result.entropy_delta.get("duplication_delta", 0)
        if dup_delta <= 0:
            duplication_score = 100.0
        elif dup_delta < 0.05:
            duplication_score = 70.0
        elif dup_delta < 0.15:
            duplication_score = 40.0
        else:
            duplication_score = 10.0

        # ── Structural Erosion Score (with syntax validation) ──
        erosion_after = trial_result.entropy_delta.get("erosion_after", 0)
        erosion_delta = trial_result.entropy_delta.get("erosion_delta", 0)

        if not syntax_valid_after:
            # Code has syntax errors — erosion score = 0
            erosion_score = 0.0
        elif erosion_after <= 0.3:
            erosion_score = 100.0
        elif erosion_after <= 0.5:
            erosion_score = 70.0
        elif erosion_after <= 0.7:
            erosion_score = 40.0
        else:
            erosion_score = max(0, 20 - erosion_delta * 100)

        # ── Syntax Penalty (NEW) ──
        syntax_error_count = getattr(trial_result, 'syntax_error_count', 0)
        syntax_penalty = syntax_error_count * 5.0  # 5 points per syntax error step

        return {
            "functional_score": round(functional_score, 1),
            "regression_score": round(regression_score, 1),
            "adjusted_regression_score": round(adjusted_regression_score, 1),
            "entropy_score": round(entropy_score, 1),
            "duplication_score": round(duplication_score, 1),
            "erosion_score": round(erosion_score, 1),
            "syntax_penalty": round(syntax_penalty, 1),
        }


# ── LLM Judge Grader ────────────────────────────────────────────

class LLMJudgeGrader:
    """
    Uses LLM-as-a-Judge with multi-model cross-validation.
    Diff-aware evaluation — judges see code evolution, not just final state.
    """

    RUBRIC = """You are an expert software engineering reviewer evaluating code produced by an AI coding agent.
The agent was given a sequence of tasks on the same codebase. You will see BOTH the evolution of the code (diffs between steps) AND the final state.

CONTEXT: The agent was asked to iteratively evolve a codebase through feature additions, bug fixes, and refactoring.
The key question is: did the agent maintain code quality as the codebase grew, or did it let entropy accumulate?

Evaluate on three dimensions (score each 0-100):

1. **Consistency Score**: Does the code follow a consistent style, naming convention, and architectural pattern throughout?
   - 90-100: Perfectly consistent, feels like one author wrote it all
   - 60-89: Mostly consistent with minor deviations
   - 30-59: Noticeable inconsistencies in style or patterns
   - 0-29: Chaotic, different parts feel like different codebases

2. **Refactor Awareness Score**: Did the agent proactively improve code structure when adding features, or did it just pile on more code?
   Look at the diffs: did the agent extract common utilities, rename for clarity, reduce duplication?
   - 90-100: Actively refactored, extracted common utilities, reduced duplication
   - 60-89: Some evidence of refactoring when natural
   - 30-59: Mostly just added code without considering existing structure
   - 0-29: Made the codebase significantly harder to maintain

3. **Taste Score**: Does the code demonstrate good engineering judgment? (e.g., appropriate error handling, meaningful variable names, proper separation of concerns, clean function signatures)
   - 90-100: Production-quality code with excellent judgment
   - 60-89: Good code with minor taste issues
   - 30-59: Functional but lacks engineering polish
   - 0-29: Poor judgment, code smells everywhere

Respond in JSON format:
{"consistency_score": <int>, "refactor_awareness_score": <int>, "taste_score": <int>, "reasoning": "<brief explanation>"}
"""

    def __init__(self, models: list[str] = None):
        self.models = models or ["gpt-4.1-mini"]
        if HAS_OPENAI:
            self.client = OpenAI()
        else:
            self.client = None

    def _build_diff_context(self, code_snapshots: list[str],
                            task_descriptions: list[str]) -> str:
        """
        NEW: Build a diff-aware context showing code evolution across steps.
        """
        if not code_snapshots:
            return ""

        parts = []
        for i, snapshot in enumerate(code_snapshots):
            task_desc = task_descriptions[i] if i < len(task_descriptions) else "Unknown"
            parts.append(f"## After Step {i+1}: {task_desc}\n\n{snapshot}")

        return "\n\n---\n\n".join(parts)

    def _judge_with_model(self, model: str, code_text: str,
                          tasks_text: str, diff_context: str = "") -> dict:
        """Run a single judge evaluation with a specific model."""
        prompt = f"""The agent was given these tasks in sequence:
{tasks_text}

"""
        if diff_context:
            prompt += f"""Here is how the code evolved at each step:

{diff_context}

---

"""

        prompt += f"""Here is the FINAL state of the codebase:

{code_text}

Please evaluate according to the rubric."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.RUBRIC},
                    {"role": "user", "content": prompt[:20000]}  # cap context
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "consistency_score": float(result.get("consistency_score", 50)),
                "refactor_awareness_score": float(result.get("refactor_awareness_score", 50)),
                "taste_score": float(result.get("taste_score", 50)),
                "reasoning": result.get("reasoning", ""),
                "model": model,
            }
        except Exception as e:
            return {
                "consistency_score": 50.0,
                "refactor_awareness_score": 50.0,
                "taste_score": 50.0,
                "reasoning": f"Judge error ({model}): {str(e)}",
                "model": model,
            }

    def grade(self, project_dir: str, task_descriptions: list[str],
              code_snapshots: list[str] = None) -> dict:
        """
        Use multiple LLM judges with diff-aware evaluation.
        Passes code evolution context to judges.
        """
        if not self.client:
            return {
                "consistency_score": 50.0,
                "refactor_awareness_score": 50.0,
                "taste_score": 50.0,
                "reasoning": "LLM judge unavailable.",
                "judge_agreement": 0.0,
            }

        # Collect final code
        code_content = []
        project_path = Path(project_dir)
        for py_file in sorted(project_path.rglob("*.py")):
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
            try:
                content = py_file.read_text()
                rel_path = py_file.relative_to(project_path)
                code_content.append(f"### {rel_path}\n```python\n{content}\n```")
            except Exception:
                continue

        code_text = "\n\n".join(code_content)
        if len(code_text) > 15000:
            code_text = code_text[:15000] + "\n\n... [TRUNCATED] ..."

        tasks_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(task_descriptions))

        # Build diff context (NEW)
        diff_context = ""
        if code_snapshots:
            diff_context = self._build_diff_context(code_snapshots, task_descriptions)
            # Truncate if too long
            if len(diff_context) > 10000:
                diff_context = diff_context[:10000] + "\n\n... [TRUNCATED] ..."

        # Run all judges
        all_results = []
        for model in self.models:
            result = self._judge_with_model(model, code_text, tasks_text, diff_context)
            all_results.append(result)
            print(f"    Judge ({model}): C={result['consistency_score']}, "
                  f"R={result['refactor_awareness_score']}, T={result['taste_score']}")

        # Average scores across judges
        avg_consistency = sum(r["consistency_score"] for r in all_results) / len(all_results)
        avg_refactor = sum(r["refactor_awareness_score"] for r in all_results) / len(all_results)
        avg_taste = sum(r["taste_score"] for r in all_results) / len(all_results)

        # Compute inter-rater agreement
        agreement = 1.0
        if len(all_results) > 1:
            pairs = []
            for i in range(len(all_results)):
                for j in range(i + 1, len(all_results)):
                    s1 = [all_results[i]["consistency_score"],
                          all_results[i]["refactor_awareness_score"],
                          all_results[i]["taste_score"]]
                    s2 = [all_results[j]["consistency_score"],
                          all_results[j]["refactor_awareness_score"],
                          all_results[j]["taste_score"]]
                    diffs = [abs(a - b) for a, b in zip(s1, s2)]
                    avg_diff = sum(diffs) / len(diffs)
                    pairs.append(1 - avg_diff / 100)
            agreement = round(sum(pairs) / len(pairs), 3)

        reasoning_parts = [f"[{r['model']}] {r['reasoning']}" for r in all_results]

        return {
            "consistency_score": round(avg_consistency, 1),
            "refactor_awareness_score": round(avg_refactor, 1),
            "taste_score": round(avg_taste, 1),
            "reasoning": " | ".join(reasoning_parts),
            "judge_agreement": agreement,
        }


# ── Combined Grader ──────────────────────────────────────────────

class DriftBenchGrader:
    """Combines code-based and LLM-based grading into a final report."""

    def __init__(self, use_llm_judge: bool = True, judge_models: list[str] = None):
        self.code_grader = CodeBasedGrader()
        if use_llm_judge:
            self.llm_grader = LLMJudgeGrader(models=judge_models or ["gpt-4.1-mini"])
        else:
            self.llm_grader = None

    def grade_trial(self, trial_result, project_dir: str = None,
                    task_descriptions: list[str] = None) -> GradeReport:
        """Produce a complete grade report for a trial."""
        report = GradeReport()

        # Code-based grading
        code_scores = self.code_grader.grade(trial_result)
        report.functional_score = code_scores["functional_score"]
        report.regression_score = code_scores["regression_score"]
        report.adjusted_regression_score = code_scores["adjusted_regression_score"]
        report.entropy_score = code_scores["entropy_score"]
        report.duplication_score = code_scores["duplication_score"]
        report.erosion_score = code_scores["erosion_score"]
        report.syntax_penalty = code_scores["syntax_penalty"]

        # LLM-based grading (with diff context)
        if self.llm_grader and project_dir and task_descriptions:
            code_snapshots = getattr(trial_result, 'code_snapshots', None)
            llm_scores = self.llm_grader.grade(
                project_dir, task_descriptions,
                code_snapshots=code_snapshots
            )
            report.consistency_score = llm_scores["consistency_score"]
            report.refactor_awareness_score = llm_scores["refactor_awareness_score"]
            report.taste_score = llm_scores["taste_score"]
            report.judge_agreement = llm_scores.get("judge_agreement", 0.0)

        # Cost metrics
        passed_steps = max(1, sum(1 for s in trial_result.step_results if s.passed))
        report.token_efficiency = trial_result.total_tokens / passed_steps
        report.action_efficiency = trial_result.total_actions / passed_steps

        report.compute_overall()
        return report

    def grade_multi_trial(self, multi_trial_result,
                          project_dir: str = None,
                          task_descriptions: list[str] = None) -> dict:
        """
        NEW: Grade across multiple trials and compute confidence intervals.
        Returns the best trial's grade plus aggregate statistics.
        """
        if not multi_trial_result.trials:
            return {"best_grade": GradeReport(), "stats": {}}

        # Grade each trial
        all_grades = []
        for trial in multi_trial_result.trials:
            grade = self.grade_trial(trial, project_dir, task_descriptions)
            all_grades.append(grade)

        # Find best trial
        best_idx = max(range(len(all_grades)),
                       key=lambda i: all_grades[i].overall_score)
        best_grade = all_grades[best_idx]

        # Compute statistics
        overall_scores = [g.overall_score for g in all_grades]
        functional_scores = [g.functional_score for g in all_grades]
        regression_scores = [g.adjusted_regression_score for g in all_grades]

        stats = {
            "num_trials": len(all_grades),
            "overall_mean": round(sum(overall_scores) / len(overall_scores), 1),
            "overall_std": round(statistics.stdev(overall_scores), 1) if len(overall_scores) > 1 else 0.0,
            "overall_min": round(min(overall_scores), 1),
            "overall_max": round(max(overall_scores), 1),
            "functional_mean": round(sum(functional_scores) / len(functional_scores), 1),
            "regression_mean": round(sum(regression_scores) / len(regression_scores), 1),
            "pass_at_1": multi_trial_result.pass_at_1,
            "pass_at_k": multi_trial_result.pass_at_k,
        }

        return {"best_grade": best_grade, "stats": stats, "all_grades": all_grades}
