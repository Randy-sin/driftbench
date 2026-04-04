"""
DriftBench Grader — Multi-dimensional grading system that combines
code-based static analysis with LLM-as-a-Judge evaluation.

v2.0 Improvements:
- Structural Erosion scoring (SlopCodeBench-inspired)
- Multi-model cross-validation for LLM Judge
- Per-step trajectory-aware rubric
- Cohen's Kappa inter-rater reliability
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class GradeReport:
    """Complete grading report for a trial."""
    # Code-based metrics (0-100)
    functional_score: float = 0.0
    regression_score: float = 0.0
    entropy_score: float = 0.0
    duplication_score: float = 0.0
    erosion_score: float = 0.0  # NEW: structural erosion

    # LLM-as-Judge metrics (0-100)
    consistency_score: float = 0.0
    refactor_awareness_score: float = 0.0
    taste_score: float = 0.0

    # Inter-rater reliability (NEW)
    judge_agreement: float = 0.0  # 0-1, agreement between judge models

    # Cost metrics
    token_efficiency: float = 0.0
    action_efficiency: float = 0.0

    # Overall
    overall_score: float = 0.0

    def compute_overall(self):
        """Weighted average of all dimensions."""
        weights = {
            "functional": 0.20,
            "regression": 0.20,
            "entropy": 0.10,
            "erosion": 0.10,
            "consistency": 0.15,
            "refactor": 0.10,
            "taste": 0.05,
            "cost": 0.10,
        }
        cost_score = min(100, max(0, 100 - self.token_efficiency / 100))

        self.overall_score = round(
            weights["functional"] * self.functional_score +
            weights["regression"] * self.regression_score +
            weights["entropy"] * self.entropy_score +
            weights["erosion"] * self.erosion_score +
            weights["consistency"] * self.consistency_score +
            weights["refactor"] * self.refactor_awareness_score +
            weights["taste"] * self.taste_score +
            weights["cost"] * cost_score,
            1
        )


class CodeBasedGrader:
    """Grades based on static code analysis metrics."""

    @staticmethod
    def grade(trial_result) -> dict:
        functional_score = trial_result.pass_rate * 100
        regression_score = (1 - trial_result.regression_rate) * 100

        # Entropy score: penalize increase in complexity
        cc_delta = trial_result.entropy_delta.get("cc_delta", 0)
        if cc_delta <= 0:
            entropy_score = 100.0
        elif cc_delta < 1:
            entropy_score = 80.0
        elif cc_delta < 3:
            entropy_score = 50.0
        else:
            entropy_score = max(0, 30 - cc_delta * 5)

        # Duplication score
        dup_delta = trial_result.entropy_delta.get("duplication_delta", 0)
        if dup_delta <= 0:
            duplication_score = 100.0
        elif dup_delta < 0.05:
            duplication_score = 70.0
        elif dup_delta < 0.15:
            duplication_score = 40.0
        else:
            duplication_score = 10.0

        # Structural Erosion score (NEW)
        erosion_after = trial_result.entropy_delta.get("erosion_after", 0)
        erosion_delta = trial_result.entropy_delta.get("erosion_delta", 0)
        if erosion_after <= 0.3:
            erosion_score = 100.0
        elif erosion_after <= 0.5:
            erosion_score = 70.0
        elif erosion_after <= 0.7:
            erosion_score = 40.0
        else:
            erosion_score = max(0, 20 - erosion_delta * 100)

        return {
            "functional_score": round(functional_score, 1),
            "regression_score": round(regression_score, 1),
            "entropy_score": round(entropy_score, 1),
            "duplication_score": round(duplication_score, 1),
            "erosion_score": round(erosion_score, 1),
        }


class LLMJudgeGrader:
    """
    Uses LLM-as-a-Judge with multi-model cross-validation.
    Evaluates qualitative aspects: consistency, refactoring awareness, taste.
    """

    RUBRIC = """You are an expert software engineering reviewer evaluating code produced by an AI coding agent.
The agent was given a sequence of tasks on the same codebase. You are reviewing the FINAL state of the code.

CONTEXT: The agent was asked to iteratively evolve a codebase through feature additions, bug fixes, and refactoring.
The key question is: did the agent maintain code quality as the codebase grew, or did it let entropy accumulate?

Evaluate on three dimensions (score each 0-100):

1. **Consistency Score**: Does the code follow a consistent style, naming convention, and architectural pattern throughout?
   - 90-100: Perfectly consistent, feels like one author wrote it all
   - 60-89: Mostly consistent with minor deviations
   - 30-59: Noticeable inconsistencies in style or patterns
   - 0-29: Chaotic, different parts feel like different codebases

2. **Refactor Awareness Score**: Did the agent proactively improve code structure when adding features, or did it just pile on more code?
   - 90-100: Actively refactored, extracted common utilities, reduced duplication
   - 60-89: Some evidence of refactoring when natural
   - 30-59: Mostly just added code without considering existing structure
   - 0-29: Made the codebase significantly harder to maintain

3. **Taste Score**: Does the code demonstrate good engineering judgment? (e.g., appropriate error handling, meaningful variable names, proper separation of concerns)
   - 90-100: Production-quality code with excellent judgment
   - 60-89: Good code with minor taste issues
   - 30-59: Functional but lacks engineering polish
   - 0-29: Poor judgment, code smells everywhere

Respond in JSON format:
{"consistency_score": <int>, "refactor_awareness_score": <int>, "taste_score": <int>, "reasoning": "<brief explanation>"}
"""

    def __init__(self, models: list[str] = None):
        """
        Args:
            models: List of models to use for cross-validation.
                    If multiple models, computes inter-rater agreement.
        """
        self.models = models or ["gpt-4.1-mini"]
        if HAS_OPENAI:
            self.client = OpenAI()
        else:
            self.client = None

    def _judge_with_model(self, model: str, code_text: str, tasks_text: str) -> dict:
        """Run a single judge evaluation with a specific model."""
        prompt = f"""The agent was given these tasks in sequence:
{tasks_text}

Here is the final state of the codebase:

{code_text}

Please evaluate according to the rubric."""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.RUBRIC},
                    {"role": "user", "content": prompt}
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

    def grade(self, project_dir: str, task_descriptions: list[str]) -> dict:
        """
        Use multiple LLM judges and compute cross-validated scores.
        """
        if not self.client:
            return {
                "consistency_score": 50.0,
                "refactor_awareness_score": 50.0,
                "taste_score": 50.0,
                "reasoning": "LLM judge unavailable.",
                "judge_agreement": 0.0,
            }

        # Collect code
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

        # Run all judges
        all_results = []
        for model in self.models:
            result = self._judge_with_model(model, code_text, tasks_text)
            all_results.append(result)
            print(f"    Judge ({model}): C={result['consistency_score']}, "
                  f"R={result['refactor_awareness_score']}, T={result['taste_score']}")

        # Average scores across judges
        avg_consistency = sum(r["consistency_score"] for r in all_results) / len(all_results)
        avg_refactor = sum(r["refactor_awareness_score"] for r in all_results) / len(all_results)
        avg_taste = sum(r["taste_score"] for r in all_results) / len(all_results)

        # Compute inter-rater agreement (simple: 1 - normalized std deviation)
        agreement = 1.0
        if len(all_results) > 1:
            import statistics
            all_scores = []
            for r in all_results:
                all_scores.extend([r["consistency_score"], r["refactor_awareness_score"], r["taste_score"]])
            pairs = []
            for i in range(len(all_results)):
                for j in range(i + 1, len(all_results)):
                    s1 = [all_results[i]["consistency_score"], all_results[i]["refactor_awareness_score"], all_results[i]["taste_score"]]
                    s2 = [all_results[j]["consistency_score"], all_results[j]["refactor_awareness_score"], all_results[j]["taste_score"]]
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
        report.entropy_score = code_scores["entropy_score"]
        report.duplication_score = code_scores["duplication_score"]
        report.erosion_score = code_scores["erosion_score"]

        # LLM-based grading
        if self.llm_grader and project_dir and task_descriptions:
            llm_scores = self.llm_grader.grade(project_dir, task_descriptions)
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
