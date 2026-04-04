#!/usr/bin/env python3
"""
DriftBench v2.0 — Multi-task, multi-model benchmark runner.

Usage:
    python run_benchmark.py --tasks todo_api calculator markdown_parser file_manager
    python run_benchmark.py --tasks todo_api --models gpt-4.1-mini gpt-4.1-nano gemini-2.5-flash
    python run_benchmark.py --tasks todo_api --no-llm-agent  # baseline only
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from driftbench.harness import DriftBenchHarness, TaskStep
from driftbench.agents import NaiveAppendAgent, LLMCodingAgent
from driftbench.grader import DriftBenchGrader
from driftbench.visualize import (
    plot_radar_chart, plot_entropy_trajectory, plot_step_heatmap,
    plot_overall_comparison, plot_regression_waterfall,
    plot_multi_task_dashboard, save_results_json
)


def load_task_chain(task_dir: str) -> list[TaskStep]:
    """Load task chain from a task directory."""
    chain_path = Path(task_dir) / "task_chain.json"
    with open(chain_path) as f:
        data = json.load(f)

    steps = []
    for s in data["steps"]:
        steps.append(TaskStep(
            step_id=s["step_id"],
            instruction=s["instruction"],
            task_type=s["task_type"],
            test_file=s["test_file"],
            timeout=s.get("timeout", 120)
        ))
    return steps


def run_single_task(task_name, task_dir, models, output_dir,
                    use_llm_judge=True, judge_models=None,
                    run_naive=True, run_llm=True):
    """Run benchmark on a single task with multiple models."""
    print(f"\n{'#'*70}")
    print(f"  TASK: {task_name}")
    print(f"{'#'*70}")

    task_chain = load_task_chain(task_dir)
    task_descriptions = [s.instruction for s in task_chain]

    trial_results = {}
    grade_reports = {}
    all_step_results = {}
    all_trajectories = {}

    # --- Naive Baseline ---
    if run_naive:
        print(f"\n>>> Running Naive Baseline on {task_name}...")
        naive_agent = NaiveAppendAgent()
        harness = DriftBenchHarness(
            base_project_dir=task_dir,
            task_chain=task_chain,
            agent_fn=naive_agent,
            agent_name="Naive Baseline"
        )
        trial = harness.run_trial()
        trial_results["Naive Baseline"] = trial
        all_step_results["Naive Baseline"] = [
            {
                "step_id": sr.step_id,
                "task_type": sr.task_type,
                "passed": sr.passed,
                "new_test_passed": sr.new_test_passed,
                "new_tests_passed": sr.new_tests_passed,
                "new_tests_total": sr.new_tests_total,
                "regression_failures": sr.regression_failures,
                "total_previous_tests": sr.total_previous_tests,
                "duration_seconds": sr.duration_seconds,
                "token_count": sr.token_count,
            }
            for sr in trial.step_results
        ]
        all_trajectories["Naive Baseline"] = trial.entropy_trajectory

        # Grade
        grader = DriftBenchGrader(use_llm_judge=use_llm_judge, judge_models=judge_models)
        sandbox_dir = str(harness.sandbox_dir / "project") if harness.sandbox_dir else None
        grade = grader.grade_trial(trial, sandbox_dir, task_descriptions)
        grade_reports["Naive Baseline"] = grade
        harness.cleanup()

        print(f"\n  Naive Baseline: pass_rate={trial.pass_rate:.2f}, "
              f"regression_rate={trial.regression_rate:.2f}, "
              f"overall={grade.overall_score:.1f}")

    # --- LLM Agents ---
    if run_llm:
        for model in models:
            agent_name = f"LLM ({model})"
            print(f"\n>>> Running {agent_name} on {task_name}...")

            try:
                llm_agent = LLMCodingAgent(model=model)
                harness = DriftBenchHarness(
                    base_project_dir=task_dir,
                    task_chain=task_chain,
                    agent_fn=llm_agent,
                    agent_name=agent_name
                )
                trial = harness.run_trial()
                trial_results[agent_name] = trial
                all_step_results[agent_name] = [
                    {
                        "step_id": sr.step_id,
                        "task_type": sr.task_type,
                        "passed": sr.passed,
                        "new_test_passed": sr.new_test_passed,
                        "new_tests_passed": sr.new_tests_passed,
                        "new_tests_total": sr.new_tests_total,
                        "regression_failures": sr.regression_failures,
                        "total_previous_tests": sr.total_previous_tests,
                        "duration_seconds": sr.duration_seconds,
                        "token_count": sr.token_count,
                    }
                    for sr in trial.step_results
                ]
                all_trajectories[agent_name] = trial.entropy_trajectory

                # Grade
                grader = DriftBenchGrader(use_llm_judge=use_llm_judge, judge_models=judge_models)
                sandbox_dir = str(harness.sandbox_dir / "project") if harness.sandbox_dir else None
                grade = grader.grade_trial(trial, sandbox_dir, task_descriptions)
                grade_reports[agent_name] = grade
                harness.cleanup()

                print(f"\n  {agent_name}: pass_rate={trial.pass_rate:.2f}, "
                      f"regression_rate={trial.regression_rate:.2f}, "
                      f"overall={grade.overall_score:.1f}")

            except Exception as e:
                print(f"\n  ERROR running {agent_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return trial_results, grade_reports, all_step_results, all_trajectories


def main():
    parser = argparse.ArgumentParser(description="DriftBench v2.0 Benchmark Runner")
    parser.add_argument("--tasks", nargs="+", default=["todo_api"],
                        help="Task directories to run (under tasks/)")
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"],
                        help="LLM models to evaluate")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results and charts")
    parser.add_argument("--no-llm-agent", action="store_true",
                        help="Skip LLM agents, run only naive baseline")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip LLM-as-Judge grading")
    parser.add_argument("--judge-models", nargs="+",
                        default=["gpt-4.1-mini", "gemini-2.5-flash"],
                        help="Models to use for LLM-as-Judge cross-validation")
    parser.add_argument("--no-naive", action="store_true",
                        help="Skip naive baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_base = Path(__file__).parent / "tasks"

    # Collect results across all tasks
    all_task_grades = {}
    all_task_trials = {}
    all_task_steps = {}
    all_task_trajectories = {}

    for task_name in args.tasks:
        task_dir = str(tasks_base / task_name)
        if not Path(task_dir).exists():
            print(f"WARNING: Task directory not found: {task_dir}")
            continue

        trials, grades, steps, trajs = run_single_task(
            task_name=task_name,
            task_dir=task_dir,
            models=args.models,
            output_dir=str(output_dir),
            use_llm_judge=not args.no_llm_judge,
            judge_models=args.judge_models,
            run_naive=not args.no_naive,
            run_llm=not args.no_llm_agent,
        )

        all_task_grades[task_name] = grades
        all_task_trials[task_name] = trials
        all_task_steps[task_name] = steps
        all_task_trajectories[task_name] = trajs

        # Per-task charts
        task_output = output_dir / task_name
        task_output.mkdir(parents=True, exist_ok=True)

        if grades:
            print(f"\n  Generating charts for {task_name}...")
            plot_radar_chart(grades, str(task_output / "radar_chart.png"))
            plot_overall_comparison(grades, str(task_output / "overall_comparison.png"))

        if trajs:
            plot_entropy_trajectory(trajs, str(task_output / "entropy_trajectory.png"))

        if steps:
            plot_step_heatmap(steps, str(task_output / "step_heatmap.png"))
            plot_regression_waterfall(steps, str(task_output / "regression_waterfall.png"))

    # Cross-task dashboard
    if len(all_task_grades) > 1:
        print(f"\n  Generating cross-task dashboard...")
        plot_multi_task_dashboard(all_task_grades, str(output_dir / "multi_task_dashboard.png"))

    # Save all results
    save_results_json({
        "tasks": {
            task_name: {
                "trials": {
                    agent: {
                        "pass_rate": t.pass_rate,
                        "regression_rate": t.regression_rate,
                        "entropy_delta": t.entropy_delta,
                        "entropy_trajectory": t.entropy_trajectory,
                        "total_tokens": t.total_tokens,
                        "total_duration": t.total_duration,
                    }
                    for agent, t in all_task_trials[task_name].items()
                },
                "steps": all_task_steps[task_name],
            }
            for task_name in all_task_grades
        },
        "grades": {
            task_name: {
                agent: {
                    "functional_score": g.functional_score,
                    "regression_score": g.regression_score,
                    "entropy_score": g.entropy_score,
                    "erosion_score": g.erosion_score,
                    "consistency_score": g.consistency_score,
                    "refactor_awareness_score": g.refactor_awareness_score,
                    "taste_score": g.taste_score,
                    "overall_score": g.overall_score,
                    "judge_agreement": g.judge_agreement,
                    "token_efficiency": g.token_efficiency,
                }
                for agent, g in all_task_grades[task_name].items()
            }
            for task_name in all_task_grades
        }
    }, str(output_dir / "results.json"))

    # Print summary table
    print(f"\n{'='*80}")
    print("  DRIFTBENCH v2.0 RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Task':<20} {'Agent':<25} {'Pass%':>6} {'Reg%':>6} {'Overall':>8}")
    print(f"{'-'*80}")
    for task_name in all_task_grades:
        for agent, grade in all_task_grades[task_name].items():
            trial = all_task_trials[task_name][agent]
            print(f"{task_name:<20} {agent:<25} {trial.pass_rate*100:>5.1f}% "
                  f"{trial.regression_rate*100:>5.1f}% {grade.overall_score:>7.1f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
