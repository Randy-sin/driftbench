#!/usr/bin/env python3
"""
DriftBench — Multi-task, multi-model, multi-trial benchmark runner.

Usage:
    # Basic: run all tasks with default models
    python run_benchmark.py --tasks todo_api calculator markdown_parser file_manager

    # Multi-trial with pass@k
    python run_benchmark.py --tasks todo_api --num-trials 3

    # ReAct agent mode
    python run_benchmark.py --tasks todo_api --agent-type react --max-turns 10

    # With error feedback retries
    python run_benchmark.py --tasks todo_api --max-retries 2

    # Baseline only
    python run_benchmark.py --tasks todo_api --no-llm-agent

    # Specific models
    python run_benchmark.py --tasks todo_api --models gpt-4.1-mini gpt-4.1-nano
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from driftbench.harness import DriftBenchHarness, TaskStep
from driftbench.agents import NaiveAppendAgent, LLMCodingAgent, ReActCodingAgent, create_agent
from driftbench.grader import DriftBenchGrader
from driftbench.visualize import (
    plot_radar_chart, plot_entropy_trajectory, plot_step_heatmap,
    plot_overall_comparison, plot_regression_waterfall,
    plot_multi_task_dashboard, plot_refactor_trap, plot_agent_ranking,
    plot_regression_adjustment, save_results_json
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


def _serialize_step(sr) -> dict:
    """Serialize a StepResult to a dict for JSON output."""
    d = {
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
        "action_count": sr.action_count,
        "retry_count": getattr(sr, 'retry_count', 0),
    }
    if sr.complexity_snapshot:
        d["syntax_valid"] = sr.complexity_snapshot.syntax_valid
    return d


def _serialize_grade(g) -> dict:
    """Serialize a GradeReport to a dict for JSON output."""
    return {
        "functional_score": g.functional_score,
        "regression_score": g.regression_score,
        "adjusted_regression_score": getattr(g, 'adjusted_regression_score', g.regression_score),
        "entropy_score": g.entropy_score,
        "erosion_score": g.erosion_score,
        "consistency_score": g.consistency_score,
        "refactor_awareness_score": g.refactor_awareness_score,
        "taste_score": g.taste_score,
        "overall_score": g.overall_score,
        "judge_agreement": g.judge_agreement,
        "token_efficiency": g.token_efficiency,
        "syntax_penalty": getattr(g, 'syntax_penalty', 0),
    }


def run_single_task(task_name, task_dir, models, output_dir,
                    agent_type="single-shot", max_turns=10,
                    max_retries=0, num_trials=1,
                    use_llm_judge=True, judge_models=None,
                    run_naive=True, run_llm=True):
    """Run benchmark on a single task with multiple models."""
    print(f"\n{'#'*70}")
    print(f"  TASK: {task_name}")
    print(f"  Agent type: {agent_type} | Trials: {num_trials} | Max retries: {max_retries}")
    print(f"{'#'*70}")

    task_chain = load_task_chain(task_dir)
    task_descriptions = [s.instruction for s in task_chain]

    trial_results = {}
    grade_reports = {}
    all_step_results = {}
    all_trajectories = {}
    multi_trial_stats = {}

    grader = DriftBenchGrader(use_llm_judge=use_llm_judge, judge_models=judge_models)

    # --- Naive Baseline ---
    if run_naive:
        print(f"\n>>> Running Naive Baseline on {task_name}...")
        naive_agent = NaiveAppendAgent()
        harness = DriftBenchHarness(
            base_project_dir=task_dir,
            task_chain=task_chain,
            agent_fn=naive_agent,
            agent_name="Naive Baseline",
            max_retries=0,  # no retries for baseline
        )
        trial = harness.run_trial()
        trial_results["Naive Baseline"] = trial
        all_step_results["Naive Baseline"] = [_serialize_step(sr) for sr in trial.step_results]
        all_trajectories["Naive Baseline"] = trial.entropy_trajectory

        sandbox_dir = str(harness.sandbox_dir / "project") if harness.sandbox_dir else None
        grade = grader.grade_trial(trial, sandbox_dir, task_descriptions)
        grade_reports["Naive Baseline"] = grade
        harness.cleanup()

        print(f"\n  Naive Baseline: pass_rate={trial.pass_rate:.2f}, "
              f"regression_rate={trial.regression_rate:.2f}, "
              f"eff_regression_rate={trial.effective_regression_rate:.2f}, "
              f"overall={grade.overall_score:.1f}")

    # --- LLM Agents ---
    if run_llm:
        for model in models:
            if agent_type == "react":
                agent_name = f"ReAct ({model})"
            else:
                agent_name = f"LLM ({model})"
            print(f"\n>>> Running {agent_name} on {task_name}...")

            try:
                agent = create_agent(agent_type, model=model, max_turns=max_turns)

                if num_trials > 1:
                    # Multi-trial mode
                    harness = DriftBenchHarness(
                        base_project_dir=task_dir,
                        task_chain=task_chain,
                        agent_fn=agent,
                        agent_name=agent_name,
                        max_retries=max_retries,
                    )
                    multi_result = harness.run_multi_trial(num_trials=num_trials)

                    # Use best trial for reporting
                    trial = multi_result.best_trial
                    trial_results[agent_name] = trial
                    all_step_results[agent_name] = [_serialize_step(sr) for sr in trial.step_results]
                    all_trajectories[agent_name] = trial.entropy_trajectory

                    # Grade best trial
                    sandbox_dir = str(harness.sandbox_dir / "project") if harness.sandbox_dir else None
                    grade = grader.grade_trial(trial, sandbox_dir, task_descriptions)
                    grade_reports[agent_name] = grade

                    multi_trial_stats[agent_name] = {
                        "num_trials": multi_result.num_trials,
                        "pass_at_1": multi_result.pass_at_1,
                        "pass_at_k": multi_result.pass_at_k,
                        "mean_pass_rate": multi_result.mean_pass_rate,
                        "std_pass_rate": multi_result.std_pass_rate,
                        "mean_regression_rate": multi_result.mean_regression_rate,
                        "std_regression_rate": multi_result.std_regression_rate,
                    }

                    print(f"\n  {agent_name} (best of {num_trials}): "
                          f"pass_rate={trial.pass_rate:.2f}, "
                          f"regression_rate={trial.regression_rate:.2f}, "
                          f"pass@1={multi_result.pass_at_1:.2f}, "
                          f"overall={grade.overall_score:.1f}")
                else:
                    # Single trial mode
                    harness = DriftBenchHarness(
                        base_project_dir=task_dir,
                        task_chain=task_chain,
                        agent_fn=agent,
                        agent_name=agent_name,
                        max_retries=max_retries,
                    )
                    trial = harness.run_trial()
                    trial_results[agent_name] = trial
                    all_step_results[agent_name] = [_serialize_step(sr) for sr in trial.step_results]
                    all_trajectories[agent_name] = trial.entropy_trajectory

                    sandbox_dir = str(harness.sandbox_dir / "project") if harness.sandbox_dir else None
                    grade = grader.grade_trial(trial, sandbox_dir, task_descriptions)
                    grade_reports[agent_name] = grade
                    harness.cleanup()

                    eff_reg = trial.effective_regression_rate
                    syntax_err = trial.syntax_error_count
                    print(f"\n  {agent_name}: pass_rate={trial.pass_rate:.2f}, "
                          f"regression_rate={trial.regression_rate:.2f}, "
                          f"eff_regression_rate={eff_reg:.2f}, "
                          f"syntax_errors={syntax_err}, "
                          f"overall={grade.overall_score:.1f}")

            except Exception as e:
                print(f"\n  ERROR running {agent_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return trial_results, grade_reports, all_step_results, all_trajectories, multi_trial_stats


def main():
    parser = argparse.ArgumentParser(description="DriftBench Benchmark Runner")

    # Task selection
    parser.add_argument("--tasks", nargs="+", default=["todo_api"],
                        help="Task directories to run (under tasks/)")

    # Model selection
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"],
                        help="LLM models to evaluate")

    # Agent configuration
    parser.add_argument("--agent-type", choices=["single-shot", "react"],
                        default="single-shot",
                        help="Agent type: single-shot or react (multi-turn)")
    parser.add_argument("--max-turns", type=int, default=10,
                        help="Max interaction turns for ReAct agent")
    parser.add_argument("--max-retries", type=int, default=0,
                        help="Max retries with error feedback per step")

    # Multi-trial
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Number of trials per agent for pass@k computation")

    # Output
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results and charts")

    # Flags
    parser.add_argument("--no-llm-agent", action="store_true",
                        help="Skip LLM agents, run only naive baseline")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip LLM-as-Judge grading")
    parser.add_argument("--judge-models", nargs="+",
                        default=["gpt-4.1-mini", "gemini-2.5-flash"],
                        help="Models for LLM-as-Judge cross-validation")
    parser.add_argument("--no-naive", action="store_true",
                        help="Skip naive baseline")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_base = Path(__file__).parent / "tasks"

    # Collect results across all tasks
    all_task_grades = {}
    all_task_trials = {}
    all_task_steps = {}
    all_task_trajectories = {}
    all_multi_trial_stats = {}

    start_time = time.time()

    for task_name in args.tasks:
        task_dir = str(tasks_base / task_name)
        if not Path(task_dir).exists():
            print(f"WARNING: Task directory not found: {task_dir}")
            continue

        trials, grades, steps, trajs, multi_stats = run_single_task(
            task_name=task_name,
            task_dir=task_dir,
            models=args.models,
            output_dir=str(output_dir),
            agent_type=args.agent_type,
            max_turns=args.max_turns,
            max_retries=args.max_retries,
            num_trials=args.num_trials,
            use_llm_judge=not args.no_llm_judge,
            judge_models=args.judge_models,
            run_naive=not args.no_naive,
            run_llm=not args.no_llm_agent,
        )

        all_task_grades[task_name] = grades
        all_task_trials[task_name] = trials
        all_task_steps[task_name] = steps
        all_task_trajectories[task_name] = trajs
        all_multi_trial_stats[task_name] = multi_stats

        # Per-task charts
        if not args.no_charts:
            task_output = output_dir / task_name
            task_output.mkdir(parents=True, exist_ok=True)

            if grades:
                print(f"\n  Generating charts for {task_name}...")
                plot_radar_chart(grades, str(task_output / "radar_chart.png"))
                plot_overall_comparison(grades, str(task_output / "overall_comparison.png"))
                plot_regression_adjustment(grades, str(task_output / "regression_adjustment.png"))

            if trajs:
                plot_entropy_trajectory(trajs, str(task_output / "entropy_trajectory.png"))

            if steps:
                plot_step_heatmap(steps, str(task_output / "step_heatmap.png"))
                plot_regression_waterfall(steps, str(task_output / "regression_waterfall.png"))
                plot_refactor_trap(steps, str(task_output / "refactor_trap.png"))

    # Cross-task charts
    if not args.no_charts and len(all_task_grades) > 0:
        print(f"\n  Generating cross-task charts...")
        plot_multi_task_dashboard(all_task_grades, str(output_dir / "multi_task_dashboard.png"))
        plot_agent_ranking(all_task_grades, str(output_dir / "agent_ranking.png"))

    # Save all results
    results_data = {
        "metadata": {
            "version": "latest",
            "agent_type": args.agent_type,
            "max_retries": args.max_retries,
            "num_trials": args.num_trials,
            "max_turns": args.max_turns if args.agent_type == "react" else None,
            "total_duration_seconds": round(time.time() - start_time, 1),
        },
        "results": {
            task_name: {
                agent: {
                    "pass_rate": t.pass_rate,
                    "regression_rate": t.regression_rate,
                    "effective_regression_rate": getattr(t, 'effective_regression_rate', t.regression_rate),
                    "syntax_error_count": getattr(t, 'syntax_error_count', 0),
                    "entropy_delta": t.entropy_delta,
                    "entropy_trajectory": t.entropy_trajectory,
                    "total_tokens": t.total_tokens,
                    "total_actions": t.total_actions,
                    "total_duration": t.total_duration,
                    "step_results": all_task_steps[task_name].get(agent, []),
                }
                for agent, t in all_task_trials[task_name].items()
            }
            for task_name in all_task_grades
        },
        "grades": {
            task_name: {
                agent: _serialize_grade(g)
                for agent, g in all_task_grades[task_name].items()
            }
            for task_name in all_task_grades
        },
    }

    # Add multi-trial stats if available
    if any(all_multi_trial_stats.values()):
        results_data["multi_trial_stats"] = all_multi_trial_stats

    save_results_json(results_data, str(output_dir / "results.json"))

    # Print summary table
    total_time = time.time() - start_time
    print(f"\n{'='*90}")
    print(f"  DRIFTBENCH RESULTS SUMMARY")
    print(f"  Agent type: {args.agent_type} | Retries: {args.max_retries} | "
          f"Trials: {args.num_trials} | Time: {total_time:.0f}s")
    print(f"{'='*90}")
    header = (f"{'Task':<18} {'Agent':<25} {'Pass%':>6} {'Reg%':>6} "
              f"{'EffReg%':>7} {'SynErr':>6} {'Overall':>8}")
    print(header)
    print(f"{'-'*90}")
    for task_name in all_task_grades:
        for agent, grade in all_task_grades[task_name].items():
            trial = all_task_trials[task_name][agent]
            eff_reg = getattr(trial, 'effective_regression_rate', trial.regression_rate)
            syn_err = getattr(trial, 'syntax_error_count', 0)
            print(f"{task_name:<18} {agent:<25} {trial.pass_rate*100:>5.1f}% "
                  f"{trial.regression_rate*100:>5.1f}% {eff_reg*100:>6.1f}% "
                  f"{syn_err:>6} {grade.overall_score:>7.1f}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
