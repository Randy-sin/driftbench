#!/usr/bin/env python3
"""
DriftBench Runner — Execute the benchmark with multiple agents and generate reports.

Usage:
    python run_benchmark.py                    # Run with naive baseline only
    python run_benchmark.py --with-llm         # Run with both naive and LLM agent
    python run_benchmark.py --model gpt-4.1-mini  # Specify LLM model
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from driftbench.harness import DriftBenchHarness, TaskStep
from driftbench.grader import DriftBenchGrader
from driftbench.agents import NaiveAppendAgent, LLMCodingAgent
from driftbench.visualize import (
    plot_radar_chart,
    plot_step_progression,
    plot_overall_comparison,
    save_results_json,
)


def load_task_chain(task_dir: str) -> list[TaskStep]:
    """Load task chain from JSON definition."""
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
        ))
    return steps


def run_agent(agent_name: str, agent_fn, task_dir: str, task_chain: list[TaskStep],
              grader: DriftBenchGrader) -> tuple:
    """Run a single agent through the benchmark."""
    print(f"\n{'#'*70}")
    print(f"  Running DriftBench: {agent_name}")
    print(f"{'#'*70}")

    harness = DriftBenchHarness(
        base_project_dir=task_dir,
        task_chain=task_chain,
        agent_fn=agent_fn,
        agent_name=agent_name,
    )

    trial_result = harness.run_trial()

    # Grade the trial
    task_descriptions = [s.instruction for s in task_chain]
    project_dir = str(harness.sandbox_dir / "project") if harness.sandbox_dir else None

    grade_report = grader.grade_trial(
        trial_result,
        project_dir=project_dir,
        task_descriptions=task_descriptions,
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"  {agent_name} — Summary")
    print(f"{'='*50}")
    print(f"  Pass Rate:        {trial_result.pass_rate:.0%}")
    print(f"  Regression Rate:  {trial_result.regression_rate:.0%}")
    print(f"  Entropy Delta:    CC {trial_result.entropy_delta.get('cc_delta', 'N/A')}")
    print(f"  Overall Score:    {grade_report.overall_score}/100")
    print(f"{'='*50}")

    harness.cleanup()
    return trial_result, grade_report


def main():
    parser = argparse.ArgumentParser(description="DriftBench — Coding Agent Entropy Resistance Benchmark")
    parser.add_argument("--with-llm", action="store_true", help="Include LLM-powered agent")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model to use")
    parser.add_argument("--task-dir", default="tasks/todo_api", help="Task directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM-as-Judge grading")
    args = parser.parse_args()

    task_dir = str(Path(__file__).parent / args.task_dir)
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    task_chain = load_task_chain(task_dir)
    grader = DriftBenchGrader(use_llm_judge=not args.no_llm_judge, model=args.model)

    trial_results = {}
    grade_reports = {}

    # --- Run Naive Baseline ---
    naive_agent = NaiveAppendAgent()
    trial, grade = run_agent("Naive Baseline", naive_agent, task_dir, task_chain, grader)
    trial_results["Naive Baseline"] = trial
    grade_reports["Naive Baseline"] = grade

    # --- Run LLM Agent ---
    if args.with_llm:
        try:
            llm_agent = LLMCodingAgent(model=args.model)
            trial, grade = run_agent(f"LLM ({args.model})", llm_agent, task_dir, task_chain, grader)
            trial_results[f"LLM ({args.model})"] = trial
            grade_reports[f"LLM ({args.model})"] = grade
        except Exception as e:
            print(f"\nWarning: LLM agent failed: {e}")

    # --- Generate Visualizations ---
    print(f"\n{'#'*70}")
    print("  Generating Visualizations")
    print(f"{'#'*70}")

    plot_radar_chart(grade_reports, str(output_dir / "radar_chart.png"))
    plot_step_progression(trial_results, str(output_dir / "step_progression.png"))
    plot_overall_comparison(grade_reports, str(output_dir / "overall_comparison.png"))
    save_results_json(trial_results, grade_reports, str(output_dir / "results.json"))

    print(f"\nAll results saved to {output_dir}/")
    print("\nDriftBench evaluation complete!")


if __name__ == "__main__":
    main()
