#!/usr/bin/env python3
"""
DriftBench Chart Generator

Generates all publication-quality charts from results.json.
Adds refactor trap analysis, regression adjustment, and agent ranking.
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from driftbench.visualize import (
    plot_radar_chart,
    plot_entropy_trajectory,
    plot_step_heatmap,
    plot_overall_comparison,
    plot_regression_waterfall,
    plot_multi_task_dashboard,
    plot_refactor_trap,
    plot_agent_ranking,
    plot_regression_adjustment,
)


@dataclass
class GradeProxy:
    """Proxy object to mimic GradeReport from results.json data."""
    functional_score: float = 0.0
    regression_score: float = 0.0
    adjusted_regression_score: float = 0.0
    entropy_score: float = 0.0
    erosion_score: float = 0.0
    consistency_score: float = 0.0
    refactor_awareness_score: float = 0.0
    taste_score: float = 0.0
    overall_score: float = 0.0
    judge_agreement: float = 0.0
    token_efficiency: float = 0.0
    syntax_penalty: float = 0.0


def load_results(results_path: str) -> dict:
    with open(results_path, 'r') as f:
        return json.load(f)


def main():
    results_dir = Path(__file__).parent / "results"
    results_path = results_dir / "results.json"

    if not results_path.exists():
        print(f"Error: {results_path} not found. Run the benchmark first.")
        sys.exit(1)

    data = load_results(str(results_path))
    os.makedirs(results_dir, exist_ok=True)

    all_task_grades = {}

    for task_name, task_data in data.get("results", {}).items():
        print(f"\n{'='*60}")
        print(f"  Generating charts for: {task_name}")
        print(f"{'='*60}")

        task_dir = results_dir / task_name
        os.makedirs(task_dir, exist_ok=True)

        # Build grade proxies
        grades = {}
        if "grades" in data and task_name in data["grades"]:
            for agent_name, grade_data in data["grades"][task_name].items():
                g = GradeProxy(
                    functional_score=grade_data.get("functional_score", 0),
                    regression_score=grade_data.get("regression_score", 0),
                    adjusted_regression_score=grade_data.get("adjusted_regression_score",
                                                             grade_data.get("regression_score", 0)),
                    entropy_score=grade_data.get("entropy_score", 0),
                    erosion_score=grade_data.get("erosion_score", 0),
                    consistency_score=grade_data.get("consistency_score", 0),
                    refactor_awareness_score=grade_data.get("refactor_awareness_score", 0),
                    taste_score=grade_data.get("taste_score", 0),
                    overall_score=grade_data.get("overall_score", 0),
                    judge_agreement=grade_data.get("judge_agreement", 0),
                    token_efficiency=grade_data.get("token_efficiency", 0),
                    syntax_penalty=grade_data.get("syntax_penalty", 0),
                )
                grades[agent_name] = g

        all_task_grades[task_name] = grades

        # Build step results and trajectories
        all_step_results = {}
        all_trajectories = {}

        for agent_name, agent_data in task_data.items():
            steps = agent_data.get("step_results", agent_data.get("steps", []))
            if isinstance(steps, list):
                all_step_results[agent_name] = steps

            traj = agent_data.get("entropy_trajectory", [])
            if isinstance(traj, list):
                all_trajectories[agent_name] = traj

        # Generate per-task charts
        if grades:
            plot_radar_chart(grades, str(task_dir / "radar_chart.png"))
            plot_overall_comparison(grades, str(task_dir / "overall_comparison.png"))
            plot_regression_adjustment(grades, str(task_dir / "regression_adjustment.png"))

        if all_trajectories:
            plot_entropy_trajectory(all_trajectories, str(task_dir / "entropy_trajectory.png"))

        if all_step_results:
            plot_step_heatmap(all_step_results, str(task_dir / "step_heatmap.png"))
            plot_regression_waterfall(all_step_results, str(task_dir / "regression_waterfall.png"))
            plot_refactor_trap(all_step_results, str(task_dir / "refactor_trap.png"))

    # Generate cross-task charts
    print(f"\n{'='*60}")
    print(f"  Generating cross-task charts")
    print(f"{'='*60}")

    if all_task_grades:
        plot_multi_task_dashboard(all_task_grades, str(results_dir / "multi_task_dashboard.png"))
        plot_agent_ranking(all_task_grades, str(results_dir / "agent_ranking.png"))

    print(f"\nAll charts generated in: {results_dir}")


if __name__ == "__main__":
    main()
