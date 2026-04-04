"""
DriftBench Visualizer — Generate radar charts and comparison plots
for benchmark results.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_radar_chart(grade_reports: dict, output_path: str):
    """
    Generate a radar chart comparing multiple agents across DriftBench dimensions.

    Args:
        grade_reports: Dict mapping agent_name -> GradeReport
        output_path: Path to save the PNG file
    """
    categories = [
        "Functional\nCorrectness",
        "Regression\nResistance",
        "Entropy\nResistance",
        "Architectural\nConsistency",
        "Refactor\nAwareness",
        "Engineering\nTaste",
    ]
    N = len(categories)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for idx, (agent_name, report) in enumerate(grade_reports.items()):
        values = [
            report.functional_score,
            report.regression_score,
            report.entropy_score,
            report.consistency_score,
            report.refactor_awareness_score,
            report.taste_score,
        ]
        values += values[:1]

        color = colors[idx % len(colors)]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=agent_name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=9, color="grey")
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.set_title("DriftBench: Agent Entropy Resistance Profile",
                 size=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Radar chart saved to {output_path}")


def plot_step_progression(trial_results: dict, output_path: str):
    """
    Plot how each agent's cumulative regression rate evolves across steps.

    Args:
        trial_results: Dict mapping agent_name -> TrialResult
        output_path: Path to save the PNG file
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    # Left plot: Cumulative regression rate per step
    ax1 = axes[0]
    for idx, (agent_name, trial) in enumerate(trial_results.items()):
        steps = []
        cum_regressions = []
        cum_total = []
        running_reg = 0
        running_total = 0

        for sr in trial.step_results:
            running_reg += sr.regression_failures
            running_total += sr.total_previous_tests
            steps.append(sr.step_id)
            rate = running_reg / max(running_total, 1) * 100
            cum_regressions.append(rate)

        color = colors[idx % len(colors)]
        ax1.plot(steps, cum_regressions, 'o-', linewidth=2.5,
                 label=agent_name, color=color, markersize=8)

    ax1.set_xlabel("Task Step", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cumulative Regression Rate (%)", fontsize=12, fontweight='bold')
    ax1.set_title("Regression Accumulation Over Task Chain", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Right plot: Pass/Fail heatmap
    ax2 = axes[1]
    agent_names = list(trial_results.keys())
    max_steps = max(len(t.step_results) for t in trial_results.values())

    heatmap_data = []
    for agent_name in agent_names:
        trial = trial_results[agent_name]
        row = []
        for sr in trial.step_results:
            if sr.passed:
                row.append(2)  # Full pass
            elif sr.new_test_passed:
                row.append(1)  # New test passed but regression
            else:
                row.append(0)  # Failed
        while len(row) < max_steps:
            row.append(-1)  # N/A
        heatmap_data.append(row)

    cmap = plt.cm.colors.ListedColormap(['#cccccc', '#FF6B6B', '#FFEAA7', '#4ECDC4'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax2.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')

    ax2.set_yticks(range(len(agent_names)))
    ax2.set_yticklabels(agent_names, fontsize=11)
    ax2.set_xticks(range(max_steps))
    ax2.set_xticklabels([f"Step {i+1}" for i in range(max_steps)], fontsize=10)
    ax2.set_title("Step-by-Step Results", fontsize=14, fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='#4ECDC4', label='Full Pass'),
        mpatches.Patch(color='#FFEAA7', label='New Pass + Regression'),
        mpatches.Patch(color='#FF6B6B', label='Failed'),
        mpatches.Patch(color='#cccccc', label='N/A'),
    ]
    ax2.legend(handles=legend_patches, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Progression chart saved to {output_path}")


def plot_overall_comparison(grade_reports: dict, output_path: str):
    """Bar chart comparing overall scores with dimension breakdown."""
    fig, ax = plt.subplots(figsize=(12, 6))

    agent_names = list(grade_reports.keys())
    dimensions = [
        ("Functional", "functional_score"),
        ("Regression", "regression_score"),
        ("Entropy", "entropy_score"),
        ("Consistency", "consistency_score"),
        ("Refactor", "refactor_awareness_score"),
        ("Taste", "taste_score"),
    ]

    x = np.arange(len(agent_names))
    width = 0.12
    dim_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

    for i, (dim_name, dim_key) in enumerate(dimensions):
        values = [getattr(grade_reports[a], dim_key) for a in agent_names]
        bars = ax.bar(x + i * width, values, width, label=dim_name,
                      color=dim_colors[i], edgecolor='white', linewidth=0.5)

    # Add overall score line
    overall_scores = [grade_reports[a].overall_score for a in agent_names]
    ax.plot(x + width * 2.5, overall_scores, 'k*-', markersize=15,
            linewidth=2, label='Overall', zorder=5)

    ax.set_xlabel("Agent", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score (0-100)", fontsize=12, fontweight='bold')
    ax.set_title("DriftBench: Multi-Dimensional Agent Comparison",
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(agent_names, fontsize=11)
    ax.legend(ncol=4, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison chart saved to {output_path}")


def save_results_json(trial_results: dict, grade_reports: dict, output_path: str):
    """Save all results to a JSON file for reproducibility."""
    data = {}
    for agent_name in trial_results:
        trial = trial_results[agent_name]
        grade = grade_reports[agent_name]
        data[agent_name] = {
            "trial": {
                "pass_rate": trial.pass_rate,
                "regression_rate": trial.regression_rate,
                "entropy_delta": trial.entropy_delta,
                "total_tokens": trial.total_tokens,
                "total_actions": trial.total_actions,
                "total_duration": trial.total_duration,
                "steps": [
                    {
                        "step_id": s.step_id,
                        "task_type": s.task_type,
                        "passed": s.passed,
                        "new_test_passed": s.new_test_passed,
                        "regression_failures": s.regression_failures,
                        "total_previous_tests": s.total_previous_tests,
                        "duration_seconds": s.duration_seconds,
                        "token_count": s.token_count,
                    }
                    for s in trial.step_results
                ]
            },
            "grade": {
                "functional_score": grade.functional_score,
                "regression_score": grade.regression_score,
                "entropy_score": grade.entropy_score,
                "consistency_score": grade.consistency_score,
                "refactor_awareness_score": grade.refactor_awareness_score,
                "taste_score": grade.taste_score,
                "overall_score": grade.overall_score,
                "token_efficiency": grade.token_efficiency,
            }
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results JSON saved to {output_path}")
