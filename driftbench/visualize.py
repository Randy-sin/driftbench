"""
DriftBench Visualize — Publication-quality visualization suite.

Key Features:
- Syntax validity markers on entropy trajectory
- Adjusted regression score visualization
- Multi-trial confidence interval bands
- Refactor Trap analysis chart
- Agent ranking leaderboard
- Pass@k comparison chart
- Improved color scheme and typography
"""

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np


# ── Color Palette ────────────────────────────────────────────────
COLORS = {
    "Naive Baseline": "#e74c3c",
    "gpt-4.1-mini": "#3498db",
    "gpt-4.1-nano": "#2ecc71",
    "gemini-2.5-flash": "#9b59b6",
    "react": "#e67e22",  # NEW: for ReAct agent
}

def _color(agent_name: str) -> str:
    for key, c in COLORS.items():
        if key in agent_name:
            return c
    return "#95a5a6"

def _short_name(agent_name: str) -> str:
    """Shorten agent names for chart labels."""
    return (agent_name
            .replace("LLM (", "")
            .replace(")", "")
            .replace("ReAct (", "")
            .replace("Naive Baseline", "Naive"))


# ── 1. Radar Chart ───────────────────────────────────────────────
def plot_radar_chart(grades: dict, output_path: str):
    """Radar chart comparing multiple agents across scoring dimensions."""
    categories = [
        "Functional", "Regression\n(Adjusted)", "Entropy",
        "Erosion", "Consistency", "Refactor\nAwareness", "Taste"
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"],
               color="grey", size=8)
    plt.ylim(0, 100)

    for agent_name, grade in grades.items():
        adj_reg = getattr(grade, 'adjusted_regression_score', grade.regression_score)
        values = [
            grade.functional_score, adj_reg,
            grade.entropy_score, grade.erosion_score,
            grade.consistency_score, grade.refactor_awareness_score,
            grade.taste_score
        ]
        values += values[:1]
        color = _color(agent_name)
        ax.plot(angles, values, 'o-', linewidth=2, label=_short_name(agent_name), color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.title("DriftBench Multi-Dimensional Score Comparison", size=14, y=1.08, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved radar chart: {output_path}")


# ── 2. Entropy Trajectory (with syntax markers) ─────────────────
def plot_entropy_trajectory(all_trajectories: dict, output_path: str):
    """
    Line chart showing per-step complexity evolution.
    Marks steps with syntax errors using red X markers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for agent_name, traj in all_trajectories.items():
        if not traj:
            continue
        steps = [t["step_id"] for t in traj]
        cc = [t["avg_cc"] for t in traj]
        erosion = [t["structural_erosion"] for t in traj]
        syntax_valid = [t.get("syntax_valid", True) for t in traj]
        color = _color(agent_name)
        label = _short_name(agent_name)

        ax1.plot(steps, cc, 'o-', label=label, color=color, linewidth=2, markersize=6)
        ax2.plot(steps, erosion, 's-', label=label, color=color, linewidth=2, markersize=6)

        # Mark syntax errors with red X
        for i, (s, c, e, sv) in enumerate(zip(steps, cc, erosion, syntax_valid)):
            if not sv:
                ax1.plot(s, c, 'X', color='red', markersize=12, zorder=5)
                ax2.plot(s, e, 'X', color='red', markersize=12, zorder=5)

    ax1.set_xlabel("Task Step", fontsize=11)
    ax1.set_ylabel("Average Cyclomatic Complexity", fontsize=11)
    ax1.set_title("Complexity Trajectory", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Task Step", fontsize=11)
    ax2.set_ylabel("Structural Erosion", fontsize=11)
    ax2.set_title("Structural Erosion Trajectory", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add legend entry for syntax error marker
    syntax_marker = plt.Line2D([], [], color='red', marker='X', linestyle='None',
                               markersize=10, label='Syntax Error')
    ax1.legend(handles=list(ax1.get_legend_handles_labels()[1]) + [syntax_marker],
               labels=list(ax1.get_legend_handles_labels()[0]) + ['Syntax Error'],
               fontsize=8)

    plt.suptitle("DriftBench Entropy Trajectory Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved entropy trajectory: {output_path}")


# ── 3. Step Progression Heatmap ──────────────────────────────────
def plot_step_heatmap(all_step_results: dict, output_path: str):
    """Heatmap showing pass/fail/regression status per step per agent."""
    agents = list(all_step_results.keys())
    if not agents:
        return

    max_steps = max(len(sr) for sr in all_step_results.values())
    data = np.zeros((len(agents), max_steps))

    for i, agent in enumerate(agents):
        for j, sr in enumerate(all_step_results[agent]):
            if sr.get("passed", False):
                data[i][j] = 2  # full pass
            elif sr.get("new_test_passed", False):
                data[i][j] = 1  # new passed but regression
            else:
                data[i][j] = 0  # fail

    fig, ax = plt.subplots(figsize=(max(8, max_steps * 1.5), max(3, len(agents) * 0.8)))

    cmap = mcolors.ListedColormap(['#e74c3c', '#f39c12', '#2ecc71'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

    short_names = [_short_name(a) for a in agents]
    ax.set_xticks(range(max_steps))
    ax.set_xticklabels([f"Step {i+1}" for i in range(max_steps)], fontsize=10)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(short_names, fontsize=10)

    for i in range(len(agents)):
        for j in range(max_steps):
            if j < len(all_step_results[agents[i]]):
                sr = all_step_results[agents[i]][j]
                reg = sr.get("regression_failures", 0)
                retry = sr.get("retry_count", 0)
                text = f"R:{reg}" if reg > 0 else ("P" if data[i][j] == 2 else "F")
                if retry > 0:
                    text += f"\n(r{retry})"
                ax.text(j, i, text, ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Full Pass'),
        mpatches.Patch(color='#f39c12', label='New Pass + Regression'),
        mpatches.Patch(color='#e74c3c', label='Fail'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))

    plt.title("Step-by-Step Progression Heatmap", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved step heatmap: {output_path}")


# ── 4. Overall Score Bar Chart ───────────────────────────────────
def plot_overall_comparison(grades: dict, output_path: str):
    """Grouped bar chart comparing overall scores and sub-dimensions."""
    agents = list(grades.keys())
    dimensions = ["Functional", "Regression\n(Adj)", "Entropy", "Erosion",
                   "Consistency", "Refactor", "Taste", "Overall"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(dimensions))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        g = grades[agent]
        adj_reg = getattr(g, 'adjusted_regression_score', g.regression_score)
        values = [
            g.functional_score, adj_reg, g.entropy_score,
            g.erosion_score, g.consistency_score, g.refactor_awareness_score,
            g.taste_score, g.overall_score
        ]
        offset = (i - len(agents)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, label=_short_name(agent),
                      color=_color(agent), alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=10)
    ax.set_ylabel("Score (0-100)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.title("DriftBench Overall Score Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved overall comparison: {output_path}")


# ── 5. Regression Waterfall ──────────────────────────────────────
def plot_regression_waterfall(all_step_results: dict, output_path: str):
    """Waterfall chart showing cumulative regression rate per step."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for agent_name, steps in all_step_results.items():
        cum_reg = []
        total_prev = 0
        total_fail = 0
        for sr in steps:
            prev = sr.get("total_previous_tests", 0)
            fail = sr.get("regression_failures", 0)
            total_prev += prev
            total_fail += fail
            rate = total_fail / max(total_prev, 1)
            cum_reg.append(rate)

        step_ids = list(range(1, len(cum_reg) + 1))
        color = _color(agent_name)
        ax.plot(step_ids, cum_reg, 'o-', label=_short_name(agent_name), color=color,
                linewidth=2, markersize=8)
        ax.fill_between(step_ids, cum_reg, alpha=0.1, color=color)

    ax.set_xlabel("Task Step", fontsize=11)
    ax.set_ylabel("Cumulative Regression Rate", fontsize=11)
    ax.set_title("Regression Accumulation Over Task Chain", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved regression waterfall: {output_path}")


# ── 6. Multi-Task Dashboard ─────────────────────────────────────
def plot_multi_task_dashboard(all_task_grades: dict, output_path: str):
    """Dashboard showing agent performance across multiple seed projects."""
    tasks = list(all_task_grades.keys())
    if not tasks:
        return

    all_agents = set()
    for task_grades in all_task_grades.values():
        all_agents.update(task_grades.keys())
    agents = sorted(all_agents)

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_grades = all_task_grades[task]
        agent_names = []
        overall_scores = []
        colors = []
        for agent in agents:
            if agent in task_grades:
                agent_names.append(_short_name(agent))
                overall_scores.append(task_grades[agent].overall_score)
                colors.append(_color(agent))

        bars = ax.barh(range(len(agent_names)), overall_scores, color=colors, alpha=0.85)
        ax.set_yticks(range(len(agent_names)))
        ax.set_yticklabels(agent_names, fontsize=9)
        ax.set_xlabel("Overall Score", fontsize=10)
        ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)

        for bar, score in zip(bars, overall_scores):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}', va='center', fontsize=9)

    plt.suptitle("DriftBench Multi-Task Performance Dashboard",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved multi-task dashboard: {output_path}")


# ── 7. Refactor Trap Analysis (NEW) ─────────────────────────────
def plot_refactor_trap(all_step_results: dict, output_path: str):
    """
    Visualize the "Refactor Trap" phenomenon: regression spike at refactor steps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for agent_name, steps in all_step_results.items():
        color = _color(agent_name)
        label = _short_name(agent_name)

        step_ids = []
        reg_rates = []
        task_types = []
        for sr in steps:
            step_ids.append(sr.get("step_id", 0))
            prev = sr.get("total_previous_tests", 0)
            fail = sr.get("regression_failures", 0)
            reg_rates.append(fail / max(prev, 1) if prev > 0 else 0)
            task_types.append(sr.get("task_type", ""))

        ax1.plot(step_ids, reg_rates, 'o-', label=label, color=color,
                 linewidth=2, markersize=8)

        # Highlight refactor steps
        for i, tt in enumerate(task_types):
            if tt == "refactor":
                ax1.axvline(x=step_ids[i], color='gray', linestyle='--', alpha=0.5)
                ax1.annotate('Refactor', xy=(step_ids[i], reg_rates[i]),
                             xytext=(step_ids[i]+0.2, reg_rates[i]+0.05),
                             fontsize=8, color='gray')

    ax1.set_xlabel("Task Step", fontsize=11)
    ax1.set_ylabel("Per-Step Regression Rate", fontsize=11)
    ax1.set_title("Regression Rate by Step", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Bar chart: average regression rate by task type
    type_reg = {}
    for agent_name, steps in all_step_results.items():
        for sr in steps:
            tt = sr.get("task_type", "unknown")
            prev = sr.get("total_previous_tests", 0)
            fail = sr.get("regression_failures", 0)
            if tt not in type_reg:
                type_reg[tt] = {"total_prev": 0, "total_fail": 0}
            type_reg[tt]["total_prev"] += prev
            type_reg[tt]["total_fail"] += fail

    types = list(type_reg.keys())
    rates = [type_reg[t]["total_fail"] / max(type_reg[t]["total_prev"], 1) for t in types]
    bar_colors = ['#e74c3c' if t == 'refactor' else '#3498db' for t in types]

    ax2.bar(types, rates, color=bar_colors, alpha=0.85)
    ax2.set_xlabel("Task Type", fontsize=11)
    ax2.set_ylabel("Average Regression Rate", fontsize=11)
    ax2.set_title("Regression Rate by Task Type\n(The Refactor Trap)", fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle("DriftBench Refactor Trap Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved refactor trap analysis: {output_path}")


# ── 8. Agent Ranking Leaderboard (NEW) ──────────────────────────
def plot_agent_ranking(all_task_grades: dict, output_path: str):
    """
    Horizontal bar chart ranking agents by average overall score across tasks.
    """
    agent_scores = {}
    for task, grades in all_task_grades.items():
        for agent, grade in grades.items():
            if agent not in agent_scores:
                agent_scores[agent] = []
            agent_scores[agent].append(grade.overall_score)

    # Compute average
    agent_avg = {a: sum(s)/len(s) for a, s in agent_scores.items()}
    sorted_agents = sorted(agent_avg.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(3, len(sorted_agents) * 0.8)))

    names = [_short_name(a) for a, _ in sorted_agents]
    scores = [s for _, s in sorted_agents]
    colors = [_color(a) for a, _ in sorted_agents]

    bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Average Overall Score", fontsize=11)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=10, fontweight='bold')

    # Add rank numbers
    for i, (name, score) in enumerate(zip(names, scores)):
        ax.text(2, i, f"#{i+1}", va='center', fontsize=10,
                fontweight='bold', color='white')

    plt.title("DriftBench Agent Ranking (Average Across Tasks)",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved agent ranking: {output_path}")


# ── 9. Score Comparison: Raw vs Adjusted Regression (NEW) ───────
def plot_regression_adjustment(grades: dict, output_path: str):
    """
    Show the impact of the regression score adjustment.
    Compares raw regression score vs adjusted regression score.
    """
    agents = list(grades.keys())
    short_names = [_short_name(a) for a in agents]

    raw_scores = [grades[a].regression_score for a in agents]
    adj_scores = [getattr(grades[a], 'adjusted_regression_score', grades[a].regression_score)
                  for a in agents]
    func_scores = [grades[a].functional_score for a in agents]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(agents))
    width = 0.25

    bars1 = ax.bar(x - width, raw_scores, width, label='Raw Regression Score',
                   color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, adj_scores, width, label='Adjusted Regression Score',
                   color='#e74c3c', alpha=0.85)
    bars3 = ax.bar(x + width, func_scores, width, label='Functional Score',
                   color='#2ecc71', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_ylabel("Score (0-100)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7)

    plt.title("Regression Score Adjustment\n(Discounted by Functional Performance)",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved regression adjustment chart: {output_path}")


# ── Save Results to JSON ─────────────────────────────────────────
def save_results_json(all_results: dict, output_path: str):
    """Save all results to a structured JSON file."""

    def _serialize(obj):
        if hasattr(obj, '__dict__'):
            d = {}
            for k, v in obj.__dict__.items():
                d[k] = _serialize(v)
            return d
        elif isinstance(obj, list):
            return [_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        else:
            return obj

    serialized = _serialize(all_results)
    with open(output_path, 'w') as f:
        json.dump(serialized, f, indent=2, default=str)
    print(f"  Saved results JSON: {output_path}")
