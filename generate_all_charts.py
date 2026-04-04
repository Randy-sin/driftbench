#!/usr/bin/env python3
"""
Generate comprehensive cross-task visualization dashboard from all experiment data.
"""
import json
import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Color palette ──────────────────────────────────────────────
COLORS = {
    "Naive Baseline": "#e74c3c",
    "gpt-4.1-mini": "#3498db",
    "gpt-4.1-nano": "#2ecc71",
    "gemini-2.5-flash": "#9b59b6",
}

def _color(agent_name):
    for key, c in COLORS.items():
        if key in agent_name:
            return c
    return "#95a5a6"

# ── Load all results ──────────────────────────────────────────
# Hardcoded from experiment runs (since each task was run separately)
DATA = {
    "todo_api": {
        "Naive Baseline":          {"pass": 60.0, "reg": 0.0,   "overall": 66.2, "func": 60.0, "reg_s": 100.0, "ent": 50.0, "ero": 40.0, "con": 45.0, "ref": 35.0, "taste": 40.0},
        "LLM (gpt-4.1-mini)":     {"pass": 60.0, "reg": 40.0,  "overall": 62.6, "func": 60.0, "reg_s": 60.0,  "ent": 50.0, "ero": 40.0, "con": 72.5, "ref": 72.5, "taste": 70.0},
        "LLM (gpt-4.1-nano)":     {"pass": 60.0, "reg": 55.6,  "overall": 56.4, "func": 60.0, "reg_s": 44.4,  "ent": 50.0, "ero": 40.0, "con": 72.5, "ref": 72.5, "taste": 70.0},
        "LLM (gemini-2.5-flash)": {"pass": 60.0, "reg": 40.0,  "overall": 65.1, "func": 60.0, "reg_s": 60.0,  "ent": 50.0, "ero": 40.0, "con": 72.5, "ref": 72.5, "taste": 70.0},
    },
    "calculator": {
        "Naive Baseline":          {"pass": 0.0,   "reg": 0.0,  "overall": 53.8, "func": 0.0,   "reg_s": 100.0, "ent": 50.0, "ero": 40.0, "con": 30.0, "ref": 27.5, "taste": 30.0},
        "LLM (gpt-4.1-mini)":     {"pass": 100.0, "reg": 0.0,  "overall": 84.7, "func": 100.0, "reg_s": 100.0, "ent": 80.0, "ero": 100.0,"con": 70.0, "ref": 72.5, "taste": 67.5},
        "LLM (gpt-4.1-nano)":     {"pass": 100.0, "reg": 0.0,  "overall": 81.9, "func": 100.0, "reg_s": 100.0, "ent": 80.0, "ero": 70.0, "con": 72.5, "ref": 70.0, "taste": 71.0},
        "LLM (gemini-2.5-flash)": {"pass": 100.0, "reg": 0.0,  "overall": 84.6, "func": 100.0, "reg_s": 100.0, "ent": 80.0, "ero": 100.0,"con": 72.5, "ref": 70.0, "taste": 71.0},
    },
    "markdown_parser": {
        "Naive Baseline":          {"pass": 0.0,  "reg": 0.0,   "overall": 51.9, "func": 0.0,  "reg_s": 100.0, "ent": 50.0, "ero": 40.0, "con": 35.0, "ref": 30.0, "taste": 32.5},
        "LLM (gpt-4.1-mini)":     {"pass": 80.0, "reg": 53.8,  "overall": 73.1, "func": 80.0, "reg_s": 46.2,  "ent": 50.0, "ero": 40.0, "con": 67.5, "ref": 65.0, "taste": 65.0},
        "LLM (gpt-4.1-nano)":     {"pass": 80.0, "reg": 53.8,  "overall": 73.0, "func": 80.0, "reg_s": 46.2,  "ent": 50.0, "ero": 40.0, "con": 67.5, "ref": 65.0, "taste": 67.5},
        "LLM (gemini-2.5-flash)": {"pass": 40.0, "reg": 85.7,  "overall": 55.1, "func": 40.0, "reg_s": 14.3,  "ent": 50.0, "ero": 40.0, "con": 67.5, "ref": 65.0, "taste": 67.5},
    },
    "file_manager": {
        "Naive Baseline":          {"pass": 0.0,  "reg": 0.0,   "overall": 50.8, "func": 0.0,  "reg_s": 100.0, "ent": 50.0, "ero": 40.0, "con": 30.0, "ref": 27.5, "taste": 30.0},
        "LLM (gpt-4.1-mini)":     {"pass": 40.0, "reg": 33.3,  "overall": 48.8, "func": 40.0, "reg_s": 66.7,  "ent": 50.0, "ero": 40.0, "con": 72.5, "ref": 70.0, "taste": 71.0},
        "LLM (gpt-4.1-nano)":     {"pass": 40.0, "reg": 33.3,  "overall": 50.9, "func": 40.0, "reg_s": 66.7,  "ent": 50.0, "ero": 40.0, "con": 72.5, "ref": 70.0, "taste": 70.0},
        "LLM (gemini-2.5-flash)": {"pass": 20.0, "reg": 100.0, "overall": 43.9, "func": 20.0, "reg_s": 0.0,   "ent": 50.0, "ero": 40.0, "con": 67.5, "ref": 65.0, "taste": 65.0},
    },
}

output_dir = Path("/home/ubuntu/driftbench/results")
output_dir.mkdir(parents=True, exist_ok=True)

agents = ["Naive Baseline", "LLM (gpt-4.1-mini)", "LLM (gpt-4.1-nano)", "LLM (gemini-2.5-flash)"]
tasks = ["todo_api", "calculator", "markdown_parser", "file_manager"]

# ── 1. Cross-Task Dashboard ──────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for idx, task in enumerate(tasks):
    ax = axes[idx]
    agent_labels = []
    scores = []
    colors = []
    for agent in agents:
        d = DATA[task][agent]
        agent_labels.append(agent.replace("LLM (", "").replace(")", ""))
        scores.append(d["overall"])
        colors.append(_color(agent))

    bars = ax.barh(range(len(agent_labels)), scores, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(agent_labels)))
    if idx == 0:
        ax.set_yticklabels(agent_labels, fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Overall Score", fontsize=10)
    ax.set_title(task.replace("_", " ").title(), fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', va='center', fontsize=9, fontweight='bold')

plt.suptitle("DriftBench v2.0 — Multi-Task Performance Dashboard",
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(str(output_dir / "multi_task_dashboard.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved multi_task_dashboard.png")


# ── 2. Aggregate Radar (averaged across tasks) ───────────────
categories = ["Functional", "Regression\nResistance", "Entropy\nResistance",
              "Structural\nErosion", "Consistency", "Refactor\nAwareness", "Taste"]
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
plt.ylim(0, 100)

for agent in agents:
    avg_func = np.mean([DATA[t][agent]["func"] for t in tasks])
    avg_reg = np.mean([DATA[t][agent]["reg_s"] for t in tasks])
    avg_ent = np.mean([DATA[t][agent]["ent"] for t in tasks])
    avg_ero = np.mean([DATA[t][agent]["ero"] for t in tasks])
    avg_con = np.mean([DATA[t][agent]["con"] for t in tasks])
    avg_ref = np.mean([DATA[t][agent]["ref"] for t in tasks])
    avg_taste = np.mean([DATA[t][agent]["taste"] for t in tasks])

    values = [avg_func, avg_reg, avg_ent, avg_ero, avg_con, avg_ref, avg_taste]
    values += values[:1]
    color = _color(agent)
    ax.plot(angles, values, 'o-', linewidth=2.5, label=agent, color=color, markersize=6)
    ax.fill(angles, values, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.title("DriftBench v2.0 — Aggregate Entropy Resistance Profile\n(Averaged Across 4 Seed Projects)",
          size=13, y=1.08, fontweight='bold')
plt.tight_layout()
plt.savefig(str(output_dir / "aggregate_radar.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved aggregate_radar.png")


# ── 3. Pass Rate vs Regression Rate Scatter ──────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for task in tasks:
    for agent in agents:
        d = DATA[task][agent]
        color = _color(agent)
        marker = {'todo_api': 'o', 'calculator': 's', 'markdown_parser': '^', 'file_manager': 'D'}[task]
        ax.scatter(d["pass"], d["reg"], c=color, marker=marker, s=150, alpha=0.8,
                   edgecolors='black', linewidth=0.5)

# Legend for agents (colors)
agent_patches = [mpatches.Patch(color=_color(a), label=a) for a in agents]
# Legend for tasks (markers)
from matplotlib.lines import Line2D
task_markers = [
    Line2D([0], [0], marker='o', color='grey', label='todo_api', markersize=10, linestyle='None'),
    Line2D([0], [0], marker='s', color='grey', label='calculator', markersize=10, linestyle='None'),
    Line2D([0], [0], marker='^', color='grey', label='markdown_parser', markersize=10, linestyle='None'),
    Line2D([0], [0], marker='D', color='grey', label='file_manager', markersize=10, linestyle='None'),
]
legend1 = ax.legend(handles=agent_patches, loc='upper left', title="Agent", fontsize=9)
ax.add_artist(legend1)
ax.legend(handles=task_markers, loc='upper right', title="Task", fontsize=9)

ax.set_xlabel("Pass Rate (%)", fontsize=12, fontweight='bold')
ax.set_ylabel("Regression Rate (%)", fontsize=12, fontweight='bold')
ax.set_title("DriftBench v2.0 — Pass Rate vs Regression Rate\n(Each dot = 1 task × 1 agent)",
             fontsize=13, fontweight='bold')
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)

# Add quadrant labels
ax.text(80, 10, "Ideal Zone\n(High Pass, Low Reg)", fontsize=9, color='green', alpha=0.7,
        ha='center', style='italic')
ax.text(20, 80, "Danger Zone\n(Low Pass, High Reg)", fontsize=9, color='red', alpha=0.7,
        ha='center', style='italic')

plt.tight_layout()
plt.savefig(str(output_dir / "pass_vs_regression_scatter.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved pass_vs_regression_scatter.png")


# ── 4. Task Difficulty Heatmap ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

# Build matrix: rows=agents, cols=tasks, values=overall scores
matrix = np.zeros((len(agents), len(tasks)))
for i, agent in enumerate(agents):
    for j, task in enumerate(tasks):
        matrix[i][j] = DATA[task][agent]["overall"]

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=30, vmax=90)
ax.set_xticks(range(len(tasks)))
ax.set_xticklabels([t.replace("_", " ").title() for t in tasks], fontsize=11)
ax.set_yticks(range(len(agents)))
ax.set_yticklabels(agents, fontsize=10)

# Add text annotations
for i in range(len(agents)):
    for j in range(len(tasks)):
        text_color = "white" if matrix[i][j] < 55 or matrix[i][j] > 80 else "black"
        ax.text(j, i, f'{matrix[i][j]:.1f}', ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold")

plt.colorbar(im, ax=ax, label="Overall Score", shrink=0.8)
plt.title("DriftBench v2.0 — Task Difficulty × Agent Performance Heatmap",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(output_dir / "task_difficulty_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved task_difficulty_heatmap.png")


# ── 5. Refactor Trap Analysis (Step 4 regression across tasks) ──
fig, ax = plt.subplots(figsize=(10, 6))

# Step 4 is the refactor step in all tasks
# Show regression rates at step 4 specifically
step4_data = {
    "todo_api": {"gpt-4.1-mini": 2/3, "gpt-4.1-nano": 3/3, "gemini-2.5-flash": 2/3},
    "calculator": {"gpt-4.1-mini": 0/3, "gpt-4.1-nano": 0/3, "gemini-2.5-flash": 0/3},
    "markdown_parser": {"gpt-4.1-mini": 0/3, "gpt-4.1-nano": 0/3, "gemini-2.5-flash": 2/2},
    "file_manager": {"gpt-4.1-mini": 1/2, "gpt-4.1-nano": 1/2, "gemini-2.5-flash": 1/1},
}

x = np.arange(len(tasks))
width = 0.25
llm_agents = ["gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash"]

for i, model in enumerate(llm_agents):
    values = [step4_data[task][model] * 100 for task in tasks]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width * 0.9,
                  label=model, color=_color(model), alpha=0.85)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([t.replace("_", " ").title() for t in tasks], fontsize=11)
ax.set_ylabel("Step 4 (Refactor) Regression Rate (%)", fontsize=11)
ax.set_title("DriftBench v2.0 — The Refactor Trap\n(Regression Rate at Step 4 Across Tasks)",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(str(output_dir / "refactor_trap_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved refactor_trap_analysis.png")


# ── 6. Model Ranking Summary ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

# Average overall score per agent across tasks
avg_scores = {}
for agent in agents:
    avg_scores[agent] = np.mean([DATA[t][agent]["overall"] for t in tasks])

sorted_agents = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
names = [a[0] for a in sorted_agents]
scores = [a[1] for a in sorted_agents]
colors = [_color(a) for a in names]

bars = ax.barh(range(len(names)), scores, color=colors, alpha=0.85, height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel("Average Overall Score (across 4 tasks)", fontsize=11)
ax.set_xlim(0, 100)
ax.grid(axis='x', alpha=0.3)

for bar, score in zip(bars, scores):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{score:.1f}', va='center', fontsize=11, fontweight='bold')

# Add rank badges
for i, (name, score) in enumerate(sorted_agents):
    medal = ["#1", "#2", "#3", "#4"][i]
    ax.text(2, i, medal, fontsize=12, va='center', fontweight='bold', color='white')

plt.title("DriftBench v2.0 — Agent Ranking\n(Average Overall Score Across 4 Seed Projects)",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(str(output_dir / "agent_ranking.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved agent_ranking.png")

print("\n✅ All 6 cross-task charts generated successfully!")
