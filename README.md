# DriftBench

**DriftBench** is an evaluation benchmark designed to measure how well AI coding agents maintain code quality, resist entropy, and avoid regressions over long-term, multi-step development cycles.

While most benchmarks (like SWE-bench or HumanEval) focus on single-shot problem solving, DriftBench simulates the real-world software lifecycle: **Feature → Bugfix → Feature → Refactor → Evolution**.

## Key Features

- **ReAct Multi-Turn Agents**: Agents can read files, write code, run tests, and search the codebase iteratively — simulating real-world agent behavior (like Devin or Cursor).
- **Error Feedback Loop**: Agents receive test failure outputs and can retry their modifications up to `max_retries` times.
- **Syntax Validation & Penalty**: Agents generating unparseable code are penalized rather than rewarded. Syntax errors incur heavy score deductions.
- **Effective Regression Rate**: Regression scores are discounted if the agent fails to implement new features, preventing the "do nothing" loophole.
- **Diff-Aware LLM Judge**: The LLM-as-a-Judge sees the *evolution* of the code (diffs between steps) rather than just the final state, enabling accurate assessment of refactoring quality.
- **Multi-Trial & Pass@k**: Support for running multiple trials per agent to compute statistically significant `pass@k` metrics and confidence intervals.

## The "Refactor Trap"

DriftBench reveals a critical vulnerability in current LLMs: **The Refactor Trap**. 
When asked to refactor code (Step 4 in our task chains), agents frequently break existing functionality, with regression rates spiking up to 100%. DriftBench specifically measures an agent's ability to navigate this trap.

## Installation

```bash
git clone https://github.com/Randy-sin/driftbench.git
cd driftbench
pip install -r requirements.txt
```

Set your OpenAI API key (used for both the agents and the LLM-as-a-Judge):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Benchmark

### Basic Run (Single-Shot Agents)
```bash
python run_benchmark.py --tasks todo_api calculator
```

### Advanced Run (ReAct Agents with Retries)
```bash
python run_benchmark.py --tasks todo_api --agent-type react --max-turns 10 --max-retries 2
```

### Multi-Trial Evaluation (Pass@k)
```bash
python run_benchmark.py --tasks todo_api --num-trials 3
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tasks` | `todo_api` | Tasks to run (e.g., `todo_api`, `calculator`, `markdown_parser`, `file_manager`) |
| `--models` | `gpt-4.1-mini gpt-4.1-nano gemini-2.5-flash` | LLM models to evaluate |
| `--agent-type` | `single-shot` | `single-shot` or `react` (multi-turn) |
| `--max-turns` | `10` | Max tool-use turns for ReAct agent |
| `--max-retries` | `0` | Max retries with error feedback per step |
| `--num-trials` | `1` | Number of trials per agent for pass@k |
| `--output-dir` | `results` | Directory for results and charts |
| `--no-llm-judge` | — | Skip LLM-as-a-Judge evaluation |
| `--no-charts` | — | Skip chart generation |

## Grading Dimensions

DriftBench evaluates agents across **7 dimensions**, combined into a single **Overall Score (0-100)**:

**Code-Based Metrics:**
1. **Functional Score** — Percentage of steps where the agent successfully implemented the requested feature/fix.
2. **Regression Score (Adjusted)** — Resistance to breaking previously passing tests, discounted if the agent fails to implement new features.
3. **Entropy Score** — Resistance to unnecessary increases in Cyclomatic Complexity (CC).
4. **Erosion Score** — Resistance to "Structural Erosion" (concentration of complexity in top-heavy functions).

**LLM-as-a-Judge Metrics (Cross-Validated):**
5. **Consistency Score** — Adherence to consistent style and architectural patterns.
6. **Refactor Awareness** — Proactive improvement of code structure and reduction of duplication.
7. **Taste Score** — General engineering judgment (error handling, naming, separation of concerns).

## Project Structure

```
driftbench/
├── driftbench/
│   ├── harness.py       # Core execution engine, sandbox isolation, metric collection
│   ├── agents.py        # Agent implementations (Naive, SingleShot, ReAct)
│   ├── grader.py        # Multi-dimensional scoring system and LLM Judge
│   └── visualize.py     # Publication-quality chart generation
├── tasks/               # Seed projects and task chains
├── run_benchmark.py     # Main CLI entry point
└── generate_all_charts.py  # Standalone chart regeneration
```

## License

MIT License
