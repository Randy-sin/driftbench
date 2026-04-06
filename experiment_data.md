# DriftBench Experiment Results

## Raw Data Summary (4 tasks × 4 agents = 16 experiments)

### todo_api
| Agent | Pass% | Reg% | Overall |
|-------|-------|------|---------|
| Naive Baseline | 60.0% | 0.0% | 66.2 |
| LLM (gpt-4.1-mini) | 60.0% | 40.0% | 62.6 |
| LLM (gpt-4.1-nano) | 60.0% | 55.6% | 56.4 |
| LLM (gemini-2.5-flash) | 60.0% | 40.0% | 65.1 |

### calculator
| Agent | Pass% | Reg% | Overall |
|-------|-------|------|---------|
| Naive Baseline | 0.0% | 0.0% | 53.8 |
| LLM (gpt-4.1-mini) | 100.0% | 0.0% | 84.7 |
| LLM (gpt-4.1-nano) | 100.0% | 0.0% | 81.9 |
| LLM (gemini-2.5-flash) | 100.0% | 0.0% | 84.6 |

### markdown_parser
| Agent | Pass% | Reg% | Overall |
|-------|-------|------|---------|
| Naive Baseline | 0.0% | 0.0% | 51.9 |
| LLM (gpt-4.1-mini) | 80.0% | 53.8% | 73.1 |
| LLM (gpt-4.1-nano) | 80.0% | 53.8% | 73.0 |
| LLM (gemini-2.5-flash) | 40.0% | 85.7% | 55.1 |

### file_manager
| Agent | Pass% | Reg% | Overall |
|-------|-------|------|---------|
| Naive Baseline | 0.0% | 0.0% | 50.8 |
| LLM (gpt-4.1-mini) | 40.0% | 33.3% | 48.8 |
| LLM (gpt-4.1-nano) | 40.0% | 33.3% | 50.9 |
| LLM (gemini-2.5-flash) | 20.0% | 100.0% | 43.9 |

## Key Findings
1. **Refactor Trap confirmed**: In todo_api, ALL LLM agents hit Step 4 regression (refactor step)
2. **Task difficulty varies**: calculator (100% pass for LLMs) vs file_manager (20-40% pass)
3. **Model differentiation**: gemini-2.5-flash shows highest regression rates on complex tasks
4. **Naive baseline paradox**: 0% regression because it fails silently (no previous tests pass)
