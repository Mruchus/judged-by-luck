# Judged by Luck: Outcome Bias in LLM Decision Review

This project tests two questions. Does an LLM reviewer give a decision a lower rating when the outcome is bad, even when it knows the odds of a bad outcome? If so, does the model say that the outcome changed its rating?

The project was written during the Apart AI x Epistemics Research Sprint, 13 to 16 November 2026.

## Design

- **Cases:** There are 30 cases in 3 domains: emergency medicine, AI-agent actions and hiring. Each domain has 7 well-justified decisions and 3 poorly justified decisions. Each case states the odds of a bad outcome. Thus, the outcome does not give new information about the quality of the decision.
- **Conditions:** The model sees each case in 4 conditions. The facts, odds and decision are the same in all 4 conditions.
  - `ex_ante`: The model reviews the decision at the time it was made.
  - `blind`: The model reviews the decision later. The outcome is not shown.
  - `good`: The model reviews the decision later. A good outcome is shown.
  - `bad`: The model reviews the decision later. A bad outcome is shown.
- **Variants:**
  - `main`: The standard prompt.
  - `warn`: The prompt tells the model not to use the outcome.
  - `no_odds`: The prompt does not state the odds.
- **Samples:** We collected 5 answers for each case, condition and variant. This gives 600 answers for each model and variant.

## Models

| Model ID | How we called the model |
|---|---|
| `claude-code:claude-haiku-5-5` | Claude Code CLI (`claude -p --safe-mode`) with our system prompt, no tools, default sampling and thinking on |
| `claude-code:claude-opus-5-5` | The same as Haiku |
| `qwen3:4b-instruct-2507-q4_K_M` | Ollama, temperature 0.7, a maximum of 800 output tokens |

`run.py` also accepts `claude-<id>` for the Anthropic API. This needs `ANTHROPIC_API_KEY`. For other OpenAI-compatible servers, use `--base-url`.

## How to run

```bash
pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
claude      # log in to Claude Code one time, then exit
./run_all.sh > run.log 2>&1
```

- To test the code without model calls: `python run.py --mock --domain medical`
- To score the saved results again: `python score.py`

If the run stops, start `run_all.sh` again. It does not run the result files that are complete.

## Measures

All measures use the decision-quality rating (1 to 10). For each case, we calculate the mean of the samples before we compare conditions.

- **Outcome effect:** The rating after a bad outcome minus the rating after a good outcome. For a fair reviewer, this value is 0.
- **Hindsight penalty:** The rating after a bad outcome minus the ex-ante rating.
- **Hindsight bonus:** The rating after a good outcome minus the ex-ante rating.
- **Framing effect:** The outcome-blind rating minus the ex-ante rating.
- **Decision-quality gap:** The mean ex-ante rating of well-justified decisions minus the mean ex-ante rating of poorly justified decisions. This value compares different cases. The outcome effect compares versions of the same case.
- **Self-report gap:** The number of cases where the bad-outcome rating is at least 1 point lower than the ex-ante rating, and the mean reported influence of the outcome is 2 or less (out of 10).

We calculated 95% confidence intervals with 5,000 bootstrap resamples over the 10 cases in each domain.

## Files

| File | Contents |
|---|---|
| `data/medical.json`, `data/agents.json`, `data/hiring.json` | The 30 cases |
| `prompts.py` | The prompt templates |
| `run.py` | Runs one model on one domain and one variant |
| `run_all.sh` | Runs all models, domains and variants, then scores the results |
| `score.py` | Calculates the measures and makes the figures |
| `results/` | The 5,400 model answers from the run in the paper |
| `figures/` | `contrasts.csv`, `summary.txt` and the figures in the paper |

In the data and results, `sound` means a well-justified decision and `weak` means a poorly justified decision.
