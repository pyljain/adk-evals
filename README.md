# Greeting Agent (ADK + Claude)

A minimal [Google ADK](https://google.github.io/adk-docs/) agent that runs on **Claude via
the direct Anthropic API** (`api.anthropic.com`) — and a **local-only eval setup** that uses
Claude as the LLM-as-judge. **No Google Cloud / Vertex AI is involved** in either the agent
or the evals.

> **Why no Vertex evals?** ADK's Vertex-backed eval metrics call the Vertex AI Gen AI
> Evaluation endpoint (`aiplatform.googleapis.com`). Reaching that endpoint through our
> company AI gateway is tricky/unreliable, so this project deliberately uses **only local
> metrics** — deterministic checks plus a Claude LLM-as-judge that goes straight to the
> Anthropic API. See [`EVALS.md`](./EVALS.md) for the full breakdown of which metrics are
> local vs. Vertex-backed.

---

## What's here

```
.
├── agent/
│   ├── __init__.py              # exposes root_agent for the adk CLI
│   ├── agent.py                 # the agent — Claude (claude-opus-4-8), direct Anthropic
│   ├── custom_metrics.py        # example custom eval metric (response_is_concise)
│   ├── local_eval_config.json   # LOCAL eval metrics (no Vertex)
│   └── greeting.evalset.json    # example eval cases
├── EVALS.md                     # deep dive: how ADK evals work, every metric, custom evals
└── README.md                    # this file
```

The agent itself (`agent/agent.py`) is a single LLM agent:

```python
root_agent = Agent(
    name="greeting_agent",
    model=AnthropicLlm(model="claude-opus-4-8"),   # direct Anthropic, not Vertex
    instruction="You are a helpful assistant. Greet the user warmly.",
)
```

It registers `AnthropicLlm` in ADK's model registry so that any bare `"claude-*"` model
string — including the **eval judge model** — resolves to the **direct** Anthropic API
instead of ADK's default Vertex-backed `Claude` class. This is the single most important
line for keeping everything off Google Cloud.

---

## Prerequisites

- Python 3.10+ with the project virtualenv at `.venv` (already set up here).
- An **Anthropic API key**.

Dependencies are already installed in `.venv`:

| Package | Purpose |
|---------|---------|
| `anthropic` | ADK's direct Claude integration (`AnthropicLlm`) |
| `google-adk[eval]` | ADK + the eval framework and its deps |
| `google-cloud-aiplatform` | **Import-only.** ADK's eval registry imports the Vertex facade at load time, so the package must be *importable* — but it is **never called** for local metrics. |

If you ever recreate the environment:

```bash
.venv/bin/python -m pip install "anthropic>=0.43.0" "google-adk[eval]" google-cloud-aiplatform
```

---

## Configure your API key

The agent and the eval judge both read `ANTHROPIC_API_KEY`. Either export it:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

…or put it in `agent/.env` (ADK auto-loads it; keep it out of git):

```
ANTHROPIC_API_KEY=sk-ant-...
```

> Using the **company AI gateway** instead of api.anthropic.com directly? See the next
> section — there's a built-in toggle, and [`GATEWAY.md`](./GATEWAY.md) covers every option.

### Using the enterprise AI gateway

To route the agent **and** the eval judge through the gateway (no gcloud, still no Vertex),
flip the `AI_GATEWAY` toggle and provide the gateway config:

```bash
export AI_GATEWAY=1
export AI_GATEWAY_URL="https://gw.corp/anthropic"          # gateway's Anthropic-format path
export AI_GATEWAY_KEY="<gateway-token>"
export AI_GATEWAY_HEADERS="x-tenant=team-x,x-project=evals" # optional custom headers
```

With `AI_GATEWAY=1`, `agent/agent.py` loads [`agent/gateway_models.py`](./agent/gateway_models.py),
which registers gateway-backed clients so bare `claude-*` (and `openai/*`) model strings —
including the eval judge — resolve to a client pointed at your gateway, with custom headers
if needed. Unset `AI_GATEWAY` to go back to the direct Anthropic API.

For OpenAI-compatible gateways and other models (gpt-*, Llama, …), and the full rationale,
see **[`GATEWAY.md`](./GATEWAY.md)**.

---

## Run the agent

```bash
# One-shot
.venv/bin/adk run agent "Hi there!"

# Interactive REPL (no query)
.venv/bin/adk run agent

# Browser dev UI
.venv/bin/adk web

# REST/SSE API server
.venv/bin/adk api_server
```

---

## Local evals — focus & steps

The eval config (`agent/local_eval_config.json`) is intentionally restricted to metrics that
run **in-process**. Three kinds, all local:

| Metric | Type | Needs a model? | Notes |
|--------|------|----------------|-------|
| `tool_trajectory_avg_score` | deterministic | no | exact match of tool calls vs. expected |
| `final_response_match_v2` | LLM-as-judge (Claude) | yes — direct Anthropic | semantic match to the golden `final_response` |
| `rubric_based_final_response_quality_v1` | LLM-as-judge (Claude) | yes — direct Anthropic | grades the response against your rubric checklist |
| `response_is_concise` | custom (deterministic) | no | example custom metric in `agent/custom_metrics.py` |

**None of these touch Vertex.** (For the full list of which built-in metrics *are*
Vertex-backed and therefore excluded here — `response_evaluation_score`, `safety_v1`,
`multi_turn_*` — see [`EVALS.md` §1 and §4](./EVALS.md).)

### Step 1 — make sure your key is set

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2 — run the eval

```bash
.venv/bin/adk eval \
  agent \
  agent/greeting.evalset.json \
  --config_file_path agent/local_eval_config.json \
  --print_detailed_results
```

- `agent` — the agent package (the folder with `__init__.py`).
- `agent/greeting.evalset.json` — the eval set (recorded user inputs + expected responses).
- `--config_file_path` — **the metrics to run**. Omit it and ADK falls back to its default
  metrics, so always pass this to stay on the local-only set.
- `--print_detailed_results` — per-case scores in the console.

Run a subset of cases by id:

```bash
.venv/bin/adk eval agent "agent/greeting.evalset.json:warm_greeting" \
  --config_file_path agent/local_eval_config.json
```

### Step 3 — read the output

Each case reports per-metric `PASSED` / `FAILED` against the thresholds in the config
(e.g. `final_response_match_v2` passes at ≥ 0.7). The LLM-as-judge metrics sample the judge
`num_samples` times (3 here) and aggregate, so scores are stable but not identical run to run.

### Step 4 — verify the wiring (optional, no network calls)

Confirms the agent and every judge resolve to the **direct Anthropic** class, never Vertex:

```bash
ANTHROPIC_API_KEY=dummy .venv/bin/python - <<'PY'
import agent.agent
from google.adk.evaluation.eval_config import get_evaluation_criteria_or_default, get_eval_metrics_from_config
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY as R
from google.adk.evaluation.custom_metric_evaluator import _CustomMetricEvaluator
from google.adk.cli.cli_eval import get_default_metric_info

cfg = get_evaluation_criteria_or_default("agent/local_eval_config.json")
for name, c in (cfg.custom_metrics or {}).items():      # mirror what `adk eval` does
    R.register_evaluator(get_default_metric_info(metric_name=name, description=c.description), _CustomMetricEvaluator)

print("agent model:", type(agent.agent.root_agent.canonical_model).__name__)
for m in get_eval_metrics_from_config(cfg):
    ev = R.get_evaluator(m)
    judge = f" -> {type(ev._judge_model).__name__}" if hasattr(ev, "_judge_model") else ""
    print(f"  {m.metric_name}: {type(ev).__name__}{judge}")
PY
```

Expected: agent model `AnthropicLlm`; both LLM-as-judge metrics resolve via `AnthropicLlm`.

---

## Add more eval cases

Append objects to the `eval_cases` array in `agent/greeting.evalset.json`. Each case is a
conversation of one or more invocations:

```json
{
  "eval_id": "polite_decline",
  "conversation": [
    {
      "invocation_id": "inv-1",
      "user_content": { "role": "user", "parts": [{ "text": "Can you book me a flight?" }] },
      "final_response": {
        "role": "model",
        "parts": [{ "text": "I'm just here to say hello — I can't book flights, but I'm happy to chat!" }]
      }
    }
  ]
}
```

The `final_response` is the **golden answer** the judge compares against. Deterministic and
reference-free metrics (`tool_trajectory_avg_score`, `rubric_based_*`, `hallucinations_v1`,
custom checks) don't strictly need it.

---

## Tune or extend the metrics

- **Thresholds / rubrics:** edit `agent/local_eval_config.json` — e.g. change the
  `rubric_based_final_response_quality_v1` rubrics to whatever behavior you want graded.
- **New custom metric:** add a function to `agent/custom_metrics.py` and register it under
  `custom_metrics` in the config (see [`EVALS.md` §7](./EVALS.md)).
- **More LLM-as-judge metrics** (`hallucinations_v1`, `rubric_based_tool_use_quality_v1`,
  `rubric_based_multi_turn_trajectory_quality_v1`): all local, all Claude-direct — config
  shapes are in [`EVALS.md` §6](./EVALS.md).

---

## The one rule to keep evals off Vertex

Stay on the metrics in [`EVALS.md` §6](./EVALS.md) (local LLM-as-judge) plus the local
deterministic ones (`tool_trajectory_avg_score`, `response_match_score`, custom metrics).
Adding any of `response_evaluation_score`, `safety_v1`, or `multi_turn_*` to the config will
make ADK call the **Vertex** eval endpoint and require `GOOGLE_CLOUD_PROJECT` /
`GOOGLE_CLOUD_LOCATION` credentials — exactly what we're avoiding.
