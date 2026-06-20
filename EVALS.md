# Local Evals with Anthropic (Claude) — Setup & Notes

This repo is configured so that **both the agent and the eval LLM-as-judge run on the
direct Anthropic API** (`api.anthropic.com`). No request goes to Google Cloud / Vertex AI.

---

## 1. How ADK evals work (the short version)

ADK's eval system lives in `google/adk/evaluation/`. A run scores an **eval set**
(`*.evalset.json` — recorded user inputs + expected agent responses/tool calls) against a
set of **metrics** defined in an **eval config** (`*.json`). Each metric maps to an
*evaluator*. Evaluators fall into two camps:

| Camp | Examples | Where scoring happens |
|------|----------|------------------------|
| **Local — deterministic** | `tool_trajectory_avg_score`, `response_match_score` (ROUGE‑1) | In-process, pure Python — no model call |
| **Local — LLM-as-judge** | `final_response_match_v2`, `hallucinations_v1`, `rubric_based_final_response_quality_v1`, `rubric_based_tool_use_quality_v1`, `rubric_based_multi_turn_trajectory_quality_v1` | In-process; calls a **judge model** you choose (here: Claude, direct) |
| **Vertex-backed** | `response_evaluation_score`, `safety_v1`, `multi_turn_task_success_v1`, `multi_turn_trajectory_quality_v1`, `multi_turn_tool_use_quality_v1` | **Google Cloud** — Vertex AI Gen AI Eval service |

> ⚠️ **Two easy-to-get-wrong classifications:** despite the names, the `multi_turn_*` metrics are **Vertex-backed** (they delegate to the Vertex Gen AI Eval SDK — see `multi_turn_*_evaluator.py`), while `response_match_score` is **local** (plain ROUGE‑1 string overlap, `RougeEvaluator`, no model). Section 6 lists every local LLM-as-judge metric in detail.

The Vertex-backed metrics delegate to `vertex_ai_eval_facade.py`, which calls
`vertexai.Client(...).evals.evaluate(...)` — i.e. the **Vertex AI Gen AI Evaluation
service** (`aiplatform.googleapis.com`, under your `GOOGLE_CLOUD_PROJECT` /
`GOOGLE_CLOUD_LOCATION`). It ships your prompts, golden answers, agent outputs, and (for
multi-turn) agent instructions + tool declarations to Vertex. **We do not use those
metrics here.**

The LLM-as-judge evaluators instead resolve their judge model through ADK's model
registry (`llm_as_judge.py` → `LLMRegistry.resolve(judge_model)`), so pointing the judge
at a Claude model keeps everything on the Anthropic API.

---

## 2. What changed in this repo

### a) Agent now uses Claude via the direct Anthropic API — `agent/agent.py`

```python
from google.adk import Agent
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.registry import LLMRegistry

LLMRegistry.register(AnthropicLlm)        # claim "claude-*" for the direct API
LLMRegistry.resolve.cache_clear()

MODEL = "claude-opus-4-8"

root_agent = Agent(
    name="greeting_agent",
    model=AnthropicLlm(model=MODEL),       # direct Anthropic, not Vertex
    instruction="You are a helpful assistant. Greet the user warmly.",
)
```

**Why the `LLMRegistry.register(...)` line matters (important gotcha):** ADK ships *two*
Claude integrations — `AnthropicLlm` (direct `api.anthropic.com`, uses `ANTHROPIC_API_KEY`)
and `Claude` (served through **Vertex AI**, uses `GOOGLE_CLOUD_PROJECT`/`LOCATION`). By
default the registry maps bare `"claude-*"` model strings to the **Vertex** `Claude` class.
Registering `AnthropicLlm` re-points those patterns at the direct API. This is what makes
**the eval judge** (which only ever sees a model *string*, `"claude-opus-4-8"`) resolve to
the direct Anthropic API too. The agent itself passes an explicit `AnthropicLlm(...)`
instance, so it's direct regardless.

### b) Local eval config — `agent/local_eval_config.json`

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 1.0,
    "final_response_match_v2": {
      "threshold": 0.7,
      "judge_model_options": { "judge_model": "claude-opus-4-8", "num_samples": 3 }
    }
  }
}
```

- `tool_trajectory_avg_score` — deterministic, no model call.
- `final_response_match_v2` — LLM-as-judge; `judge_model` is Claude → **direct Anthropic**.
  `num_samples: 3` means the judge is sampled 3× per turn and aggregated (lower than the
  default 5 to cut cost; raise for more stability).
- **No `response_evaluation_score`, `response_match_score`, or `safety_v1`** → nothing hits
  Vertex.

### c) Example eval set — `agent/greeting.evalset.json`

Two greeting cases (input + expected final response) so you can run the eval immediately.

### d) Dependencies installed into `.venv`

| Package | Why |
|---------|-----|
| `anthropic` | ADK's `AnthropicLlm` (direct Claude API) |
| `google-adk[eval]` | The eval framework + its deps (`rouge-score`, `pandas`, `litellm`, `nltk`, …) |
| `google-cloud-aiplatform` (`vertexai`) | **Import-time only.** ADK's metric registry eagerly imports the Vertex facade at module load, so the package must be *importable* even for local-only evals. It is **never called** unless you select a Vertex-backed metric. |

---

## 3. Run the local eval

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # required for both the agent and the judge

# From the repo root:
.venv/bin/adk eval \
  agent \
  agent/greeting.evalset.json \
  --config_file_path agent/local_eval_config.json \
  --print_detailed_results
```

(Equivalently, in Python, via `AgentEvaluator.evaluate(...)` pointing at the same config
and eval set.) All model traffic goes to `api.anthropic.com`.

> Tip: put `ANTHROPIC_API_KEY=...` in `agent/.env` — ADK auto-loads it.

---

## 4. If you DO want to use Vertex + the additional validators

The Vertex-backed metrics give you Google's autorater scores:

| Metric name | What it measures | Turn scope |
|-------------|------------------|------------|
| `response_evaluation_score` | Response coherence/quality (`COHERENCE`, scored 1–5) | single-turn |
| `safety_v1` | Safety / harmlessness (`SAFETY`) | single-turn |
| `multi_turn_task_success_v1` | Did the agent achieve the conversation's goal(s)? | multi-turn |
| `multi_turn_trajectory_quality_v1` | Quality of the *path* taken (reference-free) | multi-turn |
| `multi_turn_tool_use_quality_v1` | Quality of the tool/function calls (reference-free) | multi-turn |

> `response_match_score` is **not** in this list — it's a *local* ROUGE‑1 metric (needs expected responses, no GCP). Use it as a cheap, deterministic alternative to the LLM-as-judge `final_response_match_v2` (see §6).

To enable the Vertex metrics, you'd make these changes:

### Code / config changes

1. **Add them to the eval config** (`agent/local_eval_config.json` or a separate file):

   ```json
   {
     "criteria": {
       "tool_trajectory_avg_score": 1.0,
       "final_response_match_v2": {
         "threshold": 0.7,
         "judge_model_options": { "judge_model": "claude-opus-4-8", "num_samples": 3 }
       },
       "response_evaluation_score": 0.7,
       "safety_v1": 0.8,
       "multi_turn_task_success_v1": 0.7
     }
   }
   ```

   These take a **plain float threshold** — they have no `judge_model`, because the
   autorater runs **on Vertex**, not via a model you pick.

2. **Provide Google Cloud credentials** via environment (read in
   `vertex_ai_eval_facade.py`):

   ```bash
   export GOOGLE_CLOUD_PROJECT=your-gcp-project
   export GOOGLE_CLOUD_LOCATION=us-central1
   # plus Application Default Credentials:
   gcloud auth application-default login
   # (or set GOOGLE_API_KEY=... for the API-key path)
   ```

   Without these the Vertex metrics raise `ValueError("...specify both project id and
   location...")`.

3. **Dependency** — `google-cloud-aiplatform` (the `vertexai` SDK) is already installed
   here for the import; with the env vars set it becomes live.

### What gets sent to Google Cloud

When a Vertex metric runs, the facade builds a dataset and calls
`vertexai.Client(...).evals.evaluate(dataset=..., metrics=[...])` against the Vertex AI Gen
AI Eval service:

- **Single-turn** (`response_*`, `safety_v1`): one row per turn with
  `{prompt, reference, response}` — i.e. your **user input, golden/expected answer, and the
  agent's actual output**.
- **Multi-turn** metrics additionally send **agent config** — each (sub-)agent's
  `agent_id`, **full instructions/system prompt**, and **tool declarations** — plus every
  conversation turn's events.

So the trade-off is explicit: the LLM-as-judge metrics keep evaluation on Anthropic and
respect the "prefer Anthropic models" setup; the Vertex metrics always use Google's
autorater on Google Cloud and ship prompts, instructions, and tool definitions there.

### The Anthropic-only stance

Because the Vertex metrics use Google's autorater (you can't substitute a Claude judge for
them), keeping evals fully on Anthropic means **staying on the local LLM-as-judge metrics**
(`final_response_match_v2`, `hallucinations_v1`, `rubric_based_*` — detailed in §6) and the
local deterministic metrics (`tool_trajectory_avg_score`, `response_match_score`). Note this
means the `multi_turn_*` metrics are off the table for a no-Vertex setup; for multi-turn
quality on Anthropic, use `rubric_based_multi_turn_trajectory_quality_v1` instead (it's
LLM-as-judge with a Claude judge). That's the configuration shipped here.

---

## 5. Verify the wiring (no network calls)

```bash
ANTHROPIC_API_KEY=dummy .venv/bin/python - <<'PY'
import agent.agent
from google.adk.evaluation.eval_config import (
    get_evaluation_criteria_or_default, get_eval_metrics_from_config)
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY as R

print("agent model:", type(agent.agent.root_agent.canonical_model).__name__)
cfg = get_evaluation_criteria_or_default("agent/local_eval_config.json")
for m in get_eval_metrics_from_config(cfg):
    ev = R.get_evaluator(m)
    line = f"  {m.metric_name}: {type(ev).__name__}"
    if hasattr(ev, "_judge_model"):
        line += f" -> judge {ev._judge_model_options.judge_model} via {type(ev._judge_model).__name__}"
    print(line)
PY
```

Expected: agent model `AnthropicLlm`; `final_response_match_v2` judged by
`claude-opus-4-8` via `AnthropicLlm` (direct Anthropic, no Vertex).

---

## 6. The local LLM-as-judge metrics in detail

These all run **in-process** and call the `judge_model` you specify (here Claude, direct).
Common mechanics for every one of them (`llm_as_judge.py`):

- The judge model is resolved from the `judge_model` **string** via the ADK model registry
  — which is why the `LLMRegistry.register(AnthropicLlm)` line in `agent/agent.py` makes
  them all go direct-Anthropic instead of Vertex.
- The judge is sampled **`num_samples`** times per invocation (default 5; we use 3) and the
  samples are aggregated into one score, to smooth out judge variance.
- Each score is compared to the metric's `threshold` to produce `PASSED` / `FAILED`.
- Common `judge_model_options`: `judge_model` (string), `num_samples` (int),
  `judge_model_config` (optional `GenerateContentConfig`, e.g. temperature/max tokens).

What each metric needs in the **eval case** and what it scores:

### `final_response_match_v2`  ·  criterion: `LlmAsAJudgeCriterion`
- **Scores:** whether the agent's final response is semantically equivalent to your
  `final_response` (golden answer). An LLM judge replaces brittle string matching — "Paris"
  vs "The capital is Paris." both pass, where ROUGE would not.
- **Needs:** each invocation must have a golden `final_response`. Score range `[0,1]`.
- **Optional:** `include_intermediate_responses_in_final: true` also feeds text the agent
  emitted *before* tool calls to the judge (useful for agents that narrate then act).

```json
"final_response_match_v2": {
  "threshold": 0.7,
  "judge_model_options": { "judge_model": "claude-opus-4-8", "num_samples": 3 }
}
```

### `hallucinations_v1`  ·  criterion: `HallucinationsCriterion`
- **Scores:** whether the response contains false / contradictory / **unsupported** claims.
  Two-stage judge: (1) a *segmenter* splits the response into sentences; (2) a *validator*
  grades each sentence against the available context (user query, tool results, developer
  instructions). The metric is the fraction of sentences that are `supported` or
  `not_applicable` — an "accuracy score". This is **reference-free**: it grounds claims in
  context, so no golden answer is required.
- **Optional:** `evaluate_intermediate_nl_responses: true` also checks NL text emitted
  between tool calls. Score range `[0,1]`.

```json
"hallucinations_v1": {
  "threshold": 0.8,
  "judge_model_options": { "judge_model": "claude-opus-4-8", "num_samples": 3 }
}
```

### `rubric_based_*`  ·  criterion: `RubricsBasedCriterion` (requires `rubrics`)
The rubric family lets you grade against **your own checklist**. The judge scores each
rubric (≈ pass/fail per criterion) and the metric aggregates them. All three share the same
config shape — the only difference is *what* gets sent to the judge:

| Metric | Judges… | Reference-free? |
|--------|---------|------------------|
| `rubric_based_final_response_quality_v1` | the final response against your rubrics | yes |
| `rubric_based_tool_use_quality_v1` | the agent's tool/function calls against your rubrics | yes |
| `rubric_based_multi_turn_trajectory_quality_v1` | the whole multi-turn trajectory against your rubrics | yes |

Each rubric is `{ rubric_id, rubric_content: { text_property: "<testable statement>" } }`.
Write `text_property` as a statement that's true when the agent did well:

```json
"rubric_based_final_response_quality_v1": {
  "threshold": 0.7,
  "judge_model_options": { "judge_model": "claude-opus-4-8", "num_samples": 3 },
  "rubrics": [
    { "rubric_id": "greets_warmly",
      "rubric_content": { "text_property": "The response greets the user in a warm, friendly tone." } },
    { "rubric_id": "no_overpromising",
      "rubric_content": { "text_property": "The response does not promise capabilities the agent does not have." } }
  ]
}
```

> Rubrics can also be attached per-eval-case (the `rubrics` field on an `EvalCase`/`Invocation`)
> when different cases need different checklists; the config-level `rubrics` above apply to
> the whole run.

---

## 7. Writing custom evals

When the built-in metrics don't capture what you care about, write a **custom metric** — a
plain Python function ADK calls for each eval case. You can make it deterministic (regex,
JSON-schema check, latency budget, exact business rule) or call Claude yourself for a
bespoke LLM judge.

### The contract

A custom metric is any importable function with this signature (sync or `async`):

```python
# agent/custom_metrics.py
from typing import Optional
from google.adk.evaluation.eval_case import Invocation, ConversationScenario
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import (
    EvaluationResult, PerInvocationResult, EvalStatus,
)


def response_is_concise(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
    """Example: passes when the agent's final response is <= 200 characters."""
    per_invocation = []
    total = 0.0
    for actual in actual_invocations:
        text = ""
        if actual.final_response and actual.final_response.parts:
            text = "".join(p.text or "" for p in actual.final_response.parts)
        score = 1.0 if len(text) <= 200 else 0.0
        total += score
        per_invocation.append(
            PerInvocationResult(
                actual_invocation=actual,
                score=score,
                eval_status=(
                    EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED
                ),
            )
        )
    overall = total / len(per_invocation) if per_invocation else None
    return EvaluationResult(
        overall_score=overall,
        overall_eval_status=(
            EvalStatus.PASSED
            if overall is not None and overall >= 1.0
            else EvalStatus.FAILED
        ),
        per_invocation_results=per_invocation,
    )
```

Key objects:
- **`actual_invocations`** — what the agent produced. Each `Invocation` has `user_content`,
  `final_response` (a `genai.types.Content` with `.parts`), and `intermediate_data`
  (tool calls/responses + intermediate events).
- **`expected_invocations`** — the golden cases from your eval set (or `None` if the case
  has no expected output). Same length/order as `actual_invocations` when present.
- **Return** an `EvaluationResult` with an `overall_score`, an `overall_eval_status`, and a
  `PerInvocationResult` per turn. (Note: ADK sets `eval_metric.threshold = None` before
  calling you, so do your own threshold logic — read it from the config another way if you
  need it, or hard-code the pass bar like above.)

### Register it in the eval config

Add the metric name to `criteria` **and** point `custom_metrics` at the function path:

```json
{
  "criteria": {
    "response_is_concise": 1.0,
    "final_response_match_v2": {
      "threshold": 0.7,
      "judge_model_options": { "judge_model": "claude-opus-4-8", "num_samples": 3 }
    }
  },
  "custom_metrics": {
    "response_is_concise": {
      "code_config": { "name": "agent.custom_metrics.response_is_concise" },
      "description": "Final response is at most 200 characters."
    }
  }
}
```

`code_config.name` is the **import path** to your function (`module.function`). It must be
importable from where you run `adk eval` — keeping it inside the `agent` package (as above)
is the simplest option.

### Custom LLM-judge metric (still Claude, still direct)

To grade with Claude inside a custom metric, reuse ADK's judge plumbing so it routes through
the same direct-Anthropic registration — resolve the model from the registry and call it:

```python
from google.adk.models.registry import LLMRegistry
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types

async def my_llm_metric(eval_metric, actual_invocations, expected_invocations, scenario=None):
    judge = LLMRegistry.resolve("claude-opus-4-8")(model="claude-opus-4-8")  # -> AnthropicLlm
    # build a prompt from actual/expected, then:
    req = LlmRequest(
        model="claude-opus-4-8",
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])],
        config=genai_types.GenerateContentConfig(),
    )
    async for resp in judge.generate_content_async(req):
        ...  # parse a score out of resp, build EvaluationResult
```

This reuses the `LLMRegistry.register(AnthropicLlm)` already done in `agent/agent.py`, so the
custom judge also hits `api.anthropic.com` and never Vertex. (For most bespoke judges the
built-in `rubric_based_*` metrics are easier — reach for a hand-rolled judge only when you
need scoring logic the rubric framework can't express.)

### Run custom + built-in together

Custom and built-in metrics live in the same config and run in one pass — no extra flags:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/adk eval agent agent/greeting.evalset.json \
  --config_file_path agent/local_eval_config.json --print_detailed_results
```
