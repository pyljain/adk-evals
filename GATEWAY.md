# Running models (and eval judges) through the enterprise AI gateway

**Short answer: yes — there is no hard dependency on gcloud / Vertex login.** The clients
ADK instantiates for local evals are configurable to point at your gateway. This doc
explains exactly how the client is created, what knobs exist, and the three ways to wire it
up (env-only → custom-headers), with copy-paste code that's already in
[`agent/gateway_models.py`](./agent/gateway_models.py).

---

## How the client actually gets created (the mechanics)

Two things matter:

1. **The agent model.** In `agent/agent.py` we build `AnthropicLlm(model="claude-opus-4-8")`.
   Its HTTP client is `AsyncAnthropic()` — constructed **with no arguments**
   (`anthropic_llm.py`), so it picks up everything from the environment.

2. **The eval judge model.** Every local LLM-as-judge metric resolves its judge from a bare
   model **string** and instantiates it like this (`llm_as_judge.py`, `hallucinations_v1.py`):

   ```python
   llm_class = LLMRegistry.resolve(model_id)   # e.g. "claude-opus-4-8"
   return llm_class(model=model_id)            # ONLY model is passed
   ```

   **This is the crux:** the judge gives the class *only* the model name. There is no hook in
   the eval config to pass a `base_url`, `api_key`, or headers. So the gateway settings must
   come from **either (a) environment variables the SDK reads, or (b) a custom LLM class you
   register** that bakes the gateway config into its client.

What the underlying clients honor (verified against the installed SDKs):

| Client | base_url | api_key / token | custom headers |
|--------|----------|------------------|----------------|
| `AsyncAnthropic()` (native) | `ANTHROPIC_BASE_URL` env, or `base_url=` kwarg | `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` env, or `api_key=` kwarg | **kwarg only** (`default_headers=`) — not settable by env |
| `LiteLlm(model, **kwargs)` | `api_base=` kwarg (or provider env like `OPENAI_API_BASE`) | `api_key=` kwarg (or `OPENAI_API_KEY` etc.) | `extra_headers=` kwarg |

Because the judge only passes `model=`, **env vars alone get you base_url + key; custom
headers require a registered subclass.**

> None of this involves `gcloud` or Vertex. You only hit Vertex if you (1) use ADK's
> Vertex-backed `Claude` class, or (2) select a Vertex eval metric (`response_evaluation_score`,
> `safety_v1`, `multi_turn_*`). We avoid both — see [`EVALS.md`](./EVALS.md).

---

## Pick your approach

| Your gateway speaks… | Need custom headers? | Approach |
|----------------------|----------------------|----------|
| Anthropic Messages API (`/v1/messages`) | no | **A** — env vars only, zero code |
| OpenAI Chat Completions (`/v1/chat/completions`) | no | **B** — LiteLLM model string + env |
| Either, **and** you need custom routing headers / pinned-in-code config | yes | **C** — registered gateway class ([`gateway_models.py`](./agent/gateway_models.py)) |

Most enterprise gateways are **OpenAI-compatible** (Approach B) and most also let you front
Anthropic models that way — so B is the general answer for "use other models via our
gateway." Use C when the gateway needs headers (tenant/project IDs) the SDK env vars can't set.

---

## Approach A — Anthropic-compatible gateway, env only (no code)

Keep the agent and `judge_model: "claude-opus-4-8"` exactly as they are. Just set:

```bash
export ANTHROPIC_BASE_URL="https://gw.corp/anthropic"   # your gateway's Anthropic path
export ANTHROPIC_API_KEY="<gateway-token>"              # or ANTHROPIC_AUTH_TOKEN for Bearer
unset GOOGLE_CLOUD_PROJECT GOOGLE_CLOUD_LOCATION        # ensure nothing points at Vertex
```

Then run evals normally:

```bash
.venv/bin/adk eval agent agent/greeting.evalset.json \
  --config_file_path agent/local_eval_config.json --print_detailed_results
```

Both the agent's `AsyncAnthropic()` and the judge's `AsyncAnthropic()` read
`ANTHROPIC_BASE_URL` and route through the gateway. **Limitation:** you can't add arbitrary
headers (e.g. `x-tenant`) this way — if the gateway requires them, use Approach C.

---

## Approach B — OpenAI-compatible gateway via LiteLLM (works for *other* models too)

This is how you run gpt-*, Llama, Mistral, etc. through the gateway — or even Anthropic
models if the gateway exposes them OpenAI-style.

1. Point the judge at a LiteLLM model string. In `agent/local_eval_config.json`:

   ```json
   "final_response_match_v2": {
     "threshold": 0.7,
     "judge_model_options": { "judge_model": "openai/<gateway-model-name>", "num_samples": 3 }
   }
   ```

   Any `openai/*` string resolves to ADK's `LiteLlm` class (the model registry maps it there).

2. Set the OpenAI-compatible endpoint + key the SDK reads:

   ```bash
   export OPENAI_API_BASE="https://gw.corp/v1"     # gateway's OpenAI-compatible base
   export OPENAI_API_KEY="<gateway-token>"
   ```

3. For the **agent**, use the same string in `agent/agent.py`:

   ```python
   from google.adk.models.lite_llm import LiteLlm
   root_agent = Agent(name="greeting_agent", model=LiteLlm(model="openai/<gateway-model-name>"), ...)
   ```

No gcloud, no Vertex. Swap `<gateway-model-name>` to evaluate with any model the gateway
serves. **Limitation:** same as A — env can't add custom headers; use C for that.

---

## Approach C — registered gateway class (full control: base_url + key + custom headers)

Use this when the gateway needs custom headers, or you want the config pinned in code rather
than env. The module [`agent/gateway_models.py`](./agent/gateway_models.py) is ready to go —
it defines and **registers** two classes so the agent *and* the judge (resolved from a model
string) route through the gateway:

- `GatewayClaude` — claims `claude-*`; client is
  `AsyncAnthropic(base_url=..., api_key=..., default_headers=...)`.
- `GatewayLiteLlm` — claims `openai/*`; injects `api_base` / `api_key` / `extra_headers` into
  every LiteLLM call.

### Steps

1. **Set the gateway env vars** (in `agent/.env` or the shell):

   ```bash
   export AI_GATEWAY_URL="https://gw.corp/anthropic"        # path matching the client you use
   export AI_GATEWAY_KEY="<gateway-token>"
   export AI_GATEWAY_HEADERS="x-tenant=team-x,x-project=evals"   # optional, comma-separated K=V
   ```

   > If you use **both** `GatewayClaude` (Anthropic path) and `GatewayLiteLlm` (OpenAI path)
   > and they live at different URLs, set `AI_GATEWAY_URL` to the one you're actually
   > evaluating with, or fork the module to read two separate vars.

2. **Enable it** by importing the module before the agent is built. Add one line to the top
   of `agent/agent.py`:

   ```python
   from . import gateway_models  # noqa: F401  — registers gateway-backed model classes
   ```

   Importing it calls `LLMRegistry.register(...)` for both classes and clears the resolver
   cache, so any later `resolve("claude-opus-4-8")` / `resolve("openai/...")` returns the
   gateway class. (This *overrides* the direct-Anthropic registration from `agent.py` — which
   is exactly what you want when going through the gateway.)

3. **Keep your config model strings unchanged** — `judge_model: "claude-opus-4-8"` now
   resolves to `GatewayClaude`; `"openai/<name>"` to `GatewayLiteLlm`. Run evals as usual.

### What the module does (verified)

```text
claude-opus-4-8  -> GatewayClaude     base_url=https://gw.corp/anthropic/  headers={x-tenant, x-project}
openai/gpt-4o    -> GatewayLiteLlm    api_base=https://gw.corp/...         extra_headers={x-tenant, x-project}
```

Customize freely: add OAuth-token refresh, mTLS, a different header scheme, or split URLs per
provider — it's just a `cached_property` returning a client / kwargs you control.

---

## Verify the wiring (no network calls)

```bash
AI_GATEWAY_URL=https://gw.corp/anthropic AI_GATEWAY_KEY=tok \
AI_GATEWAY_HEADERS="x-tenant=team-x" \
.venv/bin/python - <<'PY'
from agent import gateway_models                     # registers gateway classes
from google.adk.models.registry import LLMRegistry
c = LLMRegistry.resolve("claude-opus-4-8"); inst = c(model="claude-opus-4-8")
print("claude judge ->", c.__name__, "| base_url:", str(inst._anthropic_client.base_url))
print("            headers:", dict(inst._anthropic_client.default_headers).get("x-tenant"))
o = LLMRegistry.resolve("openai/gpt-4o"); oi = o(model="openai/gpt-4o")
print("openai judge ->", o.__name__, "| api_base:", oi._additional_args.get("api_base"))
PY
```

Expected: `GatewayClaude` with your gateway `base_url` + header, and `GatewayLiteLlm` with
your `api_base`. If you instead see `AnthropicLlm` / `LiteLlm`, the module wasn't imported
before resolution (or the resolver cache wasn't cleared).

---

## Gotchas / checklist

- **The judge only passes `model=`** — so base_url + key work via env (A/B), but **custom
  headers require Approach C**. There is no eval-config field for endpoint/credentials.
- **Env precedence (Anthropic SDK):** a set `ANTHROPIC_API_KEY` *or* `ANTHROPIC_AUTH_TOKEN`
  takes precedence over profile-based auto-discovery. Set exactly the one your gateway wants;
  don't set both (the API rejects requests carrying both).
- **Import order matters for Approach C:** import `gateway_models` *before* anything resolves
  a model. Importing it from the top of `agent/agent.py` guarantees that, and it calls
  `LLMRegistry.resolve.cache_clear()` to drop stale mappings.
- **Still no Vertex:** none of these paths touch `gcloud`. Keep
  `GOOGLE_CLOUD_PROJECT`/`GOOGLE_CLOUD_LOCATION` unset so nothing can fall back to the
  Vertex-backed `Claude` class, and keep Vertex metrics out of the eval config.
- **`api_base` path must match the client:** `GatewayClaude` needs the gateway's
  Anthropic-format path; `GatewayLiteLlm`/Approach B needs the OpenAI-compatible path
  (`.../v1`). Point the URL at the right one.
