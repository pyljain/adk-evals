import os

from google.adk import Agent
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.registry import LLMRegistry

# Default to the latest, most capable Claude model.
MODEL = "claude-opus-4-8"


def _truthy(value: str) -> bool:
  return value.strip().lower() in ("1", "true", "yes", "on")


if _truthy(os.environ.get("AI_GATEWAY", "")):
  # Route the agent + eval judge through the enterprise AI gateway.
  #
  # Importing `gateway_models` registers gateway-backed classes for "claude-*"
  # and "openai/*", so the eval LLM-as-judge (resolved from a model string) also
  # goes through the gateway. We build the agent on the gateway-backed Claude
  # client (custom base_url / headers). Configure with AI_GATEWAY_URL /
  # AI_GATEWAY_KEY / AI_GATEWAY_HEADERS — see GATEWAY.md.
  from .gateway_models import GatewayClaude

  model = GatewayClaude(model=MODEL)
else:
  # Use Claude via the *direct* Anthropic API (reads ANTHROPIC_API_KEY).
  #
  # ADK ships two Claude integrations: `AnthropicLlm` (direct api.anthropic.com)
  # and `Claude` (served through Vertex AI / Google Cloud). By default the model
  # registry maps bare "claude-*" strings to the Vertex-backed `Claude` class, so
  # we register `AnthropicLlm` to claim those patterns. This keeps everything off
  # Google Cloud and also routes the eval LLM-as-judge (which resolves its judge
  # model from a plain string) to the direct Anthropic API.
  LLMRegistry.register(AnthropicLlm)
  LLMRegistry.resolve.cache_clear()  # drop any cached "claude-*" -> Claude mapping
  model = AnthropicLlm(model=MODEL)

root_agent = Agent(
    name="greeting_agent",
    model=model,
    instruction="You are a helpful assistant. Greet the user warmly.",
)