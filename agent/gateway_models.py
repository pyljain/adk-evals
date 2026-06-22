"""Route ADK model + eval-judge traffic through an enterprise AI gateway.

This module is **opt-in**: nothing here runs unless you import it (e.g. from
`agent/agent.py`). Importing it registers gateway-backed LLM classes so that
both the agent model and the eval LLM-as-judge — which ADK instantiates from a
bare model *string* via the registry — resolve to a client pointed at your
gateway, with custom auth headers if needed. No gcloud / Vertex involved.

Enable by adding `from . import gateway_models  # noqa: F401` to agent/agent.py
(or importing it before constructing the agent), and set the env vars below.

See GATEWAY.md for the full write-up and which approach to pick.
"""

from __future__ import annotations

import os
from functools import cached_property

from anthropic import AsyncAnthropic
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.registry import LLMRegistry

# --- gateway config (set these in the environment / agent/.env) ---------------
_GATEWAY_URL = os.environ.get("AI_GATEWAY_URL")          # e.g. https://gw.corp/anthropic
_GATEWAY_KEY = os.environ.get("AI_GATEWAY_KEY", "")      # token the gateway expects
# Optional extra headers as "K1=V1,K2=V2" (e.g. tenant / project routing)
_GATEWAY_HEADERS = dict(
    kv.split("=", 1)
    for kv in os.environ.get("AI_GATEWAY_HEADERS", "").split(",")
    if "=" in kv
)


class GatewayClaude(AnthropicLlm):
  """Claude over an Anthropic-compatible gateway, with custom base_url + headers.

  Claims the same "claude-*" patterns as AnthropicLlm, so registering it makes
  every bare "claude-*" string (agent model AND eval judge) use this client.
  """

  @classmethod
  def supported_models(cls) -> list[str]:
    return [r"claude-3-.*", r"claude-.*-4.*"]

  @cached_property
  def _anthropic_client(self) -> AsyncAnthropic:
    if not _GATEWAY_URL:
      raise ValueError("AI_GATEWAY_URL must be set to use GatewayClaude.")
    return AsyncAnthropic(
        base_url=_GATEWAY_URL,
        api_key=_GATEWAY_KEY,
        default_headers=_GATEWAY_HEADERS or None,
    )


class GatewayLiteLlm(LiteLlm):
  """Any OpenAI-compatible model over the gateway (gpt-*, llama, etc.).

  Claims "openai/*" so judge_model="openai/<gateway-model-name>" (and agent
  models) route through the gateway. LiteLLM uses the OpenAI request format
  against `api_base`.
  """

  def __init__(self, model: str, **kwargs):
    super().__init__(
        model=model,
        api_base=_GATEWAY_URL,
        api_key=_GATEWAY_KEY,
        extra_headers=_GATEWAY_HEADERS or None,
        **kwargs,
    )

  @classmethod
  def supported_models(cls) -> list[str]:
    return [r"openai/.*"]


# Register both, then drop any cached "claude-*"/"openai/*" -> default mappings.
LLMRegistry.register(GatewayClaude)
LLMRegistry.register(GatewayLiteLlm)
LLMRegistry.resolve.cache_clear()
