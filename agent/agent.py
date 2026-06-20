from google.adk import Agent
from google.adk.models.anthropic_llm import AnthropicLlm
from google.adk.models.registry import LLMRegistry

# Use Anthropic's Claude via the *direct* Anthropic API (reads ANTHROPIC_API_KEY).
#
# ADK ships two Claude integrations: `AnthropicLlm` (direct api.anthropic.com)
# and `Claude` (served through Vertex AI / Google Cloud). By default the model
# registry maps bare "claude-*" strings to the Vertex-backed `Claude` class, so
# we register `AnthropicLlm` to claim those patterns. This keeps everything off
# Google Cloud and also routes the eval LLM-as-judge (which resolves its judge
# model from a plain string) to the direct Anthropic API.
LLMRegistry.register(AnthropicLlm)
LLMRegistry.resolve.cache_clear()  # drop any cached "claude-*" -> Claude mapping

# Default to the latest, most capable Claude model.
MODEL = "claude-opus-4-8"

root_agent = Agent(
    name="greeting_agent",
    model=AnthropicLlm(model=MODEL),
    instruction="You are a helpful assistant. Greet the user warmly.",
)