"""Example custom eval metrics for the greeting agent.

Register a function here in an eval config under `custom_metrics`, e.g.:

    "custom_metrics": {
      "response_is_concise": {
        "code_config": { "name": "agent.custom_metrics.response_is_concise" }
      }
    }

See EVALS.md §7 for the full contract.
"""

from __future__ import annotations

from typing import Optional

from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult

_MAX_CHARS = 200


def _final_text(invocation: Invocation) -> str:
  if invocation.final_response and invocation.final_response.parts:
    return "".join(p.text or "" for p in invocation.final_response.parts)
  return ""


def response_is_concise(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: Optional[list[Invocation]],
    conversation_scenario: Optional[ConversationScenario] = None,
) -> EvaluationResult:
  """Deterministic metric: passes when the final response is <= 200 chars."""
  per_invocation: list[PerInvocationResult] = []
  total = 0.0
  for actual in actual_invocations:
    score = 1.0 if len(_final_text(actual)) <= _MAX_CHARS else 0.0
    total += score
    per_invocation.append(
        PerInvocationResult(
            actual_invocation=actual,
            score=score,
            eval_status=EvalStatus.PASSED if score >= 1.0 else EvalStatus.FAILED,
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
