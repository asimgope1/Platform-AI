"""
Response Orchestrator — Executes Permitted Decision Engine Actions.

Separates policy decisions from LLM text generation:
  • Policy actions (CLARIFY, CONFIRM, OFFER_TRANSFER, FEEDBACK, TRANSFER)
    produce deterministic, localized spoken responses and update state flags.
  • CONTINUE routes into RAG + Gemma (with intelligence_context) and
    resilient IVR Mini-Brain fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from decision_engine.decision_manager import Action, DecisionResult
from telephony.intelligence.session_state import ConversationState


@dataclass
class PlannedResponse:
    text: str
    use_llm: bool
    action: Action
    trigger_transfer_event: Optional[Dict[str, Any]] = None
    intelligence_context: Optional[Dict[str, Any]] = None


class ResponseOrchestrator:
    """Translates a DecisionResult + ConversationState into an execution plan."""

    def prepare(
        self,
        decision: DecisionResult,
        state: ConversationState,
    ) -> PlannedResponse:
        action = decision.action
        intel_ctx = state.to_intelligence_context(recommended_action=action.value)

        if action == Action.TRANSFER:
            state.transfer_completed = True
            transfer_event = (decision.metadata or {}).get("transfer_payload")
            return PlannedResponse(
                text=decision.policy_prompt or "",
                use_llm=False,
                action=action,
                trigger_transfer_event=transfer_event,
                intelligence_context=intel_ctx,
            )

        if action == Action.OFFER_TRANSFER:
            state.transfer_offered = True
            state.awaiting_transfer_confirmation = True
            return PlannedResponse(
                text=decision.policy_prompt or "",
                use_llm=False,
                action=action,
                intelligence_context=intel_ctx,
            )

        if action == Action.CLARIFY:
            state.clarification_count += 1
            return PlannedResponse(
                text=decision.policy_prompt or "",
                use_llm=False,
                action=action,
                intelligence_context=intel_ctx,
            )

        if action == Action.CONFIRM:
            if decision.reason in ("confirm_issue_resolution", "confirm_potential_resolution"):
                state.awaiting_resolution_confirmation = True
            return PlannedResponse(
                text=decision.policy_prompt or "",
                use_llm=False,
                action=action,
                intelligence_context=intel_ctx,
            )

        if action == Action.FEEDBACK:
            completed = bool((decision.metadata or {}).get("completed", False))
            if not completed:
                state.feedback_requested = True
                state.awaiting_feedback = True
            return PlannedResponse(
                text=decision.policy_prompt or "",
                use_llm=False,
                action=action,
                intelligence_context=intel_ctx,
            )

        # Action.CONTINUE -> check if agent unavailable message overrides LLM or run conversational LLM
        if decision.reason == "transfer_agent_unavailable" and decision.policy_prompt:
            return PlannedResponse(
                text=decision.policy_prompt,
                use_llm=False,
                action=action,
                intelligence_context=intel_ctx,
            )

        return PlannedResponse(
            text="",
            use_llm=True,
            action=action,
            intelligence_context=intel_ctx,
        )
