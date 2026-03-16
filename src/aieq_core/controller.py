from __future__ import annotations

from pathlib import Path
from typing import Any

from .ledger import EpistemicLedger
from .modes import default_mode_registry
from .models import (
    ActionExecutor,
    ActionProposal,
    ActionType,
    ArtifactKind,
    Claim,
    ClaimStatus,
    ControllerDecision,
    ExecutionStatus,
    action_matches,
    clamp,
)
from .policy import ExpectedInformationGainPolicy


class ResearchController:
    """Chooses the next research move from the current ledger state."""

    def __init__(
        self,
        policy: ExpectedInformationGainPolicy | None = None,
        *,
        mode_registry: Any | None = None,
        default_mode: str = "ml_research",
    ) -> None:
        self.policy = policy or ExpectedInformationGainPolicy()
        self.mode_registry = mode_registry or default_mode_registry()
        self.default_mode = default_mode

    def decide(
        self,
        ledger: EpistemicLedger,
        *,
        backlog_limit: int = 5,
        mode_hint: str = "",
    ) -> ControllerDecision:
        claims = [claim for claim in ledger.list_claims() if claim.status != ClaimStatus.ARCHIVED]
        if not claims:
            adapter = self.mode_registry.get(mode_hint or self.default_mode)
            primary = adapter.bootstrap_proposal(ledger=ledger)
            if primary is None:
                primary = ActionProposal(
                    claim_id="",
                    claim_title="ledger bootstrap",
                    action_type=ActionType.PROPOSE_HYPOTHESIS,
                    expected_information_gain=1.0,
                    priority="now",
                    reason="The ledger is empty and needs its first target or claim.",
                    executor=ActionExecutor.MANUAL,
                    mode=mode_hint or self.default_mode,
                    stage="bootstrap",
                    command_hint=(
                        "Register a target or import a project before asking the controller to run."
                    ),
                )
            return ControllerDecision(
                queue_state="bootstrap",
                summary="No claims exist yet, so the controller should bootstrap the graph.",
                primary_action=primary,
                backlog=[],
            )

        proposals: list[ActionProposal] = []
        for claim in claims:
            adapter = self.mode_registry.for_claim(claim)
            proposals.extend(adapter.build_proposals(ledger=ledger, claim=claim))
        proposals = [self._apply_history_feedback(ledger, item) for item in proposals]
        ranked = sorted(
            proposals,
            key=lambda item: item.expected_information_gain,
            reverse=True,
        )

        if not ranked:
            primary = ActionProposal(
                claim_id=claims[0].id,
                claim_title=claims[0].title,
                action_type=ActionType.ANALYZE_FAILURE,
                expected_information_gain=0.5,
                priority="next",
                reason="No automated proposal is currently available for the active claims.",
                executor=ActionExecutor.MANUAL,
                mode=self.mode_registry.for_claim(claims[0]).name,
                stage="analysis",
                command_hint="Inspect the active claim and register the missing target, eval, or project context.",
            )
            ranked = [primary]

        primary = ranked[0]
        backlog = ranked[1:backlog_limit]
        summary = self._build_summary(ledger, primary)
        return ControllerDecision(
            queue_state=primary.stage or "exploration",
            summary=summary,
            primary_action=primary,
            backlog=backlog,
        )

    def _apply_history_feedback(
        self, ledger: EpistemicLedger, proposal: ActionProposal
    ) -> ActionProposal:
        if not proposal.claim_id:
            return proposal

        decisions = [
            item
            for item in ledger.decisions_for_claim(proposal.claim_id)
            if item.action_type == proposal.action_type
        ]
        executions = [
            item
            for item in ledger.executions_for_claim(proposal.claim_id)
            if item.action_type == proposal.action_type
        ]

        executed_decision_ids = {item.decision_id for item in executions if item.decision_id}
        pending_decisions = [item for item in decisions if item.id not in executed_decision_ids]
        running = [item for item in executions if item.status == ExecutionStatus.RUNNING]
        failed = [item for item in executions if item.status == ExecutionStatus.FAILED]
        succeeded = [item for item in executions if item.status == ExecutionStatus.SUCCEEDED]

        if pending_decisions or running:
            proposal.expected_information_gain = clamp(proposal.expected_information_gain * 0.75)
            proposal.reason += " A similar action is already queued or in flight."

        if failed:
            failed_runtime = sum(item.runtime_seconds or 0.0 for item in failed)
            failed_cost = sum(item.cost_estimate_usd or 0.0 for item in failed)
            avg_failed_artifact_quality = (
                sum(item.artifact_quality or 0.0 for item in failed) / len(failed)
            )
            runtime_pressure = clamp(failed_runtime / 1800.0)
            cost_pressure = clamp(failed_cost / 5.0)
            artifact_pressure = clamp(1.0 - avg_failed_artifact_quality)
            penalty = (
                (0.75 ** min(len(failed), 3))
                * (1.0 - 0.20 * runtime_pressure)
                * (1.0 - 0.20 * cost_pressure)
                * (1.0 - 0.10 * artifact_pressure)
            )
            proposal.expected_information_gain = clamp(proposal.expected_information_gain * penalty)
            proposal.reason += (
                f" This action has failed {len(failed)} time(s) already, so the controller is "
                "discounting repeat attempts."
            )
            if runtime_pressure > 0.0 or cost_pressure > 0.0:
                proposal.reason += (
                    f" Failed runs have already burned about {failed_runtime:.1f}s and "
                    f"${failed_cost:.2f} on this action."
                )

        if succeeded:
            avg_success_artifact_quality = (
                sum(item.artifact_quality or 0.0 for item in succeeded) / len(succeeded)
            )
            if action_matches(
                proposal.action_type,
                ActionType.GENERATE_IDEA,
                ActionType.GENERATE_METHOD,
                ActionType.SYNTHESIZE_PAPER,
                ActionType.PROPOSE_HYPOTHESIS,
                ActionType.DESIGN_MUTATION,
                ActionType.SYNTHESIZE_REPORT,
                ActionType.PROMOTE_WINNER,
            ):
                proposal.expected_information_gain = clamp(
                    proposal.expected_information_gain * 0.35
                )
                proposal.reason += " This is a one-shot stage and has already succeeded once."
            elif proposal.action_type == ActionType.TRIAGE_ATTACK:
                proposal.expected_information_gain = clamp(
                    proposal.expected_information_gain * 0.8
                )
                proposal.reason += " Similar critique work already completed recently."
            else:
                quality_pressure = clamp(1.0 - avg_success_artifact_quality)
                proposal.expected_information_gain = clamp(
                    proposal.expected_information_gain * (0.92 - 0.10 * (1.0 - quality_pressure))
                )
                proposal.reason += " The controller is slightly discounting a repeated successful action."

        return proposal

    def _build_summary(self, ledger: EpistemicLedger, primary: ActionProposal) -> str:
        claim_count = len([claim for claim in ledger.list_claims() if claim.status != ClaimStatus.ARCHIVED])
        if action_matches(primary.action_type, ActionType.SYNTHESIZE_PAPER, ActionType.SYNTHESIZE_REPORT):
            return (
                f"{claim_count} active claims tracked. The top claim is stable enough that paper "
                f"synthesis/reporting is now the best move."
            )
        if action_matches(primary.action_type, ActionType.GENERATE_METHOD, ActionType.DESIGN_MUTATION):
            return (
                f"{claim_count} active claims tracked. The top claim is blocked on the next "
                f"mutation/design step."
            )
        return (
            f"{claim_count} active claims tracked. The highest-value next move is "
            f"`{primary.action_type.value}` on `{primary.claim_title}`."
        )

    def _denario_meta(self, claim: Claim) -> dict[str, Any]:
        raw = claim.metadata.get("denario", {})
        return raw if isinstance(raw, dict) else {}

    def _autoresearch_meta(self, claim: Claim) -> dict[str, Any]:
        raw = claim.metadata.get("autoresearch", {})
        return raw if isinstance(raw, dict) else {}

    def _priority(self, score: float) -> str:
        if score >= 0.75:
            return "now"
        if score >= 0.55:
            return "next"
        return "watch"
