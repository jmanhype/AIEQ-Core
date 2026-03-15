from __future__ import annotations

from pathlib import Path
from typing import Any

from .ledger import EpistemicLedger
from .models import (
    ActionExecutor,
    ActionProposal,
    ActionType,
    ArtifactKind,
    Claim,
    ClaimStatus,
    ControllerDecision,
    ExecutionStatus,
    clamp,
)
from .policy import ExpectedInformationGainPolicy


class ResearchController:
    """Chooses the next research move from the current ledger state."""

    def __init__(self, policy: ExpectedInformationGainPolicy | None = None) -> None:
        self.policy = policy or ExpectedInformationGainPolicy()

    def decide(
        self, ledger: EpistemicLedger, *, backlog_limit: int = 5
    ) -> ControllerDecision:
        claims = [claim for claim in ledger.list_claims() if claim.status != ClaimStatus.ARCHIVED]
        if not claims:
            primary = ActionProposal(
                claim_id="",
                claim_title="ledger bootstrap",
                action_type=ActionType.GENERATE_IDEA,
                expected_information_gain=1.0,
                priority="now",
                reason=(
                    "The ledger is empty. The highest-leverage move is to generate the first "
                    "candidate claim and import it into the graph."
                ),
                executor=ActionExecutor.DENARIO,
                stage="bootstrap",
                command_hint=(
                    f"Run Denario to create a project, then import it with: "
                    f"PYTHONPATH=src python -m aieq_core.cli import-denario-project "
                    f"{ledger.path} --project-dir <denario-project-dir>"
                ),
            )
            return ControllerDecision(
                queue_state="bootstrap",
                summary="No claims exist yet, so the controller should bootstrap the graph.",
                primary_action=primary,
                backlog=[],
            )

        proposals = self._specialized_proposals(ledger, claims)
        proposals.extend(self._enrich_policy_actions(ledger))
        proposals = [self._apply_history_feedback(ledger, item) for item in proposals]
        ranked = sorted(
            proposals,
            key=lambda item: item.expected_information_gain,
            reverse=True,
        )

        primary = ranked[0]
        backlog = ranked[1:backlog_limit]
        summary = self._build_summary(ledger, primary)
        return ControllerDecision(
            queue_state=primary.stage or "exploration",
            summary=summary,
            primary_action=primary,
            backlog=backlog,
        )

    def _specialized_proposals(
        self, ledger: EpistemicLedger, claims: list[Claim]
    ) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
        for claim in claims:
            metrics = ledger.claim_metrics(claim.id)
            denario_meta = self._denario_meta(claim)
            artifacts = ledger.artifacts_for_claim(claim.id)
            has_method = bool(
                metrics["method_artifact_count"]
                or str(denario_meta.get("method", "")).strip()
                or any(item.kind == ArtifactKind.METHOD for item in artifacts)
            )
            has_project = bool(str(denario_meta.get("project_dir", "")).strip())
            has_paper = bool(
                metrics["paper_artifact_count"]
                or denario_meta.get("paper_paths")
                or any(item.kind == ArtifactKind.PAPER for item in artifacts)
            )
            evidence_count = metrics["evidence_count"]
            support_score = metrics["support_score"]
            uncertainty = metrics["uncertainty"]
            attack_count = metrics["open_attack_count"]

            if has_project and not has_method:
                score = clamp(
                    0.35 * claim.novelty
                    + 0.25 * claim.falsifiability
                    + 0.20 * uncertainty
                    + 0.35
                )
                proposals.append(
                    ActionProposal(
                        claim_id=claim.id,
                        claim_title=claim.title,
                        action_type=ActionType.GENERATE_METHOD,
                        expected_information_gain=score,
                        priority=self._priority(score),
                        reason=(
                            "The claim has Denario project context but no method yet. "
                            "Generating methodology is the bottleneck before running new tests."
                        ),
                        executor=ActionExecutor.DENARIO,
                        stage="generation",
                        command_hint=(
                            f"Open the Denario project at {denario_meta['project_dir']}, run "
                            f"`get_method()`, then re-import it with: PYTHONPATH=src python -m "
                            f"aieq_core.cli import-denario-project {ledger.path} --project-dir "
                            f"{denario_meta['project_dir']} --claim-id {claim.id}"
                        ),
                    )
                )

            if (
                claim.status in {ClaimStatus.ACTIVE, ClaimStatus.SUPPORTED}
                and evidence_count >= 2
                and attack_count == 0
                and support_score >= 0.8
                and has_project
            ):
                score = clamp(
                    0.40 * claim.confidence
                    + 0.20 * claim.novelty
                    + 0.20 * (1.0 - uncertainty)
                    + 0.20 * (0.0 if has_paper else 1.0)
                )
                proposals.append(
                    ActionProposal(
                        claim_id=claim.id,
                        claim_title=claim.title,
                        action_type=ActionType.SYNTHESIZE_PAPER,
                        expected_information_gain=score,
                        priority=self._priority(score),
                        reason=(
                            "The claim is supported by multiple evidence records and has no open "
                            "attacks, so synthesis is now higher value than more blind search."
                        ),
                        executor=ActionExecutor.DENARIO,
                        stage="synthesis",
                        command_hint=(
                            f"Run Denario paper generation for {denario_meta['project_dir']}, "
                            f"then re-import with: PYTHONPATH=src python -m aieq_core.cli "
                            f"import-denario-project {ledger.path} --project-dir "
                            f"{denario_meta['project_dir']} --claim-id {claim.id}"
                        ),
                    )
                )

        return proposals

    def _enrich_policy_actions(self, ledger: EpistemicLedger) -> list[ActionProposal]:
        enriched: list[ActionProposal] = []
        for proposal in self.policy.rank_actions(ledger, limit=50):
            claim = ledger.get_claim(proposal.claim_id)
            metrics = ledger.claim_metrics(proposal.claim_id)
            denario_meta = self._denario_meta(claim)
            autoresearch_meta = self._autoresearch_meta(claim)
            results_tsv_path = str(autoresearch_meta.get("results_tsv_path", "")).strip()
            preferred_branch = str(autoresearch_meta.get("branch", "")).strip()
            if proposal.action_type == ActionType.RUN_EXPERIMENT:
                proposal.executor = ActionExecutor.AUTORESEARCH
                proposal.stage = "experimentation"
                series_hint = ""
                if results_tsv_path:
                    series_name = Path(results_tsv_path).name
                    branch_fragment = f" for `{preferred_branch}`" if preferred_branch else ""
                    series_hint = (
                        f" If `{series_name}`{branch_fragment} changed, refresh it with: "
                        f"PYTHONPATH=src python -m aieq_core.cli import-autoresearch-results "
                        f"{ledger.path} --claim-id {proposal.claim_id} --results-tsv "
                        f"{results_tsv_path}"
                        f"{f' --branch {preferred_branch}' if preferred_branch else ''}"
                    )
                proposal.command_hint = (
                    "Run an autoresearch experiment, capture the run log, then import it with: "
                    f"PYTHONPATH=src python -m aieq_core.cli import-autoresearch-run {ledger.path} "
                    f"--claim-id {proposal.claim_id} --run-log <run.log> "
                    f"--description \"{claim.title}\""
                    f"{series_hint}"
                )
            elif proposal.action_type == ActionType.REPRODUCE_RESULT:
                if metrics["open_attack_count"] > 0:
                    proposal.expected_information_gain = clamp(
                        proposal.expected_information_gain * 0.55
                    )
                    proposal.reason += " Open attacks should be triaged before spending more budget on reproduction."
                proposal.executor = ActionExecutor.AUTORESEARCH
                proposal.stage = "reproduction"
                proposal.command_hint = (
                    "Repeat the experiment under a fresh autoresearch run and import the new log "
                    f"with claim id {proposal.claim_id}."
                )
            elif proposal.action_type == ActionType.TRIAGE_ATTACK:
                proposal.expected_information_gain = clamp(
                    proposal.expected_information_gain
                    + 0.25 * metrics["open_attack_load"]
                    + (0.10 if metrics["open_attack_count"] > 0 else 0.0)
                )
                proposal.executor = (
                    ActionExecutor.DENARIO if denario_meta.get("project_dir") else ActionExecutor.MANUAL
                )
                proposal.stage = "critique"
                if denario_meta.get("project_dir"):
                    proposal.command_hint = (
                        f"Use the Denario project at {denario_meta['project_dir']} to refine the "
                        f"claim or produce a referee/literature update, then re-import it with "
                        f"--claim-id {proposal.claim_id}."
                    )
                else:
                    proposal.command_hint = "Resolve or rebut the open attack, then update the ledger."
            elif proposal.action_type == ActionType.COLLECT_COUNTEREVIDENCE:
                proposal.executor = ActionExecutor.DENARIO
                proposal.stage = "critique"
                proposal.command_hint = (
                    "Collect negative or disconfirming evidence, ideally via Denario literature "
                    "or method analysis, then add it as evidence or attacks."
                )
            elif proposal.action_type == ActionType.CHALLENGE_ASSUMPTION:
                proposal.executor = ActionExecutor.MANUAL
                proposal.stage = "critique"
                proposal.command_hint = (
                    "Design a falsification test for the highest-risk assumption and record the "
                    "result as new evidence."
                )
            enriched.append(proposal)
        return enriched

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
            if proposal.action_type in {
                ActionType.GENERATE_IDEA,
                ActionType.GENERATE_METHOD,
                ActionType.SYNTHESIZE_PAPER,
            }:
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
        if primary.action_type == ActionType.SYNTHESIZE_PAPER:
            return (
                f"{claim_count} active claims tracked. The top claim is stable enough that paper "
                f"synthesis is now the best move."
            )
        if primary.action_type == ActionType.GENERATE_METHOD:
            return (
                f"{claim_count} active claims tracked. The top claim is blocked on methodology, "
                f"so Denario should generate the next executable plan."
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
