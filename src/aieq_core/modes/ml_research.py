from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import (
    ActionExecutor,
    ActionProposal,
    ActionType,
    ArtifactKind,
    Claim,
    ClaimStatus,
    action_matches,
    clamp,
)
from ..policy import ExpectedInformationGainPolicy
from ..runtime import doctor_report
from .base import ModeAdapter


class MLResearchMode(ModeAdapter):
    name = "ml_research"
    label = "ML Research"
    description = "Denario + autoresearch hybrid for hypothesis generation, critique, and GPU-backed experiments."

    def __init__(self, policy: ExpectedInformationGainPolicy | None = None) -> None:
        self.policy = policy or ExpectedInformationGainPolicy()

    def bootstrap_proposal(self, *, ledger: Any) -> ActionProposal | None:
        return ActionProposal(
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
            mode=self.name,
            stage="bootstrap",
            command_hint=(
                f"Run Denario to create a project, then import it with: "
                f"PYTHONPATH=src python -m aieq_core.cli import-denario-project "
                f"{ledger.path} --project-dir <denario-project-dir>"
            ),
        )

    def build_proposals(self, *, ledger: Any, claim: Claim) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
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
                    mode=self.name,
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
                    mode=self.name,
                    stage="synthesis",
                    command_hint=(
                        f"Run Denario paper generation for {denario_meta['project_dir']}, "
                        f"then re-import with: PYTHONPATH=src python -m aieq_core.cli "
                        f"import-denario-project {ledger.path} --project-dir "
                        f"{denario_meta['project_dir']} --claim-id {claim.id}"
                    ),
                )
            )

        for proposal in self.policy.rank_actions(ledger, limit=50):
            if proposal.claim_id != claim.id:
                continue
            proposals.append(self._enrich_policy_action(ledger=ledger, claim=claim, proposal=proposal))

        return proposals

    def doctor(self, *, config: Any, ledger_path: str | None = None) -> dict[str, Any]:
        payload = doctor_report(config, ledger_path=ledger_path)
        payload["mode"] = self.name
        return payload

    def execute_action(
        self,
        *,
        orchestrator: Any,
        ledger: Any,
        proposal: ActionProposal,
        decision_id: str,
        data_description: str,
        data_description_file: str,
    ) -> dict[str, Any]:
        if action_matches(proposal.action_type, ActionType.GENERATE_IDEA, ActionType.PROPOSE_HYPOTHESIS):
            return orchestrator._execute_generate_idea(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
                data_description=data_description,
                data_description_file=data_description_file,
            )
        if action_matches(proposal.action_type, ActionType.GENERATE_METHOD, ActionType.DESIGN_MUTATION):
            return orchestrator._execute_generate_method(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        if action_matches(proposal.action_type, ActionType.SYNTHESIZE_PAPER, ActionType.SYNTHESIZE_REPORT):
            return orchestrator._execute_synthesize_paper(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        if action_matches(proposal.action_type, ActionType.RUN_EXPERIMENT, ActionType.REPRODUCE_RESULT, ActionType.RUN_EVAL):
            return orchestrator._execute_autoresearch(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        raise orchestrator.unsupported_action_error(proposal.action_type.value)

    @staticmethod
    def _denario_meta(claim: Claim) -> dict[str, Any]:
        raw = claim.metadata.get("denario", {})
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _autoresearch_meta(claim: Claim) -> dict[str, Any]:
        raw = claim.metadata.get("autoresearch", {})
        return raw if isinstance(raw, dict) else {}

    def _enrich_policy_action(
        self,
        *,
        ledger: Any,
        claim: Claim,
        proposal: ActionProposal,
    ) -> ActionProposal:
        metrics = ledger.claim_metrics(proposal.claim_id)
        denario_meta = self._denario_meta(claim)
        autoresearch_meta = self._autoresearch_meta(claim)
        results_tsv_path = str(autoresearch_meta.get("results_tsv_path", "")).strip()
        preferred_branch = str(autoresearch_meta.get("branch", "")).strip()
        proposal.mode = self.name
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
        return proposal

    @staticmethod
    def _priority(score: float) -> str:
        if score >= 0.75:
            return "now"
        if score >= 0.55:
            return "next"
        return "watch"
