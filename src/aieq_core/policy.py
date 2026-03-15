from __future__ import annotations

from .ledger import EpistemicLedger
from .models import ActionProposal, ActionType, ClaimStatus, clamp


class ExpectedInformationGainPolicy:
    """Ranks next actions by how much they could change the ledger's beliefs."""

    def rank_actions(
        self, ledger: EpistemicLedger, *, limit: int = 10
    ) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []

        for claim in ledger.list_claims():
            if claim.status == ClaimStatus.ARCHIVED:
                continue

            metrics = ledger.claim_metrics(claim.id)
            uncertainty = metrics["uncertainty"]
            attack_pressure = metrics["open_attack_load"]
            support_signal = clamp(metrics["support_score"])
            contradict_signal = clamp(metrics["contradict_score"])
            evidence_count = metrics["evidence_count"]
            series_failure_pressure = self._autoresearch_failure_pressure(metrics)
            series_momentum = self._autoresearch_momentum(metrics)
            series_plateau = self._autoresearch_plateau(metrics)

            experiment_score = clamp(
                0.40 * uncertainty
                + 0.25 * claim.novelty
                + 0.20 * claim.falsifiability
                + 0.15 * attack_pressure
                + (0.10 if evidence_count == 0 else 0.0)
                - 0.22 * series_failure_pressure
                + 0.08 * series_momentum
            )
            proposals.append(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.RUN_EXPERIMENT,
                    expected_information_gain=experiment_score,
                    priority=self._priority(experiment_score),
                    reason=(
                        "Novel claim remains unresolved; a bounded experiment should move "
                        "belief faster than more debate."
                    ),
                )
            )

            if metrics["highest_risk_assumption_id"]:
                assumption_score = clamp(
                    0.50 * metrics["highest_risk_assumption_risk"]
                    + 0.25 * uncertainty
                    + 0.15 * claim.novelty
                    + 0.10 * attack_pressure
                    + 0.14 * series_failure_pressure
                    + (0.10 if series_plateau else 0.0)
                )
                proposals.append(
                    ActionProposal(
                        claim_id=claim.id,
                        claim_title=claim.title,
                        action_type=ActionType.CHALLENGE_ASSUMPTION,
                        expected_information_gain=assumption_score,
                        priority=self._priority(assumption_score),
                        reason=(
                            "The riskiest assumption is a likely epistemic bottleneck; "
                            "stress-testing it could collapse or strengthen the claim."
                        ),
                    )
                )

            if metrics["open_attack_count"] > 0:
                attack_score = clamp(
                    0.55 * attack_pressure
                    + 0.20 * uncertainty
                    + 0.15 * claim.falsifiability
                    + 0.10 * claim.novelty
                )
                proposals.append(
                    ActionProposal(
                        claim_id=claim.id,
                        claim_title=claim.title,
                        action_type=ActionType.TRIAGE_ATTACK,
                        expected_information_gain=attack_score,
                        priority=self._priority(attack_score),
                        reason=(
                            "Open attacks are unresolved falsifiers. Closing them converts "
                            "generic skepticism into explicit evidence."
                        ),
                    )
                )

            if evidence_count > 0:
                counterevidence_score = clamp(
                    0.40 * abs(support_signal - contradict_signal)
                    + 0.25 * claim.novelty
                    + 0.20 * uncertainty
                    + 0.15 * claim.falsifiability
                    + 0.08 * series_failure_pressure
                    + (0.10 if series_plateau else 0.0)
                )
                proposals.append(
                    ActionProposal(
                        claim_id=claim.id,
                        claim_title=claim.title,
                        action_type=ActionType.COLLECT_COUNTEREVIDENCE,
                        expected_information_gain=counterevidence_score,
                        priority=self._priority(counterevidence_score),
                        reason=(
                            "Current evidence leans in one direction; targeted counterevidence "
                            "would reduce overfitting to the first positive result."
                        ),
                    )
                )

            if support_signal > 0.55 and evidence_count <= 1:
                reproduce_score = clamp(
                    0.45 * support_signal
                    + 0.25 * claim.novelty
                    + 0.20 * claim.falsifiability
                    + 0.10 * uncertainty
                    + 0.12 * series_momentum
                    - 0.10 * series_failure_pressure
                )
                proposals.append(
                    ActionProposal(
                        claim_id=claim.id,
                        claim_title=claim.title,
                        action_type=ActionType.REPRODUCE_RESULT,
                        expected_information_gain=reproduce_score,
                        priority=self._priority(reproduce_score),
                        reason=(
                            "A single positive result is fragile. Reproduction is the fastest "
                            "way to convert a promising anecdote into evidence."
                        ),
                    )
                )

        ranked = sorted(
            proposals,
            key=lambda item: item.expected_information_gain,
            reverse=True,
        )
        return ranked[:limit]

    def _autoresearch_failure_pressure(self, metrics: dict[str, float]) -> float:
        run_count = int(metrics["autoresearch_series_run_count"])
        if run_count == 0:
            return 0.0
        stagnation_pressure = clamp(
            metrics["autoresearch_series_stagnation_run_count"] / max(run_count, 1)
        )
        crash_pressure = clamp(metrics["autoresearch_series_crash_rate"])
        low_yield_pressure = clamp(1.0 - metrics["autoresearch_series_keep_rate"])
        base_pressure = clamp(
            0.45 * stagnation_pressure
            + 0.35 * crash_pressure
            + 0.20 * low_yield_pressure
        )
        branch_count = int(metrics.get("autoresearch_branch_count", 0))
        active_branch_count = int(metrics.get("autoresearch_active_branch_count", 0))
        if branch_count > 0 and active_branch_count > 0:
            relief = clamp(active_branch_count / branch_count)
            return clamp(base_pressure * (1.0 - 0.25 * relief))
        return base_pressure

    def _autoresearch_momentum(self, metrics: dict[str, float]) -> float:
        run_count = int(metrics["autoresearch_series_run_count"])
        if run_count == 0:
            return 0.0
        improvement_signal = clamp(
            max(metrics["autoresearch_series_best_improvement_bpb"], 0.0) / 0.005
        )
        frontier_signal = clamp(
            metrics["autoresearch_series_frontier_improvement_count"] / 3.0
        )
        branch_count = int(metrics.get("autoresearch_branch_count", 0))
        active_branch_count = int(metrics.get("autoresearch_active_branch_count", 0))
        branch_activity = (
            clamp(active_branch_count / branch_count)
            if branch_count > 0
            else 0.0
        )
        return clamp(0.60 * improvement_signal + 0.25 * frontier_signal + 0.15 * branch_activity)

    def _autoresearch_plateau(self, metrics: dict[str, float]) -> bool:
        branch_count = int(metrics.get("autoresearch_branch_count", 0))
        plateau_branch_count = int(metrics.get("autoresearch_plateau_branch_count", 0))
        active_branch_count = int(metrics.get("autoresearch_active_branch_count", 0))
        if branch_count > 0:
            return plateau_branch_count >= branch_count and active_branch_count == 0
        return (
            int(metrics["autoresearch_series_run_count"]) >= 4
            and int(metrics["autoresearch_series_stagnation_run_count"]) >= 4
            and float(metrics["autoresearch_series_best_improvement_bpb"]) <= 0.0005
        )

    def _priority(self, score: float) -> str:
        if score >= 0.75:
            return "now"
        if score >= 0.55:
            return "next"
        return "watch"
