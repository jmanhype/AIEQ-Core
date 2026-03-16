from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.ledger import EpistemicLedger
from aieq_core.models import (
    ActionExecutor,
    ActionProposal,
    ActionType,
    ArtifactKind,
    ClaimStatus,
    EvidenceDirection,
    ExecutionStatus,
    canonical_action_type,
)
from aieq_core.policy import ExpectedInformationGainPolicy


class EpistemicLedgerTests(unittest.TestCase):
    def test_roundtrip_and_status_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)

            claim = ledger.add_claim(
                title="Sparse attention improves short-budget pretraining",
                statement="A sparse attention variant should reduce validation bpb.",
                novelty=0.8,
                falsifiability=0.9,
            )
            ledger.add_assumption(
                claim_id=claim.id,
                text="The sparse kernel does not destroy training stability.",
                risk=0.7,
            )
            ledger.add_evidence(
                claim_id=claim.id,
                summary="First run improved validation bpb by 0.003.",
                direction=EvidenceDirection.SUPPORT,
                strength=0.8,
                confidence=0.9,
                source_type="autoresearch",
                source_ref="commit:abc1234",
            )

            reloaded = EpistemicLedger.load(ledger_path)
            snapshot = reloaded.claim_snapshot(claim.id)

            self.assertEqual(snapshot["claim"]["status"], ClaimStatus.SUPPORTED.value)
            self.assertEqual(snapshot["metrics"]["evidence_count"], 1)
            self.assertGreater(snapshot["claim"]["confidence"], 0.7)

    def test_open_attack_keeps_claim_contested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Generated method generalizes",
                statement="The method should generalize to held-out data.",
                novelty=0.6,
                falsifiability=0.8,
            )
            ledger.add_evidence(
                claim_id=claim.id,
                summary="Held-out improvement is promising.",
                direction=EvidenceDirection.SUPPORT,
                strength=0.8,
                confidence=0.9,
            )
            ledger.add_attack(
                claim_id=claim.id,
                description="Potential train-test leakage in preprocessing.",
                severity=0.9,
            )

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)
            self.assertEqual(snapshot["claim"]["status"], ClaimStatus.CONTESTED.value)
            self.assertEqual(snapshot["metrics"]["open_attack_count"], 1)

    def test_artifact_roundtrip_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Artifact-backed claim",
                statement="Methods and papers should be first-class ledger nodes.",
            )
            method = ledger.add_artifact(
                claim_id=claim.id,
                kind=ArtifactKind.METHOD,
                title="Method draft",
                content="Run the sparse attention comparison.",
                source_type="denario",
                source_path="/tmp/project/input_files/methods.md",
            )
            paper = ledger.add_artifact(
                claim_id=claim.id,
                kind=ArtifactKind.PAPER,
                title="paper.tex",
                content="\\\\section{Results} Promising findings.",
                source_type="denario",
                source_path="/tmp/project/paper/paper.tex",
            )

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)

            self.assertEqual(snapshot["metrics"]["artifact_count"], 2)
            self.assertEqual(snapshot["metrics"]["method_artifact_count"], 1)
            self.assertEqual(snapshot["metrics"]["paper_artifact_count"], 1)
            self.assertEqual(snapshot["artifacts"][0]["id"], method.id)
            self.assertEqual(snapshot["artifacts"][1]["id"], paper.id)

    def test_policy_prefers_experiments_for_unresolved_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="AIEQ quarantine finds neglected architectures",
                statement="Quarantining dominant papers should surface better architectures.",
                novelty=0.95,
                falsifiability=0.85,
            )
            ledger.add_assumption(
                claim_id=claim.id,
                text="The quarantine heuristic does not collapse into random search.",
                risk=0.8,
            )

            ranked = ExpectedInformationGainPolicy().rank_actions(ledger, limit=3)

            self.assertEqual(ranked[0].claim_id, claim.id)
            self.assertEqual(ranked[0].action_type.value, "run_experiment")

    def test_decision_and_execution_history_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="History-aware claim",
                statement="The controller should remember prior actions.",
            )
            decision = ledger.record_decision(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.RUN_EXPERIMENT,
                    expected_information_gain=0.8,
                    priority="now",
                    reason="Run a pilot experiment.",
                    executor=ActionExecutor.AUTORESEARCH,
                    stage="experimentation",
                    command_hint="uv run train.py > run.log 2>&1",
                )
            )
            execution = ledger.record_execution(
                decision_id=decision.id,
                status=ExecutionStatus.FAILED,
                notes="OOM at step 1.",
                runtime_seconds=612.5,
                cost_estimate_usd=2.4,
                artifact_quality=0.2,
            )

            reloaded = EpistemicLedger.load(ledger_path)
            snapshot = reloaded.claim_snapshot(claim.id)

            self.assertEqual(snapshot["metrics"]["decision_count"], 1)
            self.assertEqual(snapshot["metrics"]["execution_count"], 1)
            self.assertEqual(snapshot["metrics"]["failed_execution_count"], 1)
            self.assertEqual(snapshot["metrics"]["failed_runtime_seconds"], 612.5)
            self.assertEqual(snapshot["metrics"]["failed_cost_usd"], 2.4)
            self.assertEqual(snapshot["metrics"]["average_artifact_quality"], 0.2)
            self.assertEqual(snapshot["decisions"][0]["id"], decision.id)
            self.assertEqual(snapshot["executions"][0]["id"], execution.id)

    def test_claim_metrics_include_autoresearch_series_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Series-aware claim",
                statement="The controller should see aggregate experiment history.",
                metadata={
                    "autoresearch": {
                        "aggregate_series": {
                            "branch_count": 2,
                            "active_branch_count": 1,
                            "plateau_branch_count": 1,
                            "total_runs": 9,
                            "keep_rate": 0.3,
                            "crash_rate": 0.111111,
                            "preferred_branch": "radical",
                        },
                        "series": {
                            "total_runs": 6,
                            "keep_rate": 0.25,
                            "crash_rate": 0.166667,
                            "frontier_improvement_count": 1,
                            "stagnation_run_count": 4,
                            "best_improvement_bpb": 0.0012,
                            "average_memory_gb": 44.3,
                        }
                    }
                },
            )

            metrics = ledger.claim_metrics(claim.id)

            self.assertEqual(metrics["autoresearch_series_run_count"], 6)
            self.assertEqual(metrics["autoresearch_series_frontier_improvement_count"], 1)
            self.assertAlmostEqual(metrics["autoresearch_series_keep_rate"], 0.25)
            self.assertAlmostEqual(metrics["autoresearch_series_crash_rate"], 0.166667)
            self.assertEqual(metrics["autoresearch_branch_count"], 2)
            self.assertEqual(metrics["autoresearch_active_branch_count"], 1)
            self.assertEqual(metrics["autoresearch_plateau_branch_count"], 1)
            self.assertEqual(metrics["autoresearch_total_run_count_all_branches"], 9)
            self.assertEqual(metrics["autoresearch_best_branch"], "radical")
            self.assertAlmostEqual(metrics["autoresearch_aggregate_keep_rate"], 0.3)
            self.assertAlmostEqual(metrics["autoresearch_aggregate_crash_rate"], 0.111111)
            self.assertAlmostEqual(
                metrics["autoresearch_series_best_improvement_bpb"],
                0.0012,
            )
            self.assertAlmostEqual(
                metrics["autoresearch_series_average_memory_gb"],
                44.3,
            )

    def test_generic_target_eval_candidate_roundtrip_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Optimize a support skill",
                statement="The prompt should become more reliable against its eval suite.",
                metadata={"mode": "skill_optimizer"},
            )
            target = ledger.register_target(
                claim_id=claim.id,
                mode="skill_optimizer",
                target_type="prompt_template",
                title="Support prompt",
                content="Answer the user clearly.",
                invariant_constraints={"required_phrases": ["Answer"]},
            )
            suite = ledger.register_eval_suite(
                claim_id=claim.id,
                target_id=target.id,
                name="clarity-suite",
                compatible_target_type="prompt_template",
                pass_threshold=0.75,
                repetitions=2,
                cases=[
                    {
                        "id": "case-1",
                        "input": "Explain the refund policy.",
                        "criteria": [{"type": "contains_all", "values": ["refund", "policy"]}],
                    }
                ],
            )
            candidate = ledger.add_mutation_candidate(
                claim_id=claim.id,
                target_id=target.id,
                summary="Tightened instruction ordering.",
                content="Answer the user clearly and mention the refund policy when relevant.",
                review_status="approved",
            )
            ledger.record_eval_run(
                claim_id=claim.id,
                target_id=target.id,
                suite_id=suite.id,
                candidate_id=candidate.id,
                case_id="case-1",
                run_index=1,
                score=1.0,
                passed=True,
                raw_output="Refund policy details.",
            )
            ledger.record_eval_run(
                claim_id=claim.id,
                target_id=target.id,
                suite_id=suite.id,
                candidate_id=candidate.id,
                case_id="case-1",
                run_index=2,
                score=0.5,
                passed=False,
                raw_output="Policy details.",
            )
            ledger.promote_candidate(target_id=target.id, candidate_id=candidate.id)

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)
            metrics = snapshot["metrics"]

            self.assertEqual(metrics["target_count"], 1)
            self.assertEqual(metrics["eval_suite_count"], 1)
            self.assertEqual(metrics["mutation_candidate_count"], 1)
            self.assertEqual(metrics["eval_run_count"], 2)
            self.assertEqual(metrics["approved_candidate_count"], 1)
            self.assertEqual(metrics["promoted_candidate_count"], 1)
            self.assertEqual(metrics["optimization_best_candidate_id"], candidate.id)
            self.assertAlmostEqual(metrics["optimization_best_score"], 0.75)
            self.assertAlmostEqual(metrics["optimization_average_pass_rate"], 0.5)
            self.assertTrue(metrics["optimization_threshold_met"])
            self.assertEqual(snapshot["targets"][0]["promoted_candidate_id"], candidate.id)

    def test_legacy_actions_map_to_generic_actions(self) -> None:
        self.assertEqual(
            canonical_action_type(ActionType.GENERATE_IDEA),
            ActionType.PROPOSE_HYPOTHESIS,
        )
        self.assertEqual(
            canonical_action_type(ActionType.GENERATE_METHOD),
            ActionType.DESIGN_MUTATION,
        )
        self.assertEqual(
            canonical_action_type(ActionType.RUN_EXPERIMENT),
            ActionType.RUN_EVAL,
        )
        self.assertEqual(
            canonical_action_type(ActionType.SYNTHESIZE_PAPER),
            ActionType.SYNTHESIZE_REPORT,
        )


if __name__ == "__main__":
    unittest.main()
