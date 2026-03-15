from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.adapters.autoresearch import AutoresearchAdapter
from aieq_core.controller import ResearchController
from aieq_core.ledger import EpistemicLedger
from aieq_core.models import ArtifactKind, EvidenceDirection


class ResearchControllerTests(unittest.TestCase):
    def test_empty_ledger_bootstraps_with_denario(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = EpistemicLedger.load(Path(tmpdir) / "ledger.json")
            decision = ResearchController().decide(ledger)

            self.assertEqual(decision.queue_state, "bootstrap")
            self.assertEqual(decision.primary_action.action_type.value, "generate_idea")
            self.assertEqual(decision.primary_action.executor.value, "denario")

    def test_missing_method_prefers_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = EpistemicLedger.load(Path(tmpdir) / "ledger.json")
            claim = ledger.add_claim(
                title="Sparse attention claim",
                statement="Sparse attention should improve bpb.",
                metadata={
                    "denario": {
                        "project_dir": "/tmp/denario-project",
                        "method": "",
                        "paper_paths": [],
                    }
                },
            )

            decision = ResearchController().decide(ledger)

            self.assertEqual(decision.primary_action.claim_id, claim.id)
            self.assertEqual(decision.primary_action.action_type.value, "generate_method")

    def test_method_artifact_prevents_redundant_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = EpistemicLedger.load(Path(tmpdir) / "ledger.json")
            claim = ledger.add_claim(
                title="Sparse attention claim",
                statement="Sparse attention should improve bpb.",
                metadata={
                    "denario": {
                        "project_dir": "/tmp/denario-project",
                        "method": "",
                        "paper_paths": [],
                    }
                },
            )
            ledger.add_artifact(
                claim_id=claim.id,
                kind=ArtifactKind.METHOD,
                title="Denario method for Sparse attention claim",
                content="Run the experiment and compare to baseline.",
                source_type="denario",
                source_path="/tmp/denario-project/input_files/methods.md",
            )

            decision = ResearchController().decide(ledger)
            queued_actions = [decision.primary_action, *decision.backlog]

            self.assertTrue(
                all(action.action_type.value != "generate_method" for action in queued_actions)
            )

    def test_supported_claim_without_attacks_prefers_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = EpistemicLedger.load(Path(tmpdir) / "ledger.json")
            claim = ledger.add_claim(
                title="Stable claim",
                statement="Two successful runs support the claim.",
                novelty=0.8,
                falsifiability=0.7,
                metadata={
                    "denario": {
                        "project_dir": "/tmp/denario-project",
                        "method": "Run the experiment and compare to baseline.",
                        "paper_paths": [],
                    }
                },
            )
            ledger.add_evidence(
                claim_id=claim.id,
                summary="Run one improved bpb.",
                direction=EvidenceDirection.SUPPORT,
                strength=0.9,
                confidence=0.9,
            )
            ledger.add_evidence(
                claim_id=claim.id,
                summary="Run two reproduced the gain.",
                direction=EvidenceDirection.SUPPORT,
                strength=0.9,
                confidence=0.9,
            )

            decision = ResearchController().decide(ledger)

            self.assertEqual(decision.primary_action.claim_id, claim.id)
            self.assertEqual(decision.primary_action.action_type.value, "synthesize_paper")

    def test_open_attack_prefers_critique(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = EpistemicLedger.load(Path(tmpdir) / "ledger.json")
            claim = ledger.add_claim(
                title="Contested claim",
                statement="The result looks promising but has a major caveat.",
                novelty=0.7,
                falsifiability=0.8,
                metadata={"denario": {"project_dir": "/tmp/denario-project", "method": "Method"}},
            )
            ledger.add_evidence(
                claim_id=claim.id,
                summary="Pilot result is positive.",
                direction=EvidenceDirection.SUPPORT,
                strength=0.8,
                confidence=0.9,
            )
            ledger.add_attack(
                claim_id=claim.id,
                description="Possible leakage issue.",
                severity=0.95,
            )

            decision = ResearchController().decide(ledger)

            self.assertEqual(decision.primary_action.claim_id, claim.id)
            self.assertEqual(decision.primary_action.action_type.value, "triage_attack")

    def test_failed_experiment_shifts_priority_to_assumption_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = EpistemicLedger.load(Path(tmpdir) / "ledger.json")
            claim = ledger.add_claim(
                title="Repeated experiment failures",
                statement="The same experiment should not be retried blindly.",
                novelty=0.8,
                falsifiability=0.9,
            )
            ledger.add_assumption(
                claim_id=claim.id,
                text="A hidden implementation constraint may be blocking the result.",
                risk=0.95,
            )
            decision = ledger.record_decision(
                ResearchController().decide(ledger).primary_action
            )
            ledger.record_execution(
                decision_id=decision.id,
                status="failed",
                notes="Pilot experiment crashed immediately.",
                runtime_seconds=900,
                cost_estimate_usd=4.2,
                artifact_quality=0.1,
            )

            follow_up = ResearchController().decide(ledger)

            self.assertEqual(follow_up.primary_action.claim_id, claim.id)
            self.assertEqual(
                follow_up.primary_action.action_type.value,
                "challenge_assumption",
            )

    def test_stagnant_autoresearch_series_shifts_priority_to_assumption_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_tsv = tmp / "results.tsv"
            results_tsv.write_text(
                "\n".join(
                    [
                        "commit\tval_bpb\tmemory_gb\tstatus\tdescription",
                        "a1\t1.000000\t44.0\tkeep\tbaseline",
                        "a2\t1.000400\t44.1\tdiscard\tincrease width slightly",
                        "a3\t1.000600\t44.2\tdiscard\tswap activation",
                        "a4\t0.000000\t0.0\tcrash\tdouble width oom",
                        "a5\t1.000300\t44.1\tdiscard\tchange optimizer",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ledger = EpistemicLedger.load(tmp / "ledger.json")
            claim = ledger.add_claim(
                title="Sparse attention claim",
                statement="Sparse attention should improve bpb.",
                novelty=0.8,
                falsifiability=0.9,
            )
            ledger.add_assumption(
                claim_id=claim.id,
                text="A hidden systems constraint may be blocking the result.",
                risk=0.95,
            )
            AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=results_tsv,
            )

            decision = ResearchController().decide(ledger)

            self.assertEqual(decision.primary_action.claim_id, claim.id)
            self.assertEqual(
                decision.primary_action.action_type.value,
                "challenge_assumption",
            )

    def test_preferred_autoresearch_branch_drives_experiment_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            main_results = tmp / "main.tsv"
            radical_results = tmp / "radical.tsv"
            main_results.write_text(
                "\n".join(
                    [
                        "commit\tval_bpb\tmemory_gb\tstatus\tdescription",
                        "m1\t1.000000\t44.0\tkeep\tbaseline",
                        "m2\t1.000300\t44.1\tdiscard\tsmall width change",
                        "m3\t1.000400\t44.1\tdiscard\tactivation swap",
                        "m4\t0.000000\t0.0\tcrash\toom",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            radical_results.write_text(
                "\n".join(
                    [
                        "commit\tval_bpb\tmemory_gb\tstatus\tdescription",
                        "r1\t1.000000\t44.0\tkeep\tbaseline",
                        "r2\t0.992500\t44.6\tkeep\tgrouped attention",
                        "r3\t0.993100\t44.5\tdiscard\textra residual gate",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ledger = EpistemicLedger.load(tmp / "ledger.json")
            claim = ledger.add_claim(
                title="Branch-aware claim",
                statement="The best active branch should guide the next experiment hint.",
                novelty=0.8,
                falsifiability=0.9,
            )
            AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=main_results,
                branch="main",
            )
            AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=radical_results,
                branch="radical",
            )

            decision = ResearchController().decide(ledger)
            queued_actions = [decision.primary_action, *decision.backlog]
            experiment_action = next(
                action for action in queued_actions if action.action_type.value == "run_experiment"
            )

            self.assertEqual(decision.primary_action.claim_id, claim.id)
            self.assertIn(str(radical_results), experiment_action.command_hint)
            self.assertIn("--branch radical", experiment_action.command_hint)


if __name__ == "__main__":
    unittest.main()
