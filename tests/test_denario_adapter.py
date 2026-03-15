from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.adapters.denario import DenarioAdapter
from aieq_core.ledger import EpistemicLedger
from aieq_core.models import ActionExecutor, ActionProposal, ActionType


class DenarioAdapterTests(unittest.TestCase):
    def test_import_project_creates_claim_evidence_and_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "denario_project"
            input_dir = project / "input_files"
            plots_dir = input_dir / "plots"
            paper_dir = project / "paper"
            plots_dir.mkdir(parents=True)
            paper_dir.mkdir(parents=True)

            (input_dir / "idea.md").write_text(
                "# Sparse Attention for Short-Budget Training\n\nUse a sparse attention schedule to improve validation bpb.",
                encoding="utf-8",
            )
            (input_dir / "methods.md").write_text(
                "Train the model with a sparse attention variant and compare against baseline.",
                encoding="utf-8",
            )
            (input_dir / "results.md").write_text(
                "Validation bpb improved in the pilot run and produced promising plots.",
                encoding="utf-8",
            )
            (input_dir / "literature.md").write_text(
                "Similar sparse-attention ideas appear in prior work, so novelty needs checking.",
                encoding="utf-8",
            )
            (input_dir / "referee.md").write_text(
                "The draft paper is promising but the methodology section is underspecified.",
                encoding="utf-8",
            )
            (plots_dir / "plot_1.png").write_text("fake-image", encoding="utf-8")
            (paper_dir / "paper_v4_final.tex").write_text(
                "\\section{Results} Promising pilot findings.",
                encoding="utf-8",
            )

            ledger = EpistemicLedger.load(root / "ledger.json")
            imported = DenarioAdapter.import_project(
                ledger=ledger,
                project_dir=project,
                results_direction="support",
            )

            claim = imported["claim"]
            self.assertEqual(claim["title"], "Sparse Attention for Short-Budget Training")
            self.assertEqual(imported["method_artifact"]["kind"], "method")
            self.assertEqual(len(imported["paper_artifacts"]), 1)
            self.assertEqual(imported["paper_artifacts"][0]["kind"], "paper")
            self.assertEqual(imported["results_evidence"]["direction"], "support")
            self.assertEqual(len(imported["attacks"]), 2)

            snapshot = EpistemicLedger.load(root / "ledger.json").claim_snapshot(claim["id"])
            self.assertEqual(snapshot["metrics"]["evidence_count"], 1)
            self.assertEqual(snapshot["metrics"]["open_attack_count"], 2)
            self.assertEqual(snapshot["metrics"]["artifact_count"], 2)
            self.assertEqual(snapshot["metrics"]["method_artifact_count"], 1)
            self.assertEqual(snapshot["metrics"]["paper_artifact_count"], 1)

    def test_import_project_into_existing_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            input_dir = project / "input_files"
            input_dir.mkdir(parents=True)
            (input_dir / "results.md").write_text("Generated results exist.", encoding="utf-8")

            ledger = EpistemicLedger.load(root / "ledger.json")
            claim = ledger.add_claim(
                title="Existing claim",
                statement="Use Denario to populate claim artifacts.",
            )

            imported = DenarioAdapter.import_project(
                ledger=ledger,
                project_dir=project,
                claim_id=claim.id,
            )

            self.assertEqual(imported["claim"]["id"], claim.id)
            snapshot = EpistemicLedger.load(root / "ledger.json").claim_snapshot(claim.id)
            self.assertEqual(snapshot["metrics"]["evidence_count"], 1)
            self.assertEqual(snapshot["metrics"]["artifact_count"], 0)

    def test_reimport_project_updates_method_and_paper_artifacts_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "denario_project"
            input_dir = project / "input_files"
            paper_dir = project / "paper"
            input_dir.mkdir(parents=True)
            paper_dir.mkdir(parents=True)
            (input_dir / "idea.md").write_text("# Existing claim", encoding="utf-8")
            (input_dir / "methods.md").write_text(
                "Version one of the method.",
                encoding="utf-8",
            )
            (input_dir / "results.md").write_text("Generated results exist.", encoding="utf-8")
            paper_path = paper_dir / "paper.tex"
            paper_path.write_text("\\section{Method} Version one.", encoding="utf-8")

            ledger = EpistemicLedger.load(root / "ledger.json")
            claim = ledger.add_claim(title="Existing claim", statement="Use Denario artifacts.")
            first = DenarioAdapter.import_project(
                ledger=ledger,
                project_dir=project,
                claim_id=claim.id,
            )

            (input_dir / "methods.md").write_text(
                "Version two of the method with a tighter ablation plan.",
                encoding="utf-8",
            )
            paper_path.write_text("\\section{Method} Version two.", encoding="utf-8")
            second = DenarioAdapter.import_project(
                ledger=ledger,
                project_dir=project,
                claim_id=claim.id,
            )

            self.assertEqual(first["method_artifact"]["id"], second["method_artifact"]["id"])
            self.assertEqual(first["paper_artifacts"][0]["id"], second["paper_artifacts"][0]["id"])
            self.assertIn("Version two", second["method_artifact"]["content"])
            self.assertIn("Version two", second["paper_artifacts"][0]["content"])
            snapshot = EpistemicLedger.load(root / "ledger.json").claim_snapshot(claim.id)
            self.assertEqual(snapshot["metrics"]["artifact_count"], 2)
            self.assertEqual(snapshot["metrics"]["method_artifact_count"], 1)
            self.assertEqual(snapshot["metrics"]["paper_artifact_count"], 1)

    def test_import_project_closes_bootstrap_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "denario_project"
            input_dir = project / "input_files"
            paper_dir = project / "paper"
            input_dir.mkdir(parents=True)
            paper_dir.mkdir(parents=True)
            (input_dir / "idea.md").write_text(
                "# First claim\n\nGenerate the first research direction.",
                encoding="utf-8",
            )
            (input_dir / "methods.md").write_text(
                "Initial Denario method.",
                encoding="utf-8",
            )
            (input_dir / "results.md").write_text(
                "Initial Denario output exists.",
                encoding="utf-8",
            )
            (paper_dir / "paper.tex").write_text(
                "\\section{Intro} Initial Denario paper.",
                encoding="utf-8",
            )

            ledger = EpistemicLedger.load(root / "ledger.json")
            decision = ledger.record_decision(
                ActionProposal(
                    claim_id="",
                    claim_title="ledger bootstrap",
                    action_type=ActionType.GENERATE_IDEA,
                    expected_information_gain=1.0,
                    priority="now",
                    reason="Create the first claim.",
                    executor=ActionExecutor.DENARIO,
                    stage="bootstrap",
                    command_hint="Run Denario and import the project.",
                )
            )

            imported = DenarioAdapter.import_project(
                ledger=ledger,
                project_dir=project,
                decision_id=decision.id,
            )

            self.assertIsNotNone(imported["execution"])
            self.assertEqual(imported["execution"]["decision_id"], decision.id)
            self.assertEqual(imported["execution"]["claim_id"], imported["claim"]["id"])
            self.assertGreater(imported["execution"]["artifact_quality"], 0.3)
            self.assertEqual(imported["method_artifact"]["metadata"]["decision_id"], decision.id)
            self.assertEqual(imported["paper_artifacts"][0]["metadata"]["decision_id"], decision.id)
            snapshot = EpistemicLedger.load(root / "ledger.json").claim_snapshot(
                imported["claim"]["id"]
            )
            self.assertEqual(snapshot["metrics"]["execution_count"], 1)
            self.assertEqual(snapshot["executions"][0]["decision_id"], decision.id)
            self.assertEqual(
                snapshot["evidence"][0]["metadata"]["decision_id"],
                decision.id,
            )


if __name__ == "__main__":
    unittest.main()
