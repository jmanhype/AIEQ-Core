from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.adapters.autoresearch import AutoresearchAdapter
from aieq_core.ledger import EpistemicLedger
from aieq_core.models import ActionExecutor, ActionProposal, ActionType

SAMPLE_RUN_LOG = """Vocab size: 8,192
Model config: {'n_layer': 8}
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
"""

SAMPLE_RESULTS_TSV = """commit\tval_bpb\tmemory_gb\tstatus\tdescription
a1b2c3d\t0.997900\t44.0\tkeep\tbaseline
b2c3d4e\t0.993200\t44.2\tkeep\tincrease LR to 0.04
c3d4e5f\t1.005000\t44.0\tdiscard\tswitch to GeLU activation
d4e5f6g\t0.000000\t0.0\tcrash\tdouble model width (OOM)
"""

ALT_BRANCH_RESULTS_TSV = """commit\tval_bpb\tmemory_gb\tstatus\tdescription
z1\t1.000000\t43.8\tkeep\tbaseline
z2\t0.991500\t44.4\tkeep\ttry grouped attention
z3\t0.992100\t44.5\tdiscard\tincrease width
"""


class AutoresearchAdapterTests(unittest.TestCase):
    def test_parse_run_text(self) -> None:
        run = AutoresearchAdapter.parse_run_text(SAMPLE_RUN_LOG)
        self.assertAlmostEqual(run.val_bpb, 0.9979)
        self.assertEqual(run.depth, 8)
        self.assertEqual(run.num_steps, 953)

    def test_import_run_adds_supporting_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_log = tmp / "run.log"
            run_log.write_text(SAMPLE_RUN_LOG, encoding="utf-8")

            ledger = EpistemicLedger.load(tmp / "ledger.json")
            claim = ledger.add_claim(
                title="Sparse attention improves short-budget pretraining",
                statement="A sparse attention variant should reduce validation bpb.",
                novelty=0.8,
                falsifiability=0.9,
            )

            evidence = AutoresearchAdapter.import_run(
                ledger=ledger,
                claim_id=claim.id,
                run_log_path=run_log,
                commit="abc1234",
                description="Sparse attention pilot",
                baseline_bpb=1.0005,
                status="keep",
            )

            self.assertEqual(evidence.direction.value, "support")
            self.assertEqual(evidence.source_type, "autoresearch")
            self.assertIn("delta_bpb", evidence.metadata)
            snapshot = EpistemicLedger.load(tmp / "ledger.json").claim_snapshot(claim.id)
            self.assertEqual(snapshot["metrics"]["evidence_count"], 1)

    def test_import_crash_maps_to_contradictory_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_log = tmp / "crash.log"
            run_log.write_text("RuntimeError: CUDA out of memory", encoding="utf-8")

            ledger = EpistemicLedger.load(tmp / "ledger.json")
            claim = ledger.add_claim(
                title="Wider model helps within the fixed time budget",
                statement="Doubling width should beat the baseline under the same budget.",
            )

            evidence = AutoresearchAdapter.import_run(
                ledger=ledger,
                claim_id=claim.id,
                run_log_path=run_log,
                description="Wide-model trial",
                status="crash",
            )

            self.assertEqual(evidence.direction.value, "contradict")
            self.assertEqual(evidence.metadata["status"], "crash")

    def test_import_run_closes_recorded_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_log = tmp / "run.log"
            run_log.write_text(SAMPLE_RUN_LOG, encoding="utf-8")

            ledger = EpistemicLedger.load(tmp / "ledger.json")
            claim = ledger.add_claim(
                title="Sparse attention improves short-budget pretraining",
                statement="A sparse attention variant should reduce validation bpb.",
            )
            decision = ledger.record_decision(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.RUN_EXPERIMENT,
                    expected_information_gain=0.8,
                    priority="now",
                    reason="Run the pilot experiment.",
                    executor=ActionExecutor.AUTORESEARCH,
                    stage="experimentation",
                    command_hint="uv run train.py > run.log 2>&1",
                )
            )

            evidence = AutoresearchAdapter.import_run(
                ledger=ledger,
                claim_id=claim.id,
                run_log_path=run_log,
                baseline_bpb=1.0005,
                status="keep",
                decision_id=decision.id,
            )

            self.assertEqual(evidence.metadata["decision_id"], decision.id)
            self.assertIn("execution_id", evidence.metadata)
            snapshot = EpistemicLedger.load(tmp / "ledger.json").claim_snapshot(claim.id)
            self.assertEqual(snapshot["metrics"]["execution_count"], 1)
            self.assertEqual(snapshot["executions"][0]["decision_id"], decision.id)
            self.assertEqual(snapshot["executions"][0]["status"], "succeeded")
            self.assertGreater(snapshot["executions"][0]["runtime_seconds"], 300)
            self.assertGreater(snapshot["executions"][0]["artifact_quality"], 0.5)

    def test_import_results_tsv_summarizes_series_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_tsv = tmp / "results.tsv"
            results_tsv.write_text(SAMPLE_RESULTS_TSV, encoding="utf-8")

            ledger = EpistemicLedger.load(tmp / "ledger.json")
            claim = ledger.add_claim(
                title="Sparse attention improves short-budget pretraining",
                statement="A sparse attention variant should reduce validation bpb.",
                novelty=0.8,
                falsifiability=0.9,
            )

            imported = AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=results_tsv,
                branch="main",
            )

            self.assertAlmostEqual(imported["series"]["best_improvement_bpb"], 0.0047, places=6)
            self.assertEqual(imported["series"]["frontier_improvement_count"], 1)
            self.assertEqual(imported["series"]["stagnation_run_count"], 2)
            self.assertEqual(imported["evidence"]["direction"], "support")

            reimported = AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=results_tsv,
                branch="main",
            )
            self.assertEqual(reimported["evidence"]["id"], imported["evidence"]["id"])

            snapshot = EpistemicLedger.load(tmp / "ledger.json").claim_snapshot(claim.id)
            self.assertEqual(snapshot["metrics"]["evidence_count"], 1)
            self.assertEqual(snapshot["metrics"]["autoresearch_series_run_count"], 4)
            self.assertAlmostEqual(
                snapshot["metrics"]["autoresearch_series_keep_rate"],
                2 / 3,
                places=6,
            )
            self.assertAlmostEqual(
                snapshot["metrics"]["autoresearch_series_best_improvement_bpb"],
                0.0047,
                places=6,
            )
            self.assertEqual(
                snapshot["claim"]["metadata"]["autoresearch"]["results_tsv_path"],
                str(results_tsv),
            )

    def test_import_results_tsv_tracks_multiple_branches_and_preferred_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            main_results = tmp / "main.tsv"
            alt_results = tmp / "radical.tsv"
            main_results.write_text(SAMPLE_RESULTS_TSV, encoding="utf-8")
            alt_results.write_text(ALT_BRANCH_RESULTS_TSV, encoding="utf-8")

            ledger = EpistemicLedger.load(tmp / "ledger.json")
            claim = ledger.add_claim(
                title="Branch-aware sparse attention claim",
                statement="Different experiment lines should be tracked separately.",
                novelty=0.8,
                falsifiability=0.9,
            )

            first = AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=main_results,
                branch="main",
            )
            second = AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=alt_results,
                branch="radical",
            )

            self.assertEqual(first["branch"], "main")
            self.assertEqual(second["branch"], "radical")
            self.assertEqual(second["aggregate_series"]["branch_count"], 2)
            self.assertEqual(second["aggregate_series"]["preferred_branch"], "radical")

            snapshot = EpistemicLedger.load(tmp / "ledger.json").claim_snapshot(claim.id)
            meta = snapshot["claim"]["metadata"]["autoresearch"]
            self.assertEqual(meta["branch"], "radical")
            self.assertEqual(meta["results_tsv_path"], str(alt_results))
            self.assertEqual(sorted(meta["series_by_branch"].keys()), ["main", "radical"])
            self.assertEqual(snapshot["metrics"]["evidence_count"], 2)
            self.assertEqual(snapshot["metrics"]["autoresearch_branch_count"], 2)
            self.assertEqual(snapshot["metrics"]["autoresearch_best_branch"], "radical")
            self.assertEqual(snapshot["metrics"]["autoresearch_total_run_count_all_branches"], 7)
            self.assertAlmostEqual(
                snapshot["metrics"]["autoresearch_series_best_improvement_bpb"],
                0.0085,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
