from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.cli import main
from aieq_core.ledger import EpistemicLedger


class CLIModeTests(unittest.TestCase):
    def test_mode_list_reports_ml_skill_and_repo_benchmark_modes(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = main(["mode", "list"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        mode_names = {item["name"] for item in payload["modes"]}
        self.assertIn("ml_research", mode_names)
        self.assertIn("skill_optimizer", mode_names)
        self.assertIn("repo_benchmark", mode_names)

    def test_target_and_eval_register_commands_create_skill_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger_path = root / "ledger.json"
            suite_path = root / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "name": "support-suite",
                        "compatible_target_type": "prompt_template",
                        "pass_threshold": 0.75,
                        "repetitions": 1,
                        "cases": [
                            {
                                "id": "case-1",
                                "input": "Explain the refund policy.",
                                "criteria": [
                                    {"type": "contains_all", "values": ["refund", "policy"]}
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            target_buffer = io.StringIO()
            with redirect_stdout(target_buffer):
                exit_code = main(
                    [
                        "target",
                        "register",
                        str(ledger_path),
                        "--mode",
                        "skill_optimizer",
                        "--title",
                        "Support skill",
                        "--content",
                        "Answer the user clearly.",
                    ]
                )
            self.assertEqual(exit_code, 0)
            target_payload = json.loads(target_buffer.getvalue())
            claim_id = target_payload["claim"]["id"]
            target_id = target_payload["target"]["id"]

            eval_buffer = io.StringIO()
            with redirect_stdout(eval_buffer):
                exit_code = main(
                    [
                        "eval",
                        "register",
                        str(ledger_path),
                        "--mode",
                        "skill_optimizer",
                        "--claim-id",
                        claim_id,
                        "--target-id",
                        target_id,
                        "--suite-file",
                        str(suite_path),
                    ]
                )
            self.assertEqual(exit_code, 0)

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim_id)
            self.assertEqual(snapshot["claim"]["metadata"]["mode"], "skill_optimizer")
            self.assertEqual(snapshot["metrics"]["target_count"], 1)
            self.assertEqual(snapshot["metrics"]["eval_suite_count"], 1)


if __name__ == "__main__":
    unittest.main()
