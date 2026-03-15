from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DemoScriptTests(unittest.TestCase):
    def test_demo_script_runs_end_to_end(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "examples" / "demo" / "run_demo.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir) / "demo-output"
            completed = subprocess.run(
                [sys.executable, str(script), "--workspace", str(workspace)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(completed.stdout)
            snapshot = json.loads((workspace / "final_snapshot.json").read_text(encoding="utf-8"))
            initial_decision = json.loads(
                (workspace / "initial_decision.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["initial_action"], "run_experiment")
            self.assertEqual(summary["initial_executor"], "autoresearch")
            self.assertEqual(summary["followup_evidence_direction"], "support")
            self.assertEqual(summary["final_claim_status"], "supported")
            self.assertEqual(summary["final_next_action"], "synthesize_paper")
            self.assertGreater(snapshot["claim"]["confidence"], 0.7)
            self.assertEqual(
                initial_decision["decision"]["primary_action"]["action_type"],
                "run_experiment",
            )


if __name__ == "__main__":
    unittest.main()
