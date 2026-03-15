from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.ledger import EpistemicLedger
from aieq_core.orchestrator import CommandResult, ExternalCommand, ResearchOrchestrator
from aieq_core.runtime import RuntimeConfig


def make_runtime_config(root: Path, *, google_key: bool = False) -> RuntimeConfig:
    autoresearch_repo = root / "external" / "autoresearch"
    denario_repo = root / "external" / "denario"
    for repo in (autoresearch_repo, denario_repo):
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (autoresearch_repo / "train.py").write_text("print('train')\n", encoding="utf-8")

    env = {}
    if google_key:
        env["GOOGLE_API_KEY"] = "test-key"

    return RuntimeConfig(
        repo_root=root,
        env_file=None,
        env=env,
        runtime_dir=root / ".aieq-runtime",
        denario_projects_dir=root / ".aieq-runtime" / "denario",
        autoresearch_output_dir=root / ".aieq-runtime" / "autoresearch",
        autoresearch_repo=autoresearch_repo,
        denario_repo=denario_repo,
        default_autoresearch_branch="main",
        autoresearch_timeout_seconds=600,
        denario_timeout_seconds=1800,
        denario_mode="fast",
        denario_idea_llm="gemini-2.0-flash",
        denario_method_llm="gemini-2.0-flash",
        denario_paper_llm="gemini-2.5-flash",
        denario_paper_journal="NONE",
        default_data_description_file="",
    )


class ResearchOrchestratorTests(unittest.TestCase):
    def test_run_next_bootstraps_denario_project_and_imports_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_runtime_config(root, google_key=True)
            ledger_path = root / "ledger.json"
            EpistemicLedger.load(ledger_path).save()

            def fake_runner(command: ExternalCommand) -> CommandResult:
                spec_path = Path(command.args[-1])
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                project_dir = Path(spec["project_dir"])
                input_dir = project_dir / "input_files"
                input_dir.mkdir(parents=True, exist_ok=True)
                (input_dir / "data_description.md").write_text(
                    "Analyze short-budget training.\n",
                    encoding="utf-8",
                )
                (input_dir / "idea.md").write_text(
                    "# Sparse Attention Hypothesis\nSparse attention should improve validation bpb.\n",
                    encoding="utf-8",
                )
                return CommandResult(returncode=0, stdout="{}", stderr="")

            with patch("aieq_core.runtime.shutil.which", return_value="/usr/bin/git"):
                orchestrator = ResearchOrchestrator(config=config, command_runner=fake_runner)
                payload = orchestrator.run_next(
                    ledger_path,
                    data_description="Analyze short-budget training.",
                )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["result"]["action"], "generate_idea")
            self.assertEqual(
                payload["result"]["imported"]["claim"]["title"],
                "Sparse Attention Hypothesis",
            )
            self.assertEqual(
                payload["follow_up_decision"]["primary_action"]["action_type"],
                "generate_method",
            )

    def test_run_next_executes_autoresearch_and_imports_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_runtime_config(root, google_key=False)
            ledger_path = root / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Sparse attention improves short-budget pretraining",
                statement="A sparse attention variant should reduce validation bpb.",
                novelty=0.8,
                falsifiability=0.9,
            )

            home = root / "home"
            (home / ".cache" / "autoresearch").mkdir(parents=True, exist_ok=True)

            def fake_runner(command: ExternalCommand) -> CommandResult:
                log_path = Path(command.stdout_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(
                    "\n".join(
                        [
                            "---",
                            "val_bpb:          0.997900",
                            "training_seconds: 300.1",
                            "total_seconds:    325.9",
                            "peak_vram_mb:     45060.2",
                            "mfu_percent:      39.80",
                            "total_tokens_M:   499.6",
                            "num_steps:        953",
                            "num_params_M:     50.3",
                            "depth:            8",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return CommandResult(returncode=0)

            def fake_which(binary: str) -> str | None:
                mapping = {
                    "git": "/usr/bin/git",
                    "nvidia-smi": "/usr/bin/nvidia-smi",
                }
                return mapping.get(binary)

            with patch("pathlib.Path.home", return_value=home), patch(
                "aieq_core.runtime.shutil.which",
                side_effect=fake_which,
            ):
                orchestrator = ResearchOrchestrator(config=config, command_runner=fake_runner)
                payload = orchestrator.run_next(ledger_path)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["result"]["action"], "run_experiment")
            self.assertEqual(payload["result"]["status"], "keep")
            self.assertTrue(Path(payload["result"]["run_log_path"]).exists())
            self.assertTrue(Path(payload["result"]["results_tsv_path"]).exists())

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)
            self.assertGreaterEqual(snapshot["metrics"]["evidence_count"], 2)
            self.assertEqual(
                snapshot["claim"]["metadata"]["autoresearch"]["branch"],
                "main",
            )


if __name__ == "__main__":
    unittest.main()
