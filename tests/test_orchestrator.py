from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.ledger import EpistemicLedger
from aieq_core.method_bridge import MethodBridgeDraft, MethodBridgeReview
from aieq_core.orchestrator import CommandResult, ExternalCommand, ResearchOrchestrator
from aieq_core.runtime import RuntimeConfig


def make_runtime_config(
    root: Path,
    *,
    google_key: bool = False,
    openai_key: bool = False,
    remote_host: str = "",
    remote_repo: str = "",
) -> RuntimeConfig:
    autoresearch_repo = root / "external" / "autoresearch"
    denario_repo = root / "external" / "denario"
    for repo in (autoresearch_repo, denario_repo):
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (autoresearch_repo / "train.py").write_text("print('train')\n", encoding="utf-8")

    env = {}
    if google_key:
        env["GOOGLE_API_KEY"] = "test-key"
    if openai_key:
        env["OPENAI_API_KEY"] = "test-openai-key"

    return RuntimeConfig(
        repo_root=root,
        env_file=None,
        env=env,
        runtime_dir=root / ".aieq-runtime",
        denario_projects_dir=root / ".aieq-runtime" / "denario",
        autoresearch_output_dir=root / ".aieq-runtime" / "autoresearch",
        autoresearch_repo=autoresearch_repo,
        denario_repo=denario_repo,
        autoresearch_remote_host=remote_host,
        autoresearch_remote_repo=remote_repo,
        default_autoresearch_branch="main",
        autoresearch_timeout_seconds=600,
        denario_timeout_seconds=1800,
        denario_mode="fast",
        denario_idea_llm="gemini-2.0-flash",
        denario_method_llm="gemini-2.0-flash",
        denario_paper_llm="gemini-2.5-flash",
        denario_paper_journal="NONE",
        default_data_description_file="",
        method_bridge_enabled=True,
        method_bridge_model="gpt-4.1",
        method_bridge_timeout_seconds=120,
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

    def test_run_next_executes_autoresearch_over_ssh_when_remote_worker_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_runtime_config(
                root,
                remote_host="3090",
                remote_repo="/home/straughter/autoresearch",
            )
            ledger_path = root / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Remote sparse attention experiment",
                statement="A remote GPU worker should execute the next experiment.",
                novelty=0.8,
                falsifiability=0.9,
            )

            def fake_runner(command: ExternalCommand) -> CommandResult:
                self.assertEqual(command.args[0], "ssh")
                self.assertIn("3090", command.args)
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

            with patch.object(
                ResearchOrchestrator,
                "doctor",
                return_value={"capabilities": {"run_experiment": {"available": True}}},
            ), patch.object(
                ResearchOrchestrator,
                "_git_short_revision_remote",
                return_value="remote123",
            ):
                orchestrator = ResearchOrchestrator(config=config, command_runner=fake_runner)
                payload = orchestrator.run_next(ledger_path)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["result"]["mode"], "remote")
            self.assertEqual(payload["result"]["host"], "3090")
            self.assertEqual(payload["result"]["commit"], "remote123")

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)
            self.assertGreaterEqual(snapshot["metrics"]["evidence_count"], 2)

    def test_run_next_applies_method_bridge_and_restores_train_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_runtime_config(root, openai_key=True)
            ledger_path = root / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Bridge sparse attention into train.py",
                statement="The generated method should become a concrete train.py mutation.",
                novelty=0.9,
                falsifiability=0.8,
            )
            method_path = root / "method.md"
            method_path.write_text(
                "Reduce attention heads early and ramp them up across training.\n",
                encoding="utf-8",
            )
            ledger.add_artifact(
                claim_id=claim.id,
                kind="method",
                title="Denario method",
                content=method_path.read_text(encoding="utf-8"),
                source_type="denario",
                source_path=str(method_path),
            )

            home = root / "home"
            (home / ".cache" / "autoresearch").mkdir(parents=True, exist_ok=True)
            train_path = config.autoresearch_repo / "train.py"
            original_train = "print('original train')\n"
            bridged_train = "print('bridged train')\n"
            train_path.write_text(original_train, encoding="utf-8")

            def fake_runner(command: ExternalCommand) -> CommandResult:
                self.assertEqual(train_path.read_text(encoding="utf-8"), bridged_train)
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
            ), patch.object(
                ResearchOrchestrator,
                "_build_method_bridge",
                return_value=MethodBridgeDraft(
                    model="gpt-4.1",
                    prompt="bridge prompt",
                    summary="Applied sparse-attention schedule.",
                    train_py=bridged_train,
                    response_id="resp_test",
                    usage={"input_tokens": 10, "output_tokens": 20},
                    raw_response={"id": "resp_test"},
                ),
            ), patch.object(
                ResearchOrchestrator,
                "_review_method_bridge",
                return_value=MethodBridgeReview(
                    model="gpt-4.1",
                    prompt="review prompt",
                    approved=True,
                    summary="Bridge looks executable.",
                    blockers=[],
                    warnings=["Preserved existing summary labels."],
                    response_id="resp_review",
                    usage={"input_tokens": 8, "output_tokens": 12},
                    raw_response={"id": "resp_review"},
                ),
            ):
                orchestrator = ResearchOrchestrator(config=config, command_runner=fake_runner)
                payload = orchestrator.run_next(ledger_path)

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["result"]["action"], "run_experiment")
            self.assertEqual(train_path.read_text(encoding="utf-8"), original_train)
            bridge = payload["result"]["bridge"]
            self.assertTrue(bridge["applied"])
            self.assertEqual(bridge["summary"], "Applied sparse-attention schedule.")
            self.assertTrue(Path(bridge["generated_train_path"]).exists())
            self.assertTrue(Path(bridge["original_train_path"]).exists())
            self.assertTrue(bridge["review"]["approved"])
            self.assertTrue(Path(bridge["review"]["prompt_path"]).exists())

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)
            execution = snapshot["executions"][-1]
            self.assertEqual(
                execution["metadata"]["bridge"]["summary"],
                "Applied sparse-attention schedule.",
            )

    def test_run_next_repairs_runtime_bridge_failure_and_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_runtime_config(root, openai_key=True)
            ledger_path = root / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Repair bridge after runtime crash",
                statement="A runtime crash should trigger one repaired bridge rerun.",
                novelty=0.9,
                falsifiability=0.8,
            )
            method_path = root / "method.md"
            method_path.write_text("Implement adaptive learning rates.\n", encoding="utf-8")
            ledger.add_artifact(
                claim_id=claim.id,
                kind="method",
                title="Denario method",
                content=method_path.read_text(encoding="utf-8"),
                source_type="denario",
                source_path=str(method_path),
            )

            home = root / "home"
            (home / ".cache" / "autoresearch").mkdir(parents=True, exist_ok=True)
            train_path = config.autoresearch_repo / "train.py"
            original_train = "print('original train')\n"
            broken_train = "print('broken bridge')\n"
            repaired_train = "print('repaired bridge')\n"
            train_path.write_text(original_train, encoding="utf-8")
            call_count = 0

            def fake_runner(command: ExternalCommand) -> CommandResult:
                nonlocal call_count
                call_count += 1
                log_path = Path(command.stdout_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if call_count == 1:
                    self.assertEqual(train_path.read_text(encoding="utf-8"), broken_train)
                    log_path.write_text(
                        "Traceback (most recent call last):\n"
                        "  File \"train.py\", line 10, in <module>\n"
                        "NameError: name 'oops' is not defined\n",
                        encoding="utf-8",
                    )
                    return CommandResult(returncode=1)

                self.assertEqual(train_path.read_text(encoding="utf-8"), repaired_train)
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
            ), patch.object(
                ResearchOrchestrator,
                "_build_method_bridge",
                return_value=MethodBridgeDraft(
                    model="gpt-4.1",
                    prompt="initial bridge prompt",
                    summary="Initial bridge.",
                    train_py=broken_train,
                    response_id="resp_initial",
                    usage={"input_tokens": 10, "output_tokens": 20},
                    raw_response={"id": "resp_initial"},
                ),
            ), patch.object(
                ResearchOrchestrator,
                "_review_method_bridge",
                return_value=MethodBridgeReview(
                    model="gpt-4.1",
                    prompt="review prompt",
                    approved=True,
                    summary="Initial bridge is safe to execute.",
                    blockers=[],
                    warnings=[],
                    response_id="resp_review",
                    usage={"input_tokens": 8, "output_tokens": 12},
                    raw_response={"id": "resp_review"},
                ),
            ), patch.object(
                ResearchOrchestrator,
                "_repair_method_bridge_after_runtime_failure",
                return_value=MethodBridgeDraft(
                    model="gpt-4.1",
                    prompt="runtime repair prompt",
                    summary="Repaired bridge.",
                    train_py=repaired_train,
                    response_id="resp_repair",
                    usage={"input_tokens": 15, "output_tokens": 25},
                    raw_response={"id": "resp_repair"},
                ),
            ):
                orchestrator = ResearchOrchestrator(config=config, command_runner=fake_runner)
                payload = orchestrator.run_next(ledger_path)

            self.assertTrue(payload["ok"])
            self.assertEqual(call_count, 2)
            self.assertEqual(train_path.read_text(encoding="utf-8"), original_train)
            bridge = payload["result"]["bridge"]
            self.assertEqual(bridge["attempt_count"], 2)
            self.assertTrue(bridge["runtime_repair_applied"])
            self.assertEqual(bridge["summary"], "Repaired bridge.")
            self.assertEqual(bridge["attempts"][0]["repair_source"], "initial")
            self.assertEqual(bridge["attempts"][1]["repair_source"], "runtime_failure")
            self.assertTrue(Path(bridge["attempts"][1]["runtime_error_path"]).exists())

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)
            execution = snapshot["executions"][-1]
            self.assertEqual(execution["metadata"]["bridge"]["attempt_count"], 2)

    def test_run_next_skips_execution_when_bridge_review_rejects_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = make_runtime_config(root, openai_key=True)
            ledger_path = root / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            claim = ledger.add_claim(
                title="Reject unsafe bridge before launch",
                statement="Unsafe train.py drafts should be blocked before execution.",
                novelty=0.9,
                falsifiability=0.8,
            )
            method_path = root / "method.md"
            method_path.write_text("Implement risky adaptive scaling.\n", encoding="utf-8")
            ledger.add_artifact(
                claim_id=claim.id,
                kind="method",
                title="Denario method",
                content=method_path.read_text(encoding="utf-8"),
                source_type="denario",
                source_path=str(method_path),
            )

            home = root / "home"
            (home / ".cache" / "autoresearch").mkdir(parents=True, exist_ok=True)
            train_path = config.autoresearch_repo / "train.py"
            original_train = "print('original train')\n"
            train_path.write_text(original_train, encoding="utf-8")

            def fake_runner(command: ExternalCommand) -> CommandResult:
                raise AssertionError("The GPU runner should not be called when review rejects the bridge.")

            def fake_which(binary: str) -> str | None:
                mapping = {
                    "git": "/usr/bin/git",
                    "nvidia-smi": "/usr/bin/nvidia-smi",
                }
                return mapping.get(binary)

            with patch("pathlib.Path.home", return_value=home), patch(
                "aieq_core.runtime.shutil.which",
                side_effect=fake_which,
            ), patch.object(
                ResearchOrchestrator,
                "_build_method_bridge",
                return_value=MethodBridgeDraft(
                    model="gpt-4.1",
                    prompt="initial bridge prompt",
                    summary="Initial bridge.",
                    train_py="print('unsafe bridge')\n",
                    response_id="resp_initial",
                    usage={"input_tokens": 10, "output_tokens": 20},
                    raw_response={"id": "resp_initial"},
                ),
            ), patch.object(
                ResearchOrchestrator,
                "_review_method_bridge",
                return_value=MethodBridgeReview(
                    model="gpt-4.1",
                    prompt="review prompt",
                    approved=False,
                    summary="Candidate still references undefined symbols.",
                    blockers=["Potential undefined name: dmodel_lr_scale"],
                    warnings=["Large rewrite scope."],
                    response_id="resp_review_reject",
                    usage={"input_tokens": 8, "output_tokens": 12},
                    raw_response={"id": "resp_review_reject"},
                ),
            ):
                orchestrator = ResearchOrchestrator(config=config, command_runner=fake_runner)
                payload = orchestrator.run_next(ledger_path)

            self.assertFalse(payload["ok"])
            self.assertEqual(payload["result"]["blocked"], "bridge_review_rejected")
            self.assertEqual(train_path.read_text(encoding="utf-8"), original_train)
            self.assertFalse(payload["result"]["bridge"]["review"]["approved"])

            snapshot = EpistemicLedger.load(ledger_path).claim_snapshot(claim.id)
            execution = snapshot["executions"][-1]
            self.assertEqual(execution["status"], "skipped")
            self.assertEqual(
                execution["metadata"]["status"],
                "bridge_review_rejected",
            )


if __name__ == "__main__":
    unittest.main()
