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
from aieq_core.intake import AIEQIntakeService, GeneratedHypothesis
from aieq_core.ledger import EpistemicLedger
from aieq_core.runtime import RuntimeConfig


def make_runtime_config(root: Path) -> RuntimeConfig:
    autoresearch_repo = root / "external" / "autoresearch"
    denario_repo = root / "external" / "denario"
    for repo in (autoresearch_repo, denario_repo):
        (repo / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (repo / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (autoresearch_repo / "train.py").write_text("print('train')\n", encoding="utf-8")
    return RuntimeConfig(
        repo_root=root,
        env_file=None,
        env={"OPENAI_API_KEY": "test-openai-key"},
        runtime_dir=root / ".aieq-runtime",
        denario_projects_dir=root / ".aieq-runtime" / "denario",
        autoresearch_output_dir=root / ".aieq-runtime" / "autoresearch",
        autoresearch_repo=autoresearch_repo,
        denario_repo=denario_repo,
        autoresearch_remote_host="",
        autoresearch_remote_repo="",
        default_autoresearch_branch="main",
        autoresearch_timeout_seconds=600,
        denario_timeout_seconds=1800,
        denario_mode="fast",
        denario_idea_llm="gpt-5-mini",
        denario_method_llm="gpt-5-mini",
        denario_paper_llm="gpt-5-mini",
        denario_paper_journal="NONE",
        default_data_description_file="",
        method_bridge_enabled=True,
        method_bridge_model="gpt-4.1",
        method_bridge_timeout_seconds=120,
        intake_hypothesis_model="gpt-5-mini",
        intake_timeout_seconds=120,
        skill_mutation_model="gpt-5-mini",
        skill_review_model="gpt-5-mini",
        skill_eval_model="gpt-5-mini",
        skill_timeout_seconds=120,
    )


class FakeIntakeClient:
    def generate_hypotheses(
        self,
        *,
        input_title: str,
        input_type: str,
        input_summary: str,
        input_content: str,
        count: int,
    ) -> tuple[str, list[GeneratedHypothesis], dict[str, object]]:
        self.last_call = {
            "input_title": input_title,
            "input_type": input_type,
            "input_summary": input_summary,
            "input_content": input_content,
            "count": count,
        }
        return (
            "OpenSquirrel control plane focused on delegation and routing.",
            [
                GeneratedHypothesis(
                    title="Remote target auto-router",
                    statement="A routing rubric can improve local vs remote target selection.",
                    summary="Optimize machine routing decisions.",
                    rationale="Machine routing is a core operator action.",
                    recommended_mode="skill_optimizer",
                    target_type="prompt_template",
                    target_title="Routing rubric",
                    target_source_strategy="inline",
                    mutable_fields=["entire_document"],
                    suggested_constraints=["Keep delegate block valid JSON."],
                    eval_outline=["score correct target selection for benchmark tasks"],
                    leverage=0.92,
                    testability=0.9,
                    novelty=0.7,
                    optimization_readiness=0.88,
                ),
                GeneratedHypothesis(
                    title="Transcript theme tweak",
                    statement="A copy polish pass may improve perceived readability.",
                    summary="Tweak wording in transcript rendering.",
                    rationale="Lower leverage and only partially testable.",
                    recommended_mode="manual",
                    target_type="design_note",
                    target_title="Theme copy tweak",
                    target_source_strategy="inline",
                    mutable_fields=["summary"],
                    suggested_constraints=["Keep terminology stable."],
                    eval_outline=["human review only"],
                    leverage=0.35,
                    testability=0.25,
                    novelty=0.4,
                    optimization_readiness=0.2,
                ),
            ][:count],
            {"id": "resp_fake_intake"},
        )


class IntakeServiceTests(unittest.TestCase):
    def test_generate_hypotheses_persists_and_ranks_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = EpistemicLedger.load(root / "ledger.json")
            item = ledger.register_input(
                title="OpenSquirrel",
                input_type="repo",
                content="README for a native multi-agent control plane.",
                source_type="inline",
                summary="multi-agent app",
            )

            service = AIEQIntakeService(
                config=make_runtime_config(root),
                client=FakeIntakeClient(),
            )
            payload = service.generate_hypotheses(ledger=ledger, input_id=item.id, count=2)

            self.assertEqual(payload["input"].summary, "OpenSquirrel control plane focused on delegation and routing.")
            self.assertEqual(len(payload["hypotheses"]), 2)
            self.assertEqual(payload["hypotheses"][0].title, "Remote target auto-router")
            reloaded = EpistemicLedger.load(root / "ledger.json")
            ranked = reloaded.hypotheses_for_input(item.id)
            self.assertEqual(len(ranked), 2)
            self.assertGreater(ranked[0].overall_score, ranked[1].overall_score)

    def test_materialize_target_creates_claim_and_target_for_skill_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_path = root / "routing-preamble.txt"
            prompt_path.write_text("Delegate remote GPU tasks to the best machine.", encoding="utf-8")
            constraints_path = root / "constraints.json"
            constraints_path.write_text(
                json.dumps({"required_phrases": ["Delegate"], "max_chars": 1200}),
                encoding="utf-8",
            )

            ledger = EpistemicLedger.load(root / "ledger.json")
            item = ledger.register_input(
                title="Routing preamble",
                input_type="document",
                content=prompt_path.read_text(encoding="utf-8"),
                source_type="file",
                source_path=str(prompt_path),
                summary="routing preamble",
            )
            hypothesis = ledger.add_hypothesis(
                input_id=item.id,
                title="Optimize routing rubric",
                statement="A tighter routing rubric can improve correct machine selection.",
                recommended_mode="skill_optimizer",
                target_type="prompt_template",
                target_title="Routing rubric",
                target_source_strategy="same_file",
                mutable_fields=["entire_document"],
                suggested_constraints=["Keep delegate JSON valid."],
                eval_outline=["benchmark remote target choices"],
                leverage=0.9,
                testability=0.85,
                novelty=0.65,
                optimization_readiness=0.9,
                overall_score=0.87,
            )

            payload = AIEQIntakeService(config=make_runtime_config(root)).materialize_target(
                ledger=ledger,
                hypothesis_id=hypothesis.id,
                constraint_file=str(constraints_path),
            )

            self.assertIsNotNone(payload["target"])
            self.assertFalse(payload["requires_target_registration"])
            target = payload["target"]
            claim = payload["claim"]
            self.assertEqual(claim.metadata["source_input_id"], item.id)
            self.assertEqual(claim.metadata["hypothesis_id"], hypothesis.id)
            self.assertEqual(target.source_path, str(prompt_path))
            self.assertIn("Delegate remote GPU tasks", target.content)
            self.assertEqual(target.invariant_constraints["max_chars"], 1200)

            snapshot = EpistemicLedger.load(root / "ledger.json").claim_snapshot(claim.id)
            self.assertEqual(snapshot["metrics"]["target_count"], 1)
            self.assertEqual(snapshot["metrics"]["input_count"], 1)
            self.assertEqual(snapshot["metrics"]["hypothesis_count"], 1)
            self.assertEqual(snapshot["hypotheses"][0]["status"], "materialized")
            self.assertEqual(snapshot["hypotheses"][0]["materialized_claim_id"], claim.id)


class IntakeCLITests(unittest.TestCase):
    def test_ingest_register_and_materialize_target_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger_path = root / "ledger.json"

            ingest_buffer = io.StringIO()
            with redirect_stdout(ingest_buffer):
                exit_code = main(
                    [
                        "ingest",
                        "register",
                        str(ledger_path),
                        "--title",
                        "Support prompt",
                        "--input-type",
                        "document",
                        "--content",
                        "Answer support questions briefly and clearly.",
                    ]
                )
            self.assertEqual(exit_code, 0)
            input_payload = json.loads(ingest_buffer.getvalue())
            input_id = input_payload["input"]["id"]

            ledger = EpistemicLedger.load(ledger_path)
            hypothesis = ledger.add_hypothesis(
                input_id=input_id,
                title="Optimize support brevity",
                statement="A narrower support prompt can improve concise answers.",
                recommended_mode="skill_optimizer",
                target_type="prompt_template",
                target_title="Support prompt",
                target_source_strategy="input_content",
                mutable_fields=["entire_document"],
                suggested_constraints=["Keep answers concise."],
                eval_outline=["binary brevity suite"],
                leverage=0.82,
                testability=0.88,
                novelty=0.55,
                optimization_readiness=0.9,
                overall_score=0.84,
            )

            materialize_buffer = io.StringIO()
            with redirect_stdout(materialize_buffer):
                exit_code = main(
                    [
                        "materialize-target",
                        str(ledger_path),
                        "--hypothesis-id",
                        hypothesis.id,
                    ]
                )
            self.assertEqual(exit_code, 0)
            materialized = json.loads(materialize_buffer.getvalue())
            self.assertEqual(materialized["recommended_mode"], "skill_optimizer")
            self.assertFalse(materialized["requires_target_registration"])
            self.assertEqual(materialized["target"]["target_type"], "prompt_template")


if __name__ == "__main__":
    unittest.main()
