from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aieq_core.cli import main
from aieq_core.intake import (
    AIEQIntakeService,
    GeneratedHypothesis,
    build_directory_digest,
    score_hypothesis,
)
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
                    strategic_novelty=0.86,
                    domain_differentiation=0.9,
                    fork_specificity=0.82,
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
                    strategic_novelty=0.12,
                    domain_differentiation=0.08,
                    fork_specificity=0.1,
                    optimization_readiness=0.2,
                ),
            ][:count],
            {"id": "resp_fake_intake"},
        )

    def compile_protocol(
        self,
        *,
        input_title: str,
        input_type: str,
        input_summary: str,
        input_content: str,
        hypothesis,
        artifact_candidates,
    ) -> tuple[dict[str, object], dict[str, object]]:
        if "Vague" in hypothesis.title:
            return (
                {
                    "status": "blocked",
                    "recommended_mode": "manual",
                    "summary": "Too vague to compile safely.",
                    "critic_notes": "No grounded artifact or concrete eval could be identified.",
                    "target_spec": {
                        "title": hypothesis.target_title or hypothesis.title,
                        "target_type": hypothesis.target_type or "artifact",
                        "source_type": "",
                        "source_path": "",
                        "content": "",
                        "extraction_strategy": "manual_extraction_required",
                        "mutable_fields": hypothesis.mutable_fields or ["entire_document"],
                        "invariant_constraints": {},
                    },
                    "eval_plan": {
                        "name": f"{hypothesis.title} eval",
                        "compatible_target_type": hypothesis.target_type or "artifact",
                        "scoring_method": "binary",
                        "aggregation": "average",
                        "pass_threshold": 0.8,
                        "repetitions": 1,
                        "cases": [],
                        "falsification_signals": ["no concrete eval cases"],
                    },
                    "baseline_plan": {
                        "description": "Need a concrete artifact first.",
                        "artifact_reference": "",
                        "success_metric": "N/A",
                        "notes": "Blocked until extraction is defined.",
                    },
                    "blockers": ["Artifact is too vague to optimize."],
                    "extraction_confidence": 0.2,
                    "eval_confidence": 0.2,
                    "execution_readiness": 0.15,
                },
                {"id": "resp_fake_protocol_blocked"},
            )

        if input_type == "repo":
            selected = next(
                (
                    candidate
                    for candidate in artifact_candidates
                    if str(candidate.get("source_path", "")).endswith("routing_policy.md")
                ),
                artifact_candidates[0],
            )
            return (
                {
                    "status": "ready",
                    "recommended_mode": "skill_optimizer",
                    "summary": "Routing policy file can be optimized directly.",
                    "critic_notes": "Good candidate, narrow enough, measurable with target-selection evals.",
                    "target_spec": {
                        "title": "Routing rubric",
                        "target_type": "prompt_template",
                        "source_type": "file",
                        "source_path": selected["source_path"],
                        "content": "",
                        "extraction_strategy": "use_grounded_file",
                        "mutable_fields": ["entire_document"],
                        "invariant_constraints": {"required_phrases": ["delegate"]},
                    },
                    "eval_plan": {
                        "name": "routing-eval",
                        "compatible_target_type": "prompt_template",
                        "scoring_method": "binary",
                        "aggregation": "average",
                        "pass_threshold": 0.85,
                        "repetitions": 1,
                        "cases": [
                            {
                                "id": "gpu-case",
                                "input": "Need CUDA debugging help on the remote GPU box.",
                                "criteria": [
                                    {"type": "contains_all", "values": ["3090"]},
                                ],
                            }
                        ],
                        "falsification_signals": ["routes GPU task to the wrong machine"],
                    },
                    "baseline_plan": {
                        "description": "Use the current routing rubric as the baseline.",
                        "artifact_reference": selected["source_path"],
                        "success_metric": "Increase aggregate routing score.",
                        "notes": "Compare against the current coordinator rubric.",
                    },
                    "blockers": [],
                    "extraction_confidence": 0.9,
                    "eval_confidence": 0.88,
                    "execution_readiness": 0.9,
                },
                {"id": "resp_fake_protocol_repo"},
            )

        return (
            {
                "status": "ready",
                "recommended_mode": "skill_optimizer",
                "summary": "Inline prompt is directly optimizable.",
                "critic_notes": "Whole-document mutation is fine here.",
                "target_spec": {
                    "title": hypothesis.target_title or hypothesis.title,
                    "target_type": hypothesis.target_type or "prompt_template",
                    "source_type": "inline",
                    "source_path": "",
                    "content": input_content,
                    "extraction_strategy": "inline_full_document",
                    "mutable_fields": hypothesis.mutable_fields or ["entire_document"],
                    "invariant_constraints": {"max_chars": 600},
                },
                "eval_plan": {
                    "name": "inline-prompt-eval",
                    "compatible_target_type": hypothesis.target_type or "prompt_template",
                    "scoring_method": "binary",
                    "aggregation": "average",
                    "pass_threshold": 0.8,
                    "repetitions": 1,
                    "cases": [
                        {
                            "id": "brevity",
                            "input": "Summarize the refund policy.",
                            "criteria": [
                                {"type": "contains_all", "values": ["refund", "policy"]},
                            ],
                        }
                    ],
                    "falsification_signals": ["misses required policy terms"],
                },
                "baseline_plan": {
                    "description": "Use the current inline prompt as the baseline.",
                    "artifact_reference": "inline_input",
                    "success_metric": "Improve aggregate pass rate.",
                    "notes": "",
                },
                "blockers": [],
                "extraction_confidence": 0.93,
                "eval_confidence": 0.82,
                "execution_readiness": 0.88,
            },
            {"id": "resp_fake_protocol_inline"},
        )


class IntakeServiceTests(unittest.TestCase):
    def test_score_hypothesis_prefers_domain_specific_capability_over_generic_cleanup(self) -> None:
        domain_specific = GeneratedHypothesis(
            title="Compliance hierarchy extraction",
            statement="Specialize extraction prompts for regulation-to-control chains.",
            summary="Improve the fork's differentiated compliance capability.",
            rationale="This directly sharpens the repo's vertical value.",
            recommended_mode="repo_benchmark",
            target_type="prompt_template",
            target_title="hierarchy prompt",
            target_source_strategy="same_file",
            leverage=0.83,
            testability=0.78,
            novelty=0.7,
            strategic_novelty=0.95,
            domain_differentiation=0.97,
            fork_specificity=0.9,
            optimization_readiness=0.72,
        )
        generic_cleanup = GeneratedHypothesis(
            title="Prompt compression",
            statement="Shorten repeated prompt instructions to save tokens.",
            summary="Reduce inference cost.",
            rationale="Easy generic cleanup with shallow upside.",
            recommended_mode="repo_benchmark",
            target_type="prompt_template",
            target_title="prompt.py",
            target_source_strategy="same_file",
            leverage=0.84,
            testability=0.92,
            novelty=0.55,
            strategic_novelty=0.24,
            domain_differentiation=0.18,
            fork_specificity=0.12,
            optimization_readiness=0.88,
        )

        self.assertGreater(score_hypothesis(domain_specific), score_hypothesis(generic_cleanup))

    def test_build_directory_digest_includes_contextual_docs_and_prompt_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            (repo_root / "docs").mkdir(parents=True)
            (repo_root / "hirag").mkdir()
            (repo_root / "README.md").write_text("# HiRAG fork\n", encoding="utf-8")
            (repo_root / "docs" / "notes_of_codes.md").write_text(
                "summary_clusters builds regulation -> requirement -> control hierarchies.\n",
                encoding="utf-8",
            )
            (repo_root / "hirag" / "prompt.py").write_text(
                "HI_PROMPT = 'Extract regulation, control, evidence relationships.'\n",
                encoding="utf-8",
            )
            (repo_root / "misc.txt").write_text("filler\n", encoding="utf-8")

            digest = build_directory_digest(repo_root, max_chars=6000)

            self.assertIn("docs/notes_of_codes.md", digest)
            self.assertIn("hirag/prompt.py", digest)
            self.assertIn("regulation -> requirement -> control", digest)

    def test_build_directory_digest_includes_fork_context_from_readme_and_git_tracking(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is required for fork-context digest test")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            upstream_bare = root / "upstream.git"
            upstream_work = root / "upstream_work"
            repo_root = root / "repo"

            subprocess.run(["git", "init", "--bare", str(upstream_bare)], check=True, capture_output=True)
            subprocess.run(["git", "clone", str(upstream_bare), str(upstream_work)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(upstream_work), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(upstream_work), "config", "user.name", "Test User"], check=True)
            (upstream_work / "README.md").write_text(
                "Based on https://github.com/microsoft/graphrag\n",
                encoding="utf-8",
            )
            (upstream_work / "hirag").mkdir()
            (upstream_work / "hirag" / "prompt.py").write_text("PROMPT = 'base'\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(upstream_work), "add", "."], check=True)
            subprocess.run(["git", "-C", str(upstream_work), "commit", "-m", "initial"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(upstream_work), "branch", "-M", "main"], check=True)
            subprocess.run(["git", "-C", str(upstream_work), "push", "-u", "origin", "main"], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(upstream_bare), "symbolic-ref", "HEAD", "refs/heads/main"],
                check=True,
                capture_output=True,
            )

            subprocess.run(["git", "clone", str(upstream_bare), str(repo_root)], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(repo_root), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(repo_root), "config", "user.name", "Test User"], check=True)
            subprocess.run(["git", "-C", str(repo_root), "checkout", "-B", "main"], check=True, capture_output=True)
            (repo_root / "hirag" / "prompt.py").write_text("PROMPT = 'specialized compliance prompt'\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo_root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo_root), "commit", "-m", "specialize prompt"], check=True, capture_output=True)

            digest = build_directory_digest(repo_root, max_chars=7000)

            self.assertIn("Fork/domain context:", digest)
            self.assertIn("https://github.com/microsoft/graphrag", digest)
            self.assertIn("Tracking ref:", digest)
            self.assertIn("Files changed relative to tracking/upstream:", digest)
            self.assertIn("hirag/prompt.py", digest)

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
            self.assertGreater(payload["hypotheses"][0].domain_differentiation, payload["hypotheses"][1].domain_differentiation)
            reloaded = EpistemicLedger.load(root / "ledger.json")
            ranked = reloaded.hypotheses_for_input(item.id)
            self.assertEqual(len(ranked), 2)
            self.assertGreater(ranked[0].overall_score, ranked[1].overall_score)

    def test_compile_protocol_for_repo_input_finds_grounded_file_and_eval_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "README.md").write_text("# OpenSquirrel\n", encoding="utf-8")
            (repo_root / "routing_policy.md").write_text(
                "delegate GPU tasks to the 3090 machine\n",
                encoding="utf-8",
            )
            ledger = EpistemicLedger.load(root / "ledger.json")
            item = ledger.register_input(
                title="OpenSquirrel",
                input_type="repo",
                content="Repository path digest for OpenSquirrel.",
                source_type="directory",
                source_path=str(repo_root),
                summary="repo",
            )
            hypothesis = ledger.add_hypothesis(
                input_id=item.id,
                title="Remote target auto-router",
                statement="A routing rubric can improve local vs remote target selection.",
                summary="Optimize machine routing decisions.",
                rationale="Machine routing is a core operator action.",
                recommended_mode="skill_optimizer",
                target_type="prompt_template",
                target_title="Routing rubric",
                target_source_strategy="same_file",
                mutable_fields=["entire_document"],
                suggested_constraints=["Keep delegate block valid JSON."],
                eval_outline=["score correct target selection for benchmark tasks"],
                leverage=0.92,
                testability=0.9,
                novelty=0.7,
                optimization_readiness=0.88,
                overall_score=0.9,
            )

            service = AIEQIntakeService(config=make_runtime_config(root), client=FakeIntakeClient())
            payload = service.compile_protocol(ledger=ledger, hypothesis_id=hypothesis.id)

            protocol = payload["protocol"]
            self.assertEqual(protocol.status.value, "ready")
            self.assertGreaterEqual(len(protocol.artifact_candidates), 1)
            self.assertTrue(protocol.target_spec["source_path"].endswith("routing_policy.md"))
            self.assertEqual(protocol.eval_plan["name"], "routing-eval")
            self.assertEqual(len(protocol.eval_plan["cases"]), 1)

            reloaded = EpistemicLedger.load(root / "ledger.json")
            stored = reloaded.get_protocol(protocol.id)
            self.assertEqual(stored.target_spec["source_path"], protocol.target_spec["source_path"])

    def test_compile_protocol_for_inline_document_creates_inline_target_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = EpistemicLedger.load(root / "ledger.json")
            item = ledger.register_input(
                title="Support prompt",
                input_type="document",
                content="Answer support questions briefly and mention the refund policy when relevant.",
                source_type="inline",
                summary="support prompt",
            )
            hypothesis = ledger.add_hypothesis(
                input_id=item.id,
                title="Optimize support brevity",
                statement="A narrower support prompt can improve concise answers.",
                recommended_mode="skill_optimizer",
                target_type="prompt_template",
                target_title="Support prompt",
                target_source_strategy="inline",
                mutable_fields=["entire_document"],
                suggested_constraints=["Keep answers concise."],
                eval_outline=["binary brevity suite"],
                leverage=0.82,
                testability=0.88,
                novelty=0.55,
                optimization_readiness=0.9,
                overall_score=0.84,
            )

            service = AIEQIntakeService(config=make_runtime_config(root), client=FakeIntakeClient())
            payload = service.compile_protocol(ledger=ledger, hypothesis_id=hypothesis.id)
            protocol = payload["protocol"]

            self.assertEqual(protocol.status.value, "ready")
            self.assertEqual(protocol.target_spec["source_type"], "inline")
            self.assertIn("refund policy", protocol.target_spec["content"])
            self.assertEqual(protocol.eval_plan["cases"][0]["id"], "brevity")

    def test_compile_protocol_prefers_repo_benchmark_when_eval_plan_mentions_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "prompt.py").write_text("PROMPT = 'long prompt'\n", encoding="utf-8")
            (repo_root / "eval").mkdir()
            (repo_root / "eval" / "measure.py").write_text("print('ok')\n", encoding="utf-8")

            ledger = EpistemicLedger.load(root / "ledger.json")
            item = ledger.register_input(
                title="HiRAG-like repo",
                input_type="repo",
                content="Repository digest",
                source_type="directory",
                source_path=str(repo_root),
                summary="repo",
            )
            hypothesis = ledger.add_hypothesis(
                input_id=item.id,
                title="Compress prompt templates",
                statement="Compress prompt templates without hurting accuracy.",
                recommended_mode="skill_optimizer",
                target_type="file",
                target_title="prompt.py",
                target_source_strategy="same_file",
                mutable_fields=["entire_document"],
                leverage=0.9,
                testability=0.8,
                novelty=0.7,
                optimization_readiness=0.8,
                overall_score=0.85,
            )

            class ScriptFakeClient(FakeIntakeClient):
                def compile_protocol(self, **kwargs):
                    artifact_candidates = kwargs["artifact_candidates"]
                    selected = artifact_candidates[0]
                    return (
                        {
                            "status": "ready",
                            "recommended_mode": "skill_optimizer",
                            "summary": "Run the repo benchmark script against the prompt file.",
                            "critic_notes": "Grounded command uses a real eval script in the repo.",
                            "target_spec": {
                                "title": "prompt.py",
                                "target_type": "file",
                                "source_type": "file",
                                "source_path": selected["source_path"],
                                "content": "",
                                "extraction_strategy": "use_grounded_file",
                                "mutable_fields": ["entire_document"],
                            },
                            "eval_plan": {
                                "name": "repo-benchmark",
                                "compatible_target_type": "file",
                                "scoring_method": "metric",
                                "aggregation": "average",
                                "pass_threshold": 0.8,
                                "repetitions": 1,
                                "cases": [
                                    {
                                        "id": "candidate",
                                        "input": "Run python eval/measure.py and compare metrics.",
                                        "notes": "Use python eval/measure.py --variant candidate",
                                        "criteria": [],
                                    }
                                ],
                                "falsification_signals": ["python eval/measure.py fails"],
                            },
                            "baseline_plan": {
                                "description": "Use python eval/measure.py --variant baseline",
                                "artifact_reference": selected["source_path"],
                                "success_metric": "Beat the baseline script metrics.",
                                "notes": "",
                            },
                            "blockers": [],
                            "extraction_confidence": 0.85,
                            "eval_confidence": 0.8,
                            "execution_readiness": 0.8,
                        },
                        {"id": "resp_script_protocol"},
                    )

            service = AIEQIntakeService(config=make_runtime_config(root), client=ScriptFakeClient())
            payload = service.compile_protocol(ledger=ledger, hypothesis_id=hypothesis.id)

            self.assertEqual(payload["protocol"].recommended_mode, "repo_benchmark")

    def test_compile_protocol_blocks_vague_raw_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ledger = EpistemicLedger.load(root / "ledger.json")
            item = ledger.register_input(
                title="Loose brainstorm",
                input_type="text",
                content="What if we reinvent collaboration somehow?",
                source_type="inline",
                summary="vague brainstorm",
            )
            hypothesis = ledger.add_hypothesis(
                input_id=item.id,
                title="Vague innovation thought",
                statement="Invent something novel for collaboration.",
                recommended_mode="manual",
                target_type="artifact",
                target_title="Collaboration idea",
                target_source_strategy="manual",
                leverage=0.4,
                testability=0.2,
                novelty=0.9,
                optimization_readiness=0.1,
                overall_score=0.32,
            )

            service = AIEQIntakeService(config=make_runtime_config(root), client=FakeIntakeClient())
            payload = service.compile_protocol(ledger=ledger, hypothesis_id=hypothesis.id)
            protocol = payload["protocol"]

            self.assertEqual(protocol.status.value, "blocked")
            self.assertGreater(len(protocol.blockers), 0)
            self.assertEqual(protocol.execution_readiness, 0.0)

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

    def test_materialize_protocol_creates_claim_target_and_eval_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_path = root / "routing-preamble.txt"
            prompt_path.write_text("Delegate remote GPU tasks to the 3090 machine.", encoding="utf-8")
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
            protocol = ledger.add_protocol_draft(
                input_id=item.id,
                hypothesis_id=hypothesis.id,
                recommended_mode="skill_optimizer",
                status="ready",
                artifact_candidates=[
                    {
                        "title": "Routing rubric",
                        "source_type": "file",
                        "source_path": str(prompt_path),
                        "target_type": "prompt_template",
                        "rationale": "Use the file directly.",
                        "mutable_fields": ["entire_document"],
                    }
                ],
                target_spec={
                    "title": "Routing rubric",
                    "target_type": "prompt_template",
                    "source_type": "file",
                    "source_path": str(prompt_path),
                    "content": "",
                    "extraction_strategy": "use_grounded_file",
                    "mutable_fields": ["entire_document"],
                    "invariant_constraints": {"required_phrases": ["Delegate"]},
                },
                eval_plan={
                    "name": "routing-eval",
                    "compatible_target_type": "prompt_template",
                    "scoring_method": "binary",
                    "aggregation": "average",
                    "pass_threshold": 0.85,
                    "repetitions": 1,
                    "cases": [
                        {
                            "id": "gpu-case",
                            "input": "Need CUDA debugging help on the remote GPU box.",
                            "criteria": [{"type": "contains_all", "values": ["3090"]}],
                        }
                    ],
                    "falsification_signals": ["wrong machine routing"],
                },
                baseline_plan={
                    "description": "Use the current routing rubric as the baseline.",
                    "artifact_reference": str(prompt_path),
                    "success_metric": "Improve aggregate routing score.",
                    "notes": "",
                },
                extraction_confidence=0.9,
                eval_confidence=0.88,
                execution_readiness=0.9,
            )

            payload = AIEQIntakeService(config=make_runtime_config(root)).materialize_protocol(
                ledger=ledger,
                protocol_id=protocol.id,
            )

            self.assertIsNotNone(payload["target"])
            self.assertIsNotNone(payload["eval_suite"])
            snapshot = EpistemicLedger.load(root / "ledger.json").claim_snapshot(payload["claim"].id)
            self.assertEqual(snapshot["metrics"]["protocol_count"], 1)
            self.assertEqual(snapshot["metrics"]["target_count"], 1)
            self.assertEqual(snapshot["metrics"]["eval_suite_count"], 1)
            self.assertEqual(snapshot["protocols"][0]["status"], "materialized")


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

    def test_compile_protocol_and_materialize_protocol_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "routing_policy.md").write_text(
                "delegate GPU tasks to the 3090 machine\n",
                encoding="utf-8",
            )
            ledger_path = root / "ledger.json"
            ledger = EpistemicLedger.load(ledger_path)
            item = ledger.register_input(
                title="OpenSquirrel",
                input_type="repo",
                content="Repository digest",
                source_type="directory",
                source_path=str(repo_root),
                summary="repo",
            )
            hypothesis = ledger.add_hypothesis(
                input_id=item.id,
                title="Remote target auto-router",
                statement="A routing rubric can improve local vs remote target selection.",
                recommended_mode="skill_optimizer",
                target_type="prompt_template",
                target_title="Routing rubric",
                target_source_strategy="same_file",
                mutable_fields=["entire_document"],
                leverage=0.92,
                testability=0.9,
                novelty=0.7,
                optimization_readiness=0.88,
                overall_score=0.9,
            )

            from unittest.mock import patch

            compile_buffer = io.StringIO()
            with patch("aieq_core.cli.AIEQIntakeService", autospec=True) as service_cls:
                service = service_cls.return_value
                service.compile_protocol.return_value = {
                    "input": item,
                    "hypothesis": hypothesis,
                    "protocol": ledger.add_protocol_draft(
                        input_id=item.id,
                        hypothesis_id=hypothesis.id,
                        recommended_mode="skill_optimizer",
                        status="ready",
                        artifact_candidates=[
                            {
                                "title": "Routing rubric",
                                "source_type": "file",
                                "source_path": str(repo_root / "routing_policy.md"),
                                "target_type": "prompt_template",
                                "rationale": "Use the file directly.",
                                "mutable_fields": ["entire_document"],
                            }
                        ],
                        target_spec={
                            "title": "Routing rubric",
                            "target_type": "prompt_template",
                            "source_type": "file",
                            "source_path": str(repo_root / "routing_policy.md"),
                            "content": "",
                            "extraction_strategy": "use_grounded_file",
                            "mutable_fields": ["entire_document"],
                            "invariant_constraints": {},
                        },
                        eval_plan={
                            "name": "routing-eval",
                            "compatible_target_type": "prompt_template",
                            "scoring_method": "binary",
                            "aggregation": "average",
                            "pass_threshold": 0.85,
                            "repetitions": 1,
                            "cases": [
                                {
                                    "id": "gpu-case",
                                    "input": "Need CUDA debugging help on the remote GPU box.",
                                    "criteria": [{"type": "contains_all", "values": ["3090"]}],
                                }
                            ],
                            "falsification_signals": ["wrong machine routing"],
                        },
                        baseline_plan={
                            "description": "Use the current routing rubric as the baseline.",
                            "artifact_reference": str(repo_root / "routing_policy.md"),
                            "success_metric": "Improve aggregate routing score.",
                            "notes": "",
                        },
                        extraction_confidence=0.9,
                        eval_confidence=0.88,
                        execution_readiness=0.9,
                    ),
                }
                with redirect_stdout(compile_buffer):
                    exit_code = main(
                        [
                            "compile-protocol",
                            str(ledger_path),
                            "--hypothesis-id",
                            hypothesis.id,
                        ]
                    )
            self.assertEqual(exit_code, 0)
            protocol_payload = json.loads(compile_buffer.getvalue())
            protocol_id = protocol_payload["protocol"]["id"]

            show_buffer = io.StringIO()
            with redirect_stdout(show_buffer):
                exit_code = main(
                    [
                        "protocol",
                        "show",
                        str(ledger_path),
                        "--protocol-id",
                        protocol_id,
                    ]
                )
            self.assertEqual(exit_code, 0)
            shown = json.loads(show_buffer.getvalue())
            self.assertEqual(shown["protocol"]["id"], protocol_id)

            materialize_buffer = io.StringIO()
            with redirect_stdout(materialize_buffer):
                exit_code = main(
                    [
                        "materialize-protocol",
                        str(ledger_path),
                        "--protocol-id",
                        protocol_id,
                    ]
                )
            self.assertEqual(exit_code, 0)
            materialized = json.loads(materialize_buffer.getvalue())
            self.assertEqual(materialized["recommended_mode"], "skill_optimizer")
            self.assertIsNotNone(materialized["target"])
            self.assertIsNotNone(materialized["eval_suite"])


if __name__ == "__main__":
    unittest.main()
