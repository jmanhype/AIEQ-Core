from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ledger import EpistemicLedger
from .models import HypothesisStatus, InnovationHypothesis, ProtocolDraftStatus, clamp
from .runtime import RuntimeConfig

IGNORE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".turbo",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}

PRIORITY_REPO_FILES = [
    "README.md",
    "readme.md",
    "CLAUDE.md",
    "SPEC.md",
    "ARCHITECTURE.md",
    "docs/notes_of_codes.md",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
]
PRIORITY_REPO_FILE_SET = {entry.lower() for entry in PRIORITY_REPO_FILES}
DIGEST_UPSTREAM_PATTERNS = (
    "based on",
    "forked from",
    "adapted from",
    "built on",
    "derived from",
    "reference:",
    "references:",
)
DOMAIN_SIGNAL_HINTS = (
    "compliance",
    "regulation",
    "regulatory",
    "requirement",
    "control",
    "audit",
    "evidence",
    "legal",
    "bank",
    "banking",
    "pharma",
    "medical",
    "health",
    "retrieval",
    "graph",
    "rag",
    "entity",
    "relation",
    "hierarch",
    "cluster",
    "summary",
)

DIGEST_CONTEXT_HINTS = (
    "prompt",
    "policy",
    "router",
    "delegate",
    "entity",
    "relation",
    "cluster",
    "summary",
    "extract",
    "hierarch",
    "compliance",
    "control",
    "requirement",
    "audit",
    "evidence",
    "graph",
    "rag",
    "retriev",
    "eval",
    "benchmark",
    "metric",
    "notes",
    "architecture",
    "design",
)

HYPOTHESIS_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "input_summary": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "statement": {"type": "string"},
                    "summary": {"type": "string"},
                    "rationale": {"type": "string"},
                    "recommended_mode": {"type": "string"},
                    "target_type": {"type": "string"},
                    "target_title": {"type": "string"},
                    "target_source_strategy": {"type": "string"},
                    "mutable_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "suggested_constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "eval_outline": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "leverage": {"type": "number"},
                    "testability": {"type": "number"},
                    "novelty": {"type": "number"},
                    "strategic_novelty": {"type": "number"},
                    "domain_differentiation": {"type": "number"},
                    "fork_specificity": {"type": "number"},
                    "optimization_readiness": {"type": "number"},
                },
                "required": [
                    "title",
                    "statement",
                    "summary",
                    "rationale",
                    "recommended_mode",
                    "target_type",
                    "target_title",
                    "target_source_strategy",
                    "mutable_fields",
                    "suggested_constraints",
                    "eval_outline",
                    "leverage",
                    "testability",
                    "novelty",
                    "strategic_novelty",
                    "domain_differentiation",
                    "fork_specificity",
                    "optimization_readiness",
                ],
            },
        },
    },
    "required": ["input_summary", "hypotheses"],
}

HYPOTHESIS_SYSTEM_PROMPT = """You are the intake layer for AIEQ-Core.

Your job is to transform arbitrary input into testable innovation hypotheses.

Available downstream modes:
- skill_optimizer: optimize prompt, skill, policy, markdown, or other text artifacts against eval suites
- ml_research: Denario + autoresearch for research/training workflows
- manual: useful hypothesis, but not directly optimizable yet

Rules:
- Prefer hypotheses that can be pressure-tested.
- Name the mutable artifact as concretely as possible.
- Do not invent fake code paths or files.
- If the input is not optimization-ready, still propose the best testable next hypothesis and say what artifact should be extracted.
- Keep hypotheses specific and practical, not visionary fluff.
- If the input looks like a specialized fork or vertical application, prefer hypotheses that improve the differentiated domain capability over generic cost/latency cleanups when both are testable.
- Score domain-defining capability improvements higher than generic prompt shortening, token savings, or style cleanup unless the generic optimization is clearly the main bottleneck.
"""

PROTOCOL_CRITERION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "type": {"type": "string"},
        "values": {
            "type": "array",
            "items": {"type": "string"},
        },
        "value": {"type": "string"},
        "pattern": {"type": "string"},
    },
    "required": ["type", "values", "value", "pattern"],
}

PROTOCOL_CASE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "input": {"type": "string"},
        "criteria": {
            "type": "array",
            "items": PROTOCOL_CRITERION_SCHEMA,
        },
        "notes": {"type": "string"},
    },
    "required": ["id", "input", "criteria", "notes"],
}

PROTOCOL_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string"},
        "recommended_mode": {"type": "string"},
        "summary": {"type": "string"},
        "critic_notes": {"type": "string"},
        "target_spec": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "target_type": {"type": "string"},
                "source_type": {"type": "string"},
                "source_path": {"type": "string"},
                "content": {"type": "string"},
                "extraction_strategy": {"type": "string"},
                "mutable_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "title",
                "target_type",
                "source_type",
                "source_path",
                "content",
                "extraction_strategy",
                "mutable_fields",
            ],
        },
        "eval_plan": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "compatible_target_type": {"type": "string"},
                "scoring_method": {"type": "string"},
                "aggregation": {"type": "string"},
                "pass_threshold": {"type": "number"},
                "repetitions": {"type": "integer"},
                "cases": {
                    "type": "array",
                    "items": PROTOCOL_CASE_SCHEMA,
                },
                "falsification_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "name",
                "compatible_target_type",
                "scoring_method",
                "aggregation",
                "pass_threshold",
                "repetitions",
                "cases",
                "falsification_signals",
            ],
        },
        "baseline_plan": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "description": {"type": "string"},
                "artifact_reference": {"type": "string"},
                "success_metric": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": [
                "description",
                "artifact_reference",
                "success_metric",
                "notes",
            ],
        },
        "blockers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "extraction_confidence": {"type": "number"},
        "eval_confidence": {"type": "number"},
        "execution_readiness": {"type": "number"},
    },
    "required": [
        "status",
        "recommended_mode",
        "summary",
        "critic_notes",
        "target_spec",
        "eval_plan",
        "baseline_plan",
        "blockers",
        "extraction_confidence",
        "eval_confidence",
        "execution_readiness",
    ],
}

PROTOCOL_SYSTEM_PROMPT = """You are the protocol compiler for AIEQ-Core.

Your job is to turn one innovation hypothesis into a grounded, pressure-testable protocol.

Rules:
- You must ground any file selection in the provided artifact candidates.
- Never invent file paths.
- Prefer a narrow mutable artifact over a broad repo-wide target.
- If no artifact can be grounded, return a blocked protocol with explicit blockers.
- Build evals that are concrete enough to execute later, even if they are only v1 approximations.
- The protocol must stop short of execution. Do not assume it will run automatically.
"""


class IntakeError(RuntimeError):
    """Raised when intake generation or materialization cannot proceed."""


@dataclass(slots=True)
class GeneratedHypothesis:
    title: str
    statement: str
    summary: str
    rationale: str
    recommended_mode: str
    target_type: str
    target_title: str
    target_source_strategy: str
    mutable_fields: list[str] = field(default_factory=list)
    suggested_constraints: list[str] = field(default_factory=list)
    eval_outline: list[str] = field(default_factory=list)
    leverage: float = 0.5
    testability: float = 0.5
    novelty: float = 0.5
    strategic_novelty: float = 0.5
    domain_differentiation: float = 0.5
    fork_specificity: float = 0.5
    optimization_readiness: float = 0.5


class OpenAIIntakeClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int = 120,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise IntakeError("OPENAI_API_KEY is required for hypothesis generation.")

    def generate_hypotheses(
        self,
        *,
        input_title: str,
        input_type: str,
        input_summary: str,
        input_content: str,
        count: int,
    ) -> tuple[str, list[GeneratedHypothesis], dict[str, Any]]:
        prompt = build_hypothesis_prompt(
            input_title=input_title,
            input_type=input_type,
            input_summary=input_summary,
            input_content=input_content,
            count=count,
        )
        response = self._post_json(
            "/responses",
            {
                "model": self.model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": HYPOTHESIS_SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "innovation_hypotheses",
                        "schema": HYPOTHESIS_RESPONSE_SCHEMA,
                        "strict": True,
                    }
                },
                "max_output_tokens": 6000,
            },
        )
        output_text = self._extract_output_text(response)
        parsed = self._parse_hypothesis_json(output_text)
        hypotheses = [
            GeneratedHypothesis(
                title=str(item["title"]).strip(),
                statement=str(item["statement"]).strip(),
                summary=str(item["summary"]).strip(),
                rationale=str(item["rationale"]).strip(),
                recommended_mode=str(item["recommended_mode"]).strip(),
                target_type=str(item["target_type"]).strip(),
                target_title=str(item["target_title"]).strip(),
                target_source_strategy=str(item["target_source_strategy"]).strip(),
                mutable_fields=[str(entry).strip() for entry in item["mutable_fields"] if str(entry).strip()],
                suggested_constraints=[
                    str(entry).strip() for entry in item["suggested_constraints"] if str(entry).strip()
                ],
                eval_outline=[str(entry).strip() for entry in item["eval_outline"] if str(entry).strip()],
                leverage=float(item["leverage"]),
                testability=float(item["testability"]),
                novelty=float(item["novelty"]),
                strategic_novelty=float(item["strategic_novelty"]),
                domain_differentiation=float(item["domain_differentiation"]),
                fork_specificity=float(item["fork_specificity"]),
                optimization_readiness=float(item["optimization_readiness"]),
            )
            for item in parsed["hypotheses"]
        ]
        return parsed["input_summary"], hypotheses, response

    def compile_protocol(
        self,
        *,
        input_title: str,
        input_type: str,
        input_summary: str,
        input_content: str,
        hypothesis: InnovationHypothesis,
        artifact_candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = build_protocol_prompt(
            input_title=input_title,
            input_type=input_type,
            input_summary=input_summary,
            input_content=input_content,
            hypothesis=hypothesis,
            artifact_candidates=artifact_candidates,
        )
        response = self._post_json(
            "/responses",
            {
                "model": self.model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": PROTOCOL_SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "protocol_draft",
                        "schema": PROTOCOL_RESPONSE_SCHEMA,
                        "strict": True,
                    }
                },
                "max_output_tokens": 6000,
            },
        )
        output_text = self._extract_output_text(response)
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise IntakeError("Protocol output was not valid JSON.") from exc
        return parsed, response

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise IntakeError(f"OpenAI intake request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise IntakeError(f"OpenAI intake request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise IntakeError("OpenAI intake response was not valid JSON.") from exc

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        top_level = response.get("output_text")
        if isinstance(top_level, str) and top_level.strip():
            return top_level
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                direct = content.get("text")
                if isinstance(direct, str) and direct.strip():
                    return direct
                if isinstance(direct, dict):
                    nested = direct.get("value")
                    if isinstance(nested, str) and nested.strip():
                        return nested
                nested_text = content.get("output_text")
                if isinstance(nested_text, str) and nested_text.strip():
                    return nested_text
        raise IntakeError("OpenAI intake response did not contain output text.")

    @staticmethod
    def _parse_hypothesis_json(output_text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise IntakeError("Hypothesis output was not valid JSON.") from exc
        if not isinstance(parsed.get("input_summary"), str):
            raise IntakeError("Hypothesis output is missing `input_summary`.")
        hypotheses = parsed.get("hypotheses")
        if not isinstance(hypotheses, list):
            raise IntakeError("Hypothesis output is missing `hypotheses`.")
        return {
            "input_summary": parsed["input_summary"].strip(),
            "hypotheses": hypotheses,
        }


class AIEQIntakeService:
    def __init__(self, *, config: RuntimeConfig, client: OpenAIIntakeClient | None = None) -> None:
        self.config = config
        self._client = client

    def generate_hypotheses(
        self,
        *,
        ledger: EpistemicLedger,
        input_id: str,
        count: int = 5,
    ) -> dict[str, Any]:
        item = ledger.get_input(input_id)
        client = self._client or OpenAIIntakeClient(
            api_key=self.config.env.get("OPENAI_API_KEY", ""),
            model=self.config.intake_hypothesis_model,
            timeout_seconds=self.config.intake_timeout_seconds,
            base_url=self.config.env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        input_summary, hypotheses, response = client.generate_hypotheses(
            input_title=item.title,
            input_type=item.input_type,
            input_summary=item.summary,
            input_content=item.content,
            count=max(1, count),
        )
        item.summary = input_summary or item.summary
        item.updated_at = now_utc()
        created: list[InnovationHypothesis] = []
        generation_group = f"gen-{now_utc()}"
        for generated in hypotheses[: max(1, count)]:
            overall_score = score_hypothesis(generated)
            hypothesis = ledger.add_hypothesis(
                input_id=item.id,
                title=generated.title,
                statement=generated.statement,
                summary=generated.summary,
                rationale=generated.rationale,
                recommended_mode=generated.recommended_mode,
                target_type=generated.target_type,
                target_title=generated.target_title,
                target_source_strategy=generated.target_source_strategy,
                mutable_fields=generated.mutable_fields,
                suggested_constraints=generated.suggested_constraints,
                eval_outline=generated.eval_outline,
                leverage=generated.leverage,
                testability=generated.testability,
                novelty=generated.novelty,
                strategic_novelty=generated.strategic_novelty,
                domain_differentiation=generated.domain_differentiation,
                fork_specificity=generated.fork_specificity,
                optimization_readiness=generated.optimization_readiness,
                overall_score=overall_score,
                metadata={
                    "generation_group": generation_group,
                    "model": self.config.intake_hypothesis_model,
                    "response_id": str(response.get("id", "")).strip(),
                },
            )
            created.append(hypothesis)
        ledger.save()
        return {
            "input": item,
            "hypotheses": self.rank_hypotheses(ledger=ledger, input_id=input_id, limit=len(created)),
        }

    def rank_hypotheses(
        self,
        *,
        ledger: EpistemicLedger,
        input_id: str,
        limit: int = 10,
    ) -> list[InnovationHypothesis]:
        hypotheses = ledger.hypotheses_for_input(input_id)
        return sorted(
            hypotheses,
            key=lambda item: (
                item.overall_score,
                item.strategic_novelty,
                item.domain_differentiation,
                item.fork_specificity,
                item.testability,
                item.leverage,
                item.optimization_readiness,
                item.novelty,
            ),
            reverse=True,
        )[: max(1, limit)]

    def compile_protocol(
        self,
        *,
        ledger: EpistemicLedger,
        hypothesis_id: str,
    ) -> dict[str, Any]:
        hypothesis = ledger.get_hypothesis(hypothesis_id)
        item = ledger.get_input(hypothesis.input_id)
        artifact_candidates = extract_artifact_candidates(item=item, hypothesis=hypothesis)
        client = self._client or OpenAIIntakeClient(
            api_key=self.config.env.get("OPENAI_API_KEY", ""),
            model=self.config.intake_hypothesis_model,
            timeout_seconds=self.config.intake_timeout_seconds,
            base_url=self.config.env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        compiled, response = client.compile_protocol(
            input_title=item.title,
            input_type=item.input_type,
            input_summary=item.summary,
            input_content=item.content,
            hypothesis=hypothesis,
            artifact_candidates=artifact_candidates,
        )
        protocol_payload = normalize_protocol_payload(
            item=item,
            hypothesis=hypothesis,
            compiled=compiled,
            artifact_candidates=artifact_candidates,
        )
        protocol = ledger.add_protocol_draft(
            input_id=item.id,
            hypothesis_id=hypothesis.id,
            recommended_mode=protocol_payload["recommended_mode"],
            status=protocol_payload["status"],
            artifact_candidates=protocol_payload["artifact_candidates"],
            target_spec=protocol_payload["target_spec"],
            eval_plan=protocol_payload["eval_plan"],
            baseline_plan=protocol_payload["baseline_plan"],
            blockers=protocol_payload["blockers"],
            extraction_confidence=protocol_payload["extraction_confidence"],
            eval_confidence=protocol_payload["eval_confidence"],
            execution_readiness=protocol_payload["execution_readiness"],
            metadata={
                "model": self.config.intake_hypothesis_model,
                "response_id": str(response.get("id", "")).strip(),
                "summary": str(compiled.get("summary", "")).strip(),
                "critic_notes": str(compiled.get("critic_notes", "")).strip(),
            },
        )
        return {
            "input": item,
            "hypothesis": hypothesis,
            "protocol": protocol,
        }

    def materialize_target(
        self,
        *,
        ledger: EpistemicLedger,
        hypothesis_id: str,
        mode: str = "",
        title: str = "",
        target_type: str = "",
        source_file: str = "",
        content: str = "",
        content_file: str = "",
        constraint_file: str = "",
        mutable_fields: list[str] | None = None,
        claim_only: bool = False,
    ) -> dict[str, Any]:
        hypothesis = ledger.get_hypothesis(hypothesis_id)
        item = ledger.get_input(hypothesis.input_id)
        chosen_mode = (mode or hypothesis.recommended_mode or "manual").strip()
        chosen_title = (title or hypothesis.target_title or hypothesis.title).strip() or hypothesis.title
        chosen_target_type = (target_type or hypothesis.target_type or "prompt_template").strip()
        claim = ledger.add_claim(
            title=chosen_title,
            statement=hypothesis.statement,
            novelty=hypothesis.novelty,
            falsifiability=hypothesis.testability,
            tags=list(dict.fromkeys([*item.tags, "materialized"])),
            metadata={
                "mode": chosen_mode,
                "source_input_id": item.id,
                "hypothesis_id": hypothesis.id,
                "intake": {
                    "input_type": item.input_type,
                    "target_source_strategy": hypothesis.target_source_strategy,
                    "suggested_constraints": hypothesis.suggested_constraints,
                    "eval_outline": hypothesis.eval_outline,
                },
            },
        )
        ledger.link_input_to_claim(input_id=item.id, claim_id=claim.id)
        ledger.link_hypothesis_to_claim(hypothesis_id=hypothesis.id, claim_id=claim.id)
        hypothesis.status = HypothesisStatus.MATERIALIZED
        hypothesis.materialized_claim_id = claim.id
        hypothesis.updated_at = now_utc()

        target_payload = resolve_materialization_payload(
            source_file=source_file,
            content=content,
            content_file=content_file,
            fallback_input=item,
            target_source_strategy=hypothesis.target_source_strategy,
        )
        constraints = load_constraint_payload(constraint_file) if constraint_file else {}
        target = None
        if chosen_mode in {"skill_optimizer", "repo_benchmark"} and not claim_only and target_payload["content"].strip():
            target = ledger.register_target(
                claim_id=claim.id,
                mode=chosen_mode,
                target_type=chosen_target_type,
                title=chosen_title,
                content=target_payload["content"],
                source_type=target_payload["source_type"],
                source_path=target_payload["source_path"],
                mutable_fields=mutable_fields or hypothesis.mutable_fields or ["entire_document"],
                invariant_constraints=constraints,
                metadata={
                    "materialized_from_hypothesis_id": hypothesis.id,
                    "input_id": item.id,
                },
            )
        ledger.save()
        return {
            "claim": claim,
            "target": target,
            "requires_target_registration": target is None and chosen_mode in {"skill_optimizer", "repo_benchmark"},
            "recommended_mode": chosen_mode,
        }

    def materialize_protocol(
        self,
        *,
        ledger: EpistemicLedger,
        protocol_id: str,
        claim_only: bool = False,
    ) -> dict[str, Any]:
        protocol = ledger.get_protocol(protocol_id)
        hypothesis = ledger.get_hypothesis(protocol.hypothesis_id)
        item = ledger.get_input(protocol.input_id)
        chosen_mode = (protocol.recommended_mode or hypothesis.recommended_mode or "manual").strip()
        target_spec = dict(protocol.target_spec)
        eval_plan = dict(protocol.eval_plan)
        claim = ledger.add_claim(
            title=str(target_spec.get("title", "")).strip() or hypothesis.target_title or hypothesis.title,
            statement=hypothesis.statement,
            novelty=hypothesis.novelty,
            falsifiability=hypothesis.testability,
            tags=list(dict.fromkeys([*item.tags, "materialized"])),
            metadata={
                "mode": chosen_mode,
                "source_input_id": item.id,
                "hypothesis_id": hypothesis.id,
                "protocol_id": protocol.id,
                "intake": {
                    "input_type": item.input_type,
                    "target_source_strategy": hypothesis.target_source_strategy,
                    "suggested_constraints": hypothesis.suggested_constraints,
                    "eval_outline": hypothesis.eval_outline,
                    "protocol_blockers": list(protocol.blockers),
                    "baseline_plan": protocol.baseline_plan,
                },
            },
        )
        ledger.link_input_to_claim(input_id=item.id, claim_id=claim.id)
        ledger.link_hypothesis_to_claim(hypothesis_id=hypothesis.id, claim_id=claim.id)
        ledger.link_protocol_to_claim(protocol_id=protocol.id, claim_id=claim.id)

        target = None
        eval_suite = None
        requires_target_registration = False
        requires_eval_registration = False
        if chosen_mode in {"skill_optimizer", "repo_benchmark"} and not claim_only:
            target_payload = resolve_protocol_target_payload(protocol=protocol, fallback_input=item)
            if target_payload["content"].strip():
                target = ledger.register_target(
                    claim_id=claim.id,
                    mode=chosen_mode,
                    target_type=str(target_spec.get("target_type", "")).strip() or hypothesis.target_type or "prompt_template",
                    title=str(target_spec.get("title", "")).strip() or hypothesis.target_title or hypothesis.title,
                    content=target_payload["content"],
                    source_type=target_payload["source_type"],
                    source_path=target_payload["source_path"],
                    mutable_fields=list(target_spec.get("mutable_fields") or hypothesis.mutable_fields or ["entire_document"]),
                    invariant_constraints=dict(target_spec.get("invariant_constraints") or {}),
                    metadata={
                        "materialized_from_protocol_id": protocol.id,
                        "materialized_from_hypothesis_id": hypothesis.id,
                        "input_id": item.id,
                    },
                )
                cases = coerce_eval_cases(eval_plan.get("cases"))
                if cases:
                    eval_suite = ledger.register_eval_suite(
                        claim_id=claim.id,
                        target_id=target.id,
                        name=str(eval_plan.get("name", "")).strip() or "compiled-eval-suite",
                        compatible_target_type=str(
                            eval_plan.get("compatible_target_type", target.target_type)
                        ).strip()
                        or target.target_type,
                        scoring_method=str(eval_plan.get("scoring_method", "binary")).strip() or "binary",
                        aggregation=str(eval_plan.get("aggregation", "average")).strip() or "average",
                        pass_threshold=float(eval_plan.get("pass_threshold", 1.0)),
                        repetitions=max(1, int(eval_plan.get("repetitions", 1))),
                        cases=cases,
                        metadata={
                            "falsification_signals": list(eval_plan.get("falsification_signals", [])),
                            "compiled_from_protocol_id": protocol.id,
                        },
                    )
                else:
                    requires_eval_registration = True
            else:
                requires_target_registration = True

        ledger.save()
        return {
            "claim": claim,
            "target": target,
            "eval_suite": eval_suite,
            "requires_target_registration": requires_target_registration,
            "requires_eval_registration": requires_eval_registration,
            "recommended_mode": chosen_mode,
            "protocol": protocol,
        }


def infer_input_type(*, source_path: str = "", explicit_type: str = "") -> str:
    if explicit_type.strip():
        return explicit_type.strip()
    if not source_path:
        return "text"
    path = Path(source_path).expanduser()
    if path.is_dir():
        return "repo"
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".rst"}:
        return "document"
    if suffix in {".json", ".yaml", ".yml", ".toml"}:
        return "config"
    if suffix in {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go"}:
        return "code"
    return "file"


def load_input_payload(
    *,
    source_path: str = "",
    content: str = "",
    content_file: str = "",
) -> tuple[str, str, str]:
    if content_file:
        path = Path(content_file).expanduser().resolve()
        return path.read_text(encoding="utf-8"), "file", str(path)
    if content:
        return content, "inline", ""
    if source_path:
        path = Path(source_path).expanduser().resolve()
        if path.is_dir():
            return build_directory_digest(path, max_chars=22000), "directory", str(path)
        return path.read_text(encoding="utf-8"), "file", str(path)
    raise IntakeError("Provide --source-path, --content-file, or --content.")


def build_directory_digest(root: Path, *, max_files: int = 160, max_chars: int = 18000) -> str:
    file_list: list[str] = []
    discovered_paths: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in sorted(dirnames)
            if dirname not in IGNORE_DIR_NAMES and not dirname.startswith(".git")
        ]
        rel_root = Path(current_root).relative_to(root)
        for filename in sorted(filenames):
            rel_path = str((rel_root / filename)).strip(".")
            if not rel_path or rel_path.startswith(".git"):
                continue
            file_list.append(rel_path.lstrip("/"))
            discovered_paths.append(Path(current_root) / filename)
            if len(file_list) >= max_files:
                break
        if len(file_list) >= max_files:
            break

    sections = [
        f"Repository path: {root}",
        "",
        "Top-level file digest:",
        "\n".join(f"- {item}" for item in file_list) or "- <no files discovered>",
    ]
    payload = "\n".join(sections)
    repo_context = build_repo_context(root=root, discovered_paths=discovered_paths)
    if repo_context:
        payload += f"\n\nFork/domain context:\n{repo_context}"
    contextual_paths = select_digest_paths(root=root, discovered_paths=discovered_paths)
    if contextual_paths:
        payload += "\n\nContext-rich file excerpts:"
    for path in contextual_paths:
        remaining = max_chars - len(payload)
        if remaining < 500:
            break
        excerpt = build_digest_excerpt(path, root=root, max_chars=min(2200, remaining - 200))
        if not excerpt:
            continue
        payload += f"\n\nFile: {path.relative_to(root)}\n```\n{excerpt}\n```"
    return payload[:max_chars]


def summarize_content(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return ""
    first = stripped.splitlines()[0].strip()
    if len(first) >= 160:
        return first[:157] + "..."
    return first


def score_hypothesis(item: GeneratedHypothesis | InnovationHypothesis) -> float:
    recommended_mode = str(item.recommended_mode).strip()
    mode_bonus = 0.0
    if recommended_mode == "skill_optimizer":
        mode_bonus = 0.04
    elif recommended_mode == "repo_benchmark":
        mode_bonus = 0.04
    elif recommended_mode == "ml_research":
        mode_bonus = 0.02
    return clamp(
        0.23 * float(item.leverage)
        + 0.23 * float(item.testability)
        + 0.14 * float(item.optimization_readiness)
        + 0.10 * float(item.novelty)
        + 0.10 * float(item.strategic_novelty)
        + 0.15 * float(item.domain_differentiation)
        + 0.05 * float(item.fork_specificity)
        + mode_bonus
    )


def build_hypothesis_prompt(
    *,
    input_title: str,
    input_type: str,
    input_summary: str,
    input_content: str,
    count: int,
) -> str:
    excerpt = input_content[:20000]
    domain_context = extract_domain_context_block(input_content) if input_type.strip().lower() == "repo" else ""
    parts = [
        f"Input title:\n{input_title}\n\n",
        f"Input type:\n{input_type}\n\n",
        f"Existing summary:\n{input_summary}\n\n",
        f"Requested hypothesis count:\n{count}\n\n",
    ]
    if domain_context:
        parts.append(f"Domain/fork context:\n{domain_context}\n\n")
    parts.extend(
        [
            "Return hypotheses that could become real AIEQ claims.\n",
            "If the input is a repo, focus on feature or behavior hypotheses grounded in the repo.\n",
            "If the input is a prompt/skill/document, focus on optimization hypotheses around that artifact.\n\n",
            "Scoring guidance:\n",
            "- leverage: user/business impact if the hypothesis succeeds\n",
            "- testability: how concretely the hypothesis can be pressure-tested\n",
            "- novelty: general novelty of the idea\n",
            "- strategic_novelty: how much the idea strengthens a distinctive capability, not just generic cleanup\n",
            "- domain_differentiation: how directly it improves the domain-specific value proposition\n",
            "- fork_specificity: how specific it is to this repo/fork rather than a generic upstream cleanup\n",
            "- optimization_readiness: how ready it is to materialize into a real target and eval plan now\n\n",
            "Repo-specific instructions:\n",
            "- Detect whether the repo looks like a specialized fork, applied vertical, or domain adaptation.\n",
            "- If so, include at least one hypothesis aimed at the differentiated capability, not just cost/latency reduction.\n",
            "- Only prefer generic prompt compression or cleanup when the input strongly suggests that cost or verbosity is the real bottleneck.\n\n",
            f"Input content:\n```\n{excerpt}\n```",
        ]
    )
    return "".join(parts)


def extract_domain_context_block(input_content: str) -> str:
    marker = "Fork/domain context:\n"
    start = input_content.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end = input_content.find("\n\nContext-rich file excerpts:", start)
    if end < 0:
        end = input_content.find("\n\nFile:", start)
    block = input_content[start:end if end >= 0 else None].strip()
    return block[:2500]


def build_repo_context(root: Path, *, discovered_paths: list[Path]) -> str:
    lines: list[str] = []
    upstream_lines = detect_upstream_references(root=root)
    if upstream_lines:
        lines.append("Upstream references:")
        lines.extend(f"- {entry}" for entry in upstream_lines[:4])

    git_context = detect_git_context(root=root)
    tracking_ref = str(git_context.get("tracking_ref", "")).strip()
    if tracking_ref:
        lines.append(f"Tracking ref: {tracking_ref}")
    remotes = list(git_context.get("remotes", []))
    if remotes:
        lines.append("Git remotes:")
        lines.extend(f"- {entry}" for entry in remotes[:4])
    changed_files = list(git_context.get("changed_files", []))
    if changed_files:
        lines.append("Files changed relative to tracking/upstream:")
        lines.extend(f"- {entry}" for entry in changed_files[:10])

    domain_hints = infer_domain_hints(root=root, discovered_paths=discovered_paths)
    if domain_hints:
        lines.append("Domain hints:")
        lines.extend(f"- {entry}" for entry in domain_hints[:8])

    return "\n".join(lines).strip()


def detect_upstream_references(root: Path) -> list[str]:
    references: list[str] = []
    seen: set[str] = set()
    for relative in ("README.md", "readme.md", "README.rst", "readme.rst"):
        path = root / relative
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")[:8000]
        except (UnicodeDecodeError, OSError):
            continue
        for match in re.findall(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", content):
            if match not in seen:
                seen.add(match)
                references.append(match)
        for line in content.splitlines():
            lowered = line.lower()
            if any(pattern in lowered for pattern in DIGEST_UPSTREAM_PATTERNS):
                cleaned = line.strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    references.append(cleaned)
        if references:
            break
    return references


def detect_git_context(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {}
    inside = run_git(root, "rev-parse", "--is-inside-work-tree")
    if inside.strip() != "true":
        return {}

    remotes_output = run_git(root, "remote", "-v")
    remotes: list[str] = []
    seen_remote_lines: set[str] = set()
    for line in remotes_output.splitlines():
        cleaned = line.strip()
        if cleaned and cleaned not in seen_remote_lines and cleaned.endswith("(fetch)"):
            seen_remote_lines.add(cleaned)
            remotes.append(cleaned)

    tracking_ref = run_git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    changed_files: list[str] = []
    if tracking_ref.strip():
        diff_output = run_git(root, "diff", "--name-only", tracking_ref.strip())
        changed_files = [line.strip() for line in diff_output.splitlines() if line.strip()]

    return {
        "remotes": remotes,
        "tracking_ref": tracking_ref.strip(),
        "changed_files": changed_files,
    }


def run_git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def infer_domain_hints(root: Path, *, discovered_paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    hints: list[str] = []
    for path in discovered_paths[:120]:
        rel_path = str(path.relative_to(root)).lower()
        for hint in DOMAIN_SIGNAL_HINTS:
            if hint in rel_path and hint not in seen:
                seen.add(hint)
                hints.append(hint)
    for path in select_digest_paths(root=root, discovered_paths=discovered_paths, max_context_files=6):
        try:
            excerpt = path.read_text(encoding="utf-8")[:1600].lower()
        except (UnicodeDecodeError, OSError):
            continue
        for hint in DOMAIN_SIGNAL_HINTS:
            if hint in excerpt and hint not in seen:
                seen.add(hint)
                hints.append(hint)
    return hints


def score_digest_path(path: Path, *, root: Path) -> float:
    rel_path = str(path.relative_to(root)).lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    score = 0.0
    if rel_path in PRIORITY_REPO_FILE_SET:
        score += 8.0
    if rel_path.startswith("docs/"):
        score += 3.8
    if "/prompts/" in f"/{rel_path}" or "prompt" in rel_path:
        score += 4.5
    if any(token in rel_path for token in ("policy", "router", "delegate", "workflow", "rubric")):
        score += 3.8
    if suffix in {".md", ".txt", ".rst"}:
        score += 3.0
    elif suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        score += 2.5
    elif suffix in {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go"}:
        score += 1.8
    if path.parent == root:
        score += 0.8
    score += sum(0.8 for hint in DIGEST_CONTEXT_HINTS if hint in rel_path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > 180_000:
        score -= 1.5
    elif size > 80_000:
        score -= 0.5
    return score


def select_digest_paths(root: Path, *, discovered_paths: list[Path], max_context_files: int = 10) -> list[Path]:
    scored: list[tuple[float, str, Path]] = []
    for path in discovered_paths:
        if not path.is_file():
            continue
        score = score_digest_path(path, root=root)
        scored.append((score, str(path.relative_to(root)), path))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[Path] = []
    seen: set[str] = set()
    for _, rel_path, path in scored:
        if rel_path in seen:
            continue
        seen.add(rel_path)
        selected.append(path)
        if len(selected) >= max_context_files:
            break
    return selected


def build_digest_excerpt(path: Path, *, root: Path, max_chars: int = 2200) -> str:
    try:
        excerpt = path.read_text(encoding="utf-8")[:max_chars]
    except (UnicodeDecodeError, OSError):
        return ""
    return excerpt.strip()


def build_protocol_prompt(
    *,
    input_title: str,
    input_type: str,
    input_summary: str,
    input_content: str,
    hypothesis: InnovationHypothesis,
    artifact_candidates: list[dict[str, Any]],
) -> str:
    candidate_lines: list[str] = []
    for index, candidate in enumerate(artifact_candidates, start=1):
        candidate_lines.append(
            json.dumps(
                {
                    "rank": index,
                    "title": candidate.get("title", ""),
                    "source_type": candidate.get("source_type", ""),
                    "source_path": candidate.get("source_path", ""),
                    "target_type": candidate.get("target_type", ""),
                    "rationale": candidate.get("rationale", ""),
                    "mutable_fields": candidate.get("mutable_fields", []),
                    "excerpt": candidate.get("excerpt", ""),
                },
                ensure_ascii=True,
            )
        )
    return (
        f"Input title:\n{input_title}\n\n"
        f"Input type:\n{input_type}\n\n"
        f"Input summary:\n{input_summary}\n\n"
        "Chosen hypothesis:\n"
        f"Title: {hypothesis.title}\n"
        f"Statement: {hypothesis.statement}\n"
        f"Summary: {hypothesis.summary}\n"
        f"Rationale: {hypothesis.rationale}\n"
        f"Recommended mode: {hypothesis.recommended_mode}\n"
        f"Target type hint: {hypothesis.target_type}\n"
        f"Target title hint: {hypothesis.target_title}\n"
        f"Mutable fields hint: {json.dumps(hypothesis.mutable_fields)}\n"
        f"Suggested constraints: {json.dumps(hypothesis.suggested_constraints)}\n"
        f"Eval outline: {json.dumps(hypothesis.eval_outline)}\n\n"
        "Grounded artifact candidates:\n"
        + ("\n".join(candidate_lines) if candidate_lines else "<none>")
        + "\n\n"
        "Build a protocol draft that stops at planning. If the candidates are weak or the eval cannot be grounded, block the protocol explicitly.\n\n"
        f"Input content excerpt:\n```\n{input_content[:12000]}\n```"
    )


def resolve_materialization_payload(
    *,
    source_file: str,
    content: str,
    content_file: str,
    fallback_input: Any,
    target_source_strategy: str,
) -> dict[str, str]:
    if content_file:
        path = Path(content_file).expanduser().resolve()
        return {"content": path.read_text(encoding="utf-8"), "source_type": "file", "source_path": str(path)}
    if content:
        return {"content": content, "source_type": "inline", "source_path": ""}
    if source_file:
        path = Path(source_file).expanduser().resolve()
        return {"content": path.read_text(encoding="utf-8"), "source_type": "file", "source_path": str(path)}
    if (
        target_source_strategy in {"input_content", "same_file", "direct_input"}
        and fallback_input.source_path
        and Path(fallback_input.source_path).is_file()
    ):
        path = Path(fallback_input.source_path)
        return {"content": path.read_text(encoding="utf-8"), "source_type": "file", "source_path": str(path)}
    if target_source_strategy in {"input_content", "inline"} and fallback_input.content.strip():
        return {"content": fallback_input.content, "source_type": fallback_input.source_type, "source_path": fallback_input.source_path}
    return {"content": "", "source_type": "", "source_path": ""}


def resolve_protocol_target_payload(
    *,
    protocol: Any,
    fallback_input: Any,
) -> dict[str, str]:
    target_spec = dict(protocol.target_spec or {})
    source_type = str(target_spec.get("source_type", "")).strip()
    source_path = str(target_spec.get("source_path", "")).strip()
    content = str(target_spec.get("content", ""))
    if source_type == "file" and source_path:
        path = Path(source_path).expanduser().resolve()
        if path.exists() and path.is_file():
            return {
                "content": path.read_text(encoding="utf-8"),
                "source_type": "file",
                "source_path": str(path),
            }
    if content.strip():
        return {
            "content": content,
            "source_type": source_type or "inline",
            "source_path": source_path,
        }
    if fallback_input.source_path and Path(fallback_input.source_path).is_file():
        path = Path(fallback_input.source_path).expanduser().resolve()
        return {
            "content": path.read_text(encoding="utf-8"),
            "source_type": "file",
            "source_path": str(path),
        }
    if fallback_input.content.strip():
        return {
            "content": fallback_input.content,
            "source_type": fallback_input.source_type,
            "source_path": fallback_input.source_path,
        }
    return {"content": "", "source_type": "", "source_path": ""}


def tokenize_keywords(*values: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in re.findall(r"[a-z0-9_/-]{3,}", value.lower()):
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def infer_target_type_from_path(path: Path, *, fallback: str = "") -> str:
    if fallback.strip():
        return fallback.strip()
    suffix = path.suffix.lower()
    name = path.name.lower()
    if name in {"skill.md"} or "prompt" in name or "policy" in name:
        return "prompt_template"
    if suffix in {".md", ".txt", ".rst"}:
        return "document"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "config"
    if suffix in {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go"}:
        return "code_file"
    return "artifact"


def build_candidate_excerpt(path: Path, *, max_chars: int = 400) -> str:
    try:
        excerpt = path.read_text(encoding="utf-8")[:max_chars]
    except (UnicodeDecodeError, OSError):
        return ""
    return excerpt.strip()


def score_candidate_path(path: Path, *, root: Path, keywords: list[str], preferred_mode: str) -> float:
    rel_path = str(path.relative_to(root)).lower()
    name = path.name.lower()
    suffix = path.suffix.lower()
    score = 0.0
    if path.name in PRIORITY_REPO_FILES:
        score += 4.0
    if suffix in {".md", ".txt", ".rst"}:
        score += 3.0
    elif suffix in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        score += 2.8
    elif suffix in {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go"}:
        score += 1.8
    if preferred_mode == "skill_optimizer" and suffix in {".md", ".txt", ".json", ".yaml", ".yml", ".toml"}:
        score += 1.2
    for keyword in keywords:
        if keyword in rel_path:
            score += 1.0
        elif keyword in name:
            score += 0.8
    for hint in ("prompt", "skill", "policy", "router", "delegate", "config", "settings", "rule"):
        if hint in rel_path:
            score += 1.1
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > 120_000:
        score -= 1.0
    return score


def extract_artifact_candidates(
    *,
    item: Any,
    hypothesis: InnovationHypothesis,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    keywords = tokenize_keywords(
        hypothesis.title,
        hypothesis.statement,
        hypothesis.summary,
        hypothesis.target_title,
        hypothesis.target_type,
        " ".join(hypothesis.mutable_fields),
        " ".join(hypothesis.suggested_constraints),
    )
    fallback_target_type = hypothesis.target_type or ""
    source_path = str(item.source_path or "").strip()
    if source_path:
        path = Path(source_path).expanduser().resolve()
        if path.is_file():
            return [
                {
                    "title": hypothesis.target_title or item.title or path.name,
                    "source_type": "file",
                    "source_path": str(path),
                    "target_type": infer_target_type_from_path(path, fallback=fallback_target_type),
                    "rationale": "Use the provided input file directly as the mutable artifact.",
                    "mutable_fields": hypothesis.mutable_fields or ["entire_document"],
                    "excerpt": build_candidate_excerpt(path),
                }
            ]
        if path.is_dir():
            scored: list[tuple[float, Path]] = []
            for current_root, dirnames, filenames in os.walk(path):
                dirnames[:] = [
                    dirname
                    for dirname in sorted(dirnames)
                    if dirname not in IGNORE_DIR_NAMES and not dirname.startswith(".git")
                ]
                for filename in sorted(filenames):
                    candidate = Path(current_root) / filename
                    if not candidate.is_file():
                        continue
                    scored.append(
                        (
                            score_candidate_path(
                                candidate,
                                root=path,
                                keywords=keywords,
                                preferred_mode=hypothesis.recommended_mode,
                            ),
                            candidate,
                        )
                    )
            scored.sort(key=lambda item: item[0], reverse=True)
            candidates: list[dict[str, Any]] = []
            for score, candidate in scored[: max_candidates]:
                candidates.append(
                    {
                        "title": candidate.name,
                        "source_type": "file",
                        "source_path": str(candidate),
                        "target_type": infer_target_type_from_path(candidate, fallback=fallback_target_type),
                        "rationale": f"Grounded repo candidate selected heuristically (score {score:.2f}).",
                        "mutable_fields": hypothesis.mutable_fields or ["entire_document"],
                        "excerpt": build_candidate_excerpt(candidate),
                    }
                )
            if candidates:
                return candidates
    if str(item.content).strip():
        return [
            {
                "title": hypothesis.target_title or item.title or "inline artifact",
                "source_type": item.source_type or "inline",
                "source_path": source_path,
                "target_type": fallback_target_type or infer_target_type_from_source_type(item.input_type),
                "rationale": "Use the full input content as the mutable artifact.",
                "mutable_fields": hypothesis.mutable_fields or ["entire_document"],
                "excerpt": item.content[:400].strip(),
            }
        ]
    return []


def infer_target_type_from_source_type(input_type: str) -> str:
    normalized = input_type.strip().lower()
    if normalized in {"document", "text"}:
        return "document"
    if normalized in {"config"}:
        return "config"
    if normalized in {"code"}:
        return "code_file"
    if normalized in {"repo"}:
        return "artifact"
    return "artifact"


def normalize_protocol_payload(
    *,
    item: Any,
    hypothesis: InnovationHypothesis,
    compiled: dict[str, Any],
    artifact_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_paths = {
        str(candidate.get("source_path", "")).strip()
        for candidate in artifact_candidates
        if str(candidate.get("source_path", "")).strip()
    }
    blockers = [str(entry).strip() for entry in compiled.get("blockers", []) if str(entry).strip()]
    target_spec = dict(compiled.get("target_spec") or {})
    target_spec.setdefault("title", hypothesis.target_title or hypothesis.title)
    target_spec.setdefault("target_type", hypothesis.target_type or infer_target_type_from_source_type(item.input_type))
    target_spec.setdefault("source_type", "inline" if item.content.strip() else "")
    target_spec.setdefault("source_path", "")
    target_spec.setdefault("content", "")
    target_spec.setdefault("extraction_strategy", "compiled_protocol")
    target_spec.setdefault("mutable_fields", hypothesis.mutable_fields or ["entire_document"])
    target_spec.setdefault("invariant_constraints", {})

    source_path = str(target_spec.get("source_path", "")).strip()
    source_type = str(target_spec.get("source_type", "")).strip()
    if source_path and candidate_paths and source_path not in candidate_paths:
        blockers.append("Compiled protocol selected a source_path outside the grounded candidate set.")
        source_path = ""
    if source_type == "file" and source_path:
        path = Path(source_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            blockers.append(f"Compiled protocol selected a missing file target: {source_path}.")
            source_path = ""
    target_spec["source_path"] = source_path
    if source_type == "inline" and not str(target_spec.get("content", "")).strip() and item.content.strip():
        target_spec["content"] = item.content

    eval_plan = dict(compiled.get("eval_plan") or {})
    eval_plan.setdefault("name", f"{hypothesis.title} eval")
    eval_plan.setdefault("compatible_target_type", str(target_spec.get("target_type", "")).strip() or "artifact")
    eval_plan.setdefault("scoring_method", "binary")
    eval_plan.setdefault("aggregation", "average")
    eval_plan["pass_threshold"] = float(eval_plan.get("pass_threshold", 0.8))
    eval_plan["repetitions"] = max(1, int(eval_plan.get("repetitions", 1)))
    eval_plan["cases"] = coerce_eval_cases(eval_plan.get("cases"))
    eval_plan["falsification_signals"] = [
        str(entry).strip()
        for entry in eval_plan.get("falsification_signals", [])
        if str(entry).strip()
    ]

    baseline_plan = dict(compiled.get("baseline_plan") or {})
    baseline_plan.setdefault("description", "Use the current artifact as the baseline.")
    baseline_plan.setdefault("artifact_reference", source_path or item.source_path or "inline_input")
    baseline_plan.setdefault("success_metric", "Improve aggregate eval score over the current baseline.")
    baseline_plan.setdefault("notes", "")

    extraction_confidence = clamp(float(compiled.get("extraction_confidence", 0.0)))
    eval_confidence = clamp(float(compiled.get("eval_confidence", 0.0)))
    raw_readiness = clamp(float(compiled.get("execution_readiness", 0.0)))

    if not artifact_candidates:
        blockers.append("No grounded artifact candidates were found for this input.")
    if not (str(target_spec.get("content", "")).strip() or source_path):
        blockers.append("Protocol could not ground a concrete mutable artifact.")
    if not eval_plan["cases"]:
        blockers.append("Protocol could not compile any executable eval cases.")

    readiness_penalty = min(0.45, 0.12 * len(blockers))
    execution_readiness = clamp(min(raw_readiness, extraction_confidence, eval_confidence) - readiness_penalty)
    recommended_mode = str(compiled.get("recommended_mode", "")).strip() or hypothesis.recommended_mode or "manual"
    if protocol_prefers_repo_benchmark(
        item=item,
        target_spec=target_spec,
        eval_plan=eval_plan,
        baseline_plan=baseline_plan,
    ):
        recommended_mode = "repo_benchmark"
    status_hint = str(compiled.get("status", "")).strip().lower()
    status = ProtocolDraftStatus.READY
    if blockers or status_hint == ProtocolDraftStatus.BLOCKED.value or execution_readiness < 0.5:
        status = ProtocolDraftStatus.BLOCKED
    elif status_hint == ProtocolDraftStatus.DRAFT.value:
        status = ProtocolDraftStatus.DRAFT

    return {
        "recommended_mode": recommended_mode,
        "status": status,
        "artifact_candidates": artifact_candidates,
        "target_spec": target_spec,
        "eval_plan": eval_plan,
        "baseline_plan": baseline_plan,
        "blockers": list(dict.fromkeys(blockers)),
        "extraction_confidence": extraction_confidence,
        "eval_confidence": eval_confidence,
        "execution_readiness": execution_readiness,
    }


def protocol_prefers_repo_benchmark(
    *,
    item: Any,
    target_spec: dict[str, Any],
    eval_plan: dict[str, Any],
    baseline_plan: dict[str, Any],
) -> bool:
    if str(getattr(item, "input_type", "")).strip().lower() != "repo":
        return False
    haystacks = [
        json.dumps(target_spec, ensure_ascii=True, sort_keys=True),
        json.dumps(eval_plan, ensure_ascii=True, sort_keys=True),
        json.dumps(baseline_plan, ensure_ascii=True, sort_keys=True),
    ]
    return any(
        re.search(r"[A-Za-z0-9_./-]*(?:eval|bench|script|tests?)/[A-Za-z0-9_.-]+\.py", text) is not None
        or (
            re.search(r"[A-Za-z0-9_./-]+\.py", text) is not None
            and any(keyword in text.lower() for keyword in ("token", "accuracy", "benchmark", "eval"))
        )
        for text in haystacks
    )


def coerce_eval_cases(raw_cases: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_cases, list):
        return []
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(raw_cases, start=1):
        if not isinstance(case, dict):
            continue
        candidate = dict(case)
        candidate.setdefault("id", f"case-{index}")
        candidate.setdefault("input", "")
        candidate.setdefault("criteria", [])
        if not str(candidate["input"]).strip():
            continue
        if not isinstance(candidate["criteria"], list):
            candidate["criteria"] = []
        cases.append(candidate)
    return cases


def load_constraint_payload(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def now_utc() -> str:
    from .models import utc_now

    return utc_now()
