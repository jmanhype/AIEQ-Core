from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ledger import EpistemicLedger
from .models import HypothesisStatus, InnovationHypothesis, clamp
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
    "CLAUDE.md",
    "SPEC.md",
    "ARCHITECTURE.md",
    "Cargo.toml",
    "package.json",
    "pyproject.toml",
]

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
                optimization_readiness=float(item["optimization_readiness"]),
            )
            for item in parsed["hypotheses"]
        ]
        return parsed["input_summary"], hypotheses, response

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
                item.testability,
                item.optimization_readiness,
                item.leverage,
                item.novelty,
            ),
            reverse=True,
        )[: max(1, limit)]

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
        if chosen_mode == "skill_optimizer" and not claim_only and target_payload["content"].strip():
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
            "requires_target_registration": target is None and chosen_mode == "skill_optimizer",
            "recommended_mode": chosen_mode,
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
            return build_directory_digest(path), "directory", str(path)
        return path.read_text(encoding="utf-8"), "file", str(path)
    raise IntakeError("Provide --source-path, --content-file, or --content.")


def build_directory_digest(root: Path, *, max_files: int = 160, max_chars: int = 18000) -> str:
    file_list: list[str] = []
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
            if len(file_list) >= max_files:
                break
        if len(file_list) >= max_files:
            break

    priority_paths: list[Path] = []
    for relative in PRIORITY_REPO_FILES:
        candidate = root / relative
        if candidate.exists() and candidate.is_file():
            priority_paths.append(candidate)

    sections = [
        f"Repository path: {root}",
        "",
        "Top-level file digest:",
        "\n".join(f"- {item}" for item in file_list) or "- <no files discovered>",
    ]
    for path in priority_paths[:6]:
        try:
            excerpt = path.read_text(encoding="utf-8")[:3500]
        except UnicodeDecodeError:
            continue
        sections.extend(
            [
                "",
                f"File: {path.relative_to(root)}",
                "```",
                excerpt,
                "```",
            ]
        )
    payload = "\n".join(sections)
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
        mode_bonus = 0.05
    elif recommended_mode == "ml_research":
        mode_bonus = 0.02
    return clamp(
        0.35 * float(item.leverage)
        + 0.30 * float(item.testability)
        + 0.25 * float(item.optimization_readiness)
        + 0.10 * float(item.novelty)
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
    excerpt = input_content[:16000]
    return (
        f"Input title:\n{input_title}\n\n"
        f"Input type:\n{input_type}\n\n"
        f"Existing summary:\n{input_summary}\n\n"
        f"Requested hypothesis count:\n{count}\n\n"
        "Return hypotheses that could become real AIEQ claims.\n"
        "If the input is a repo, focus on feature or behavior hypotheses grounded in the repo.\n"
        "If the input is a prompt/skill/document, focus on optimization hypotheses around that artifact.\n\n"
        f"Input content:\n```\n{excerpt}\n```"
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


def load_constraint_payload(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def now_utc() -> str:
    from .models import utc_now

    return utc_now()
