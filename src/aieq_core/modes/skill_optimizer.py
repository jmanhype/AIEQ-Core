from __future__ import annotations

import json
import re
import statistics
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import (
    ActionExecutor,
    ActionProposal,
    ActionType,
    EvidenceDirection,
    ExecutionStatus,
    ReviewStatus,
    action_matches,
    clamp,
)
from .base import ModeAdapter


MUTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["summary", "content"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "approved": {"type": "boolean"},
        "summary": {"type": "string"},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["approved", "summary", "blockers", "warnings"],
}

MUTATION_SYSTEM_PROMPT = """You optimize a prompt or skill document against a measurable eval suite.

Return a full replacement content block for the target artifact.

Rules:
- Preserve the artifact's core purpose.
- Improve performance against the eval suite.
- Respect the provided invariant constraints.
- Do not add TODOs, placeholders, or notes to self.
- Keep the output ready to use as the new prompt or skill content.
"""

REVIEW_SYSTEM_PROMPT = """You review a rewritten prompt or skill document before evaluation.

Reject the draft if it violates invariant constraints, obviously drops the original task, or
looks malformed for immediate execution.
"""


@dataclass(slots=True)
class PromptMutationDraft:
    model: str
    prompt: str
    summary: str
    content: str
    response_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptMutationReview:
    model: str
    prompt: str
    approved: bool
    summary: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    response_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptEvalResponse:
    model: str
    prompt: str
    output_text: str
    response_id: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class SkillOptimizerError(RuntimeError):
    """Raised when skill-optimizer execution cannot proceed."""


class OpenAISkillOptimizerClient:
    def __init__(
        self,
        *,
        api_key: str,
        mutation_model: str,
        review_model: str,
        eval_model: str,
        timeout_seconds: int = 120,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.api_key = api_key.strip()
        self.mutation_model = mutation_model.strip()
        self.review_model = review_model.strip()
        self.eval_model = eval_model.strip()
        self.timeout_seconds = timeout_seconds
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            raise SkillOptimizerError("OPENAI_API_KEY is required for skill optimization.")

    def mutate(
        self,
        *,
        target_title: str,
        target_type: str,
        current_content: str,
        invariant_constraints: dict[str, Any],
        suite: dict[str, Any],
    ) -> PromptMutationDraft:
        prompt = build_mutation_prompt(
            target_title=target_title,
            target_type=target_type,
            current_content=current_content,
            invariant_constraints=invariant_constraints,
            suite=suite,
        )
        response = self._post_json(
            "/responses",
            {
                "model": self.mutation_model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": MUTATION_SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "prompt_mutation",
                        "schema": MUTATION_SCHEMA,
                        "strict": True,
                    }
                },
                "max_output_tokens": 8000,
            },
        )
        output_text = self._extract_output_text(response)
        parsed = self._parse_mutation_json(output_text)
        return PromptMutationDraft(
            model=self.mutation_model,
            prompt=prompt,
            summary=parsed["summary"],
            content=parsed["content"],
            response_id=str(response.get("id", "")).strip(),
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            raw_response=response,
        )

    def review(
        self,
        *,
        target_title: str,
        target_type: str,
        current_content: str,
        candidate_content: str,
        invariant_constraints: dict[str, Any],
    ) -> PromptMutationReview:
        prompt = build_review_prompt(
            target_title=target_title,
            target_type=target_type,
            current_content=current_content,
            candidate_content=candidate_content,
            invariant_constraints=invariant_constraints,
        )
        response = self._post_json(
            "/responses",
            {
                "model": self.review_model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": REVIEW_SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "prompt_mutation_review",
                        "schema": REVIEW_SCHEMA,
                        "strict": True,
                    }
                },
                "max_output_tokens": 4000,
            },
        )
        output_text = self._extract_output_text(response)
        parsed = self._parse_review_json(output_text)
        return PromptMutationReview(
            model=self.review_model,
            prompt=prompt,
            approved=parsed["approved"],
            summary=parsed["summary"],
            blockers=parsed["blockers"],
            warnings=parsed["warnings"],
            response_id=str(response.get("id", "")).strip(),
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            raw_response=response,
        )

    def evaluate_case(
        self,
        *,
        candidate_content: str,
        case_input: str,
    ) -> PromptEvalResponse:
        prompt = case_input.strip()
        response = self._post_json(
            "/responses",
            {
                "model": self.eval_model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": candidate_content}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    },
                ],
                "max_output_tokens": 2000,
            },
        )
        output_text = self._extract_output_text(response)
        return PromptEvalResponse(
            model=self.eval_model,
            prompt=prompt,
            output_text=output_text,
            response_id=str(response.get("id", "")).strip(),
            usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
            raw_response=response,
        )

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
            raise SkillOptimizerError(
                f"OpenAI skill-optimizer request failed with HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SkillOptimizerError(f"OpenAI skill-optimizer request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SkillOptimizerError("OpenAI skill-optimizer returned invalid JSON.") from exc

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
        raise SkillOptimizerError("OpenAI skill-optimizer response did not contain output text.")

    @staticmethod
    def _parse_mutation_json(output_text: str) -> dict[str, str]:
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise SkillOptimizerError("Mutation output was not valid JSON.") from exc
        summary = parsed.get("summary")
        content = parsed.get("content")
        if not isinstance(summary, str) or not summary.strip():
            raise SkillOptimizerError("Mutation output is missing `summary`.")
        if not isinstance(content, str) or not content.strip():
            raise SkillOptimizerError("Mutation output is missing `content`.")
        return {"summary": summary.strip(), "content": content}

    @staticmethod
    def _parse_review_json(output_text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise SkillOptimizerError("Review output was not valid JSON.") from exc
        approved = parsed.get("approved")
        summary = parsed.get("summary")
        blockers = parsed.get("blockers")
        warnings = parsed.get("warnings")
        if not isinstance(approved, bool):
            raise SkillOptimizerError("Review output is missing `approved`.")
        if not isinstance(summary, str) or not summary.strip():
            raise SkillOptimizerError("Review output is missing `summary`.")
        if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
            raise SkillOptimizerError("Review output is missing `blockers`.")
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise SkillOptimizerError("Review output is missing `warnings`.")
        return {
            "approved": approved,
            "summary": summary.strip(),
            "blockers": [item.strip() for item in blockers if item.strip()],
            "warnings": [item.strip() for item in warnings if item.strip()],
        }


def build_mutation_prompt(
    *,
    target_title: str,
    target_type: str,
    current_content: str,
    invariant_constraints: dict[str, Any],
    suite: dict[str, Any],
) -> str:
    return (
        f"Target title:\n{target_title}\n\n"
        f"Target type:\n{target_type}\n\n"
        f"Invariant constraints:\n{json.dumps(invariant_constraints, indent=2, sort_keys=True)}\n\n"
        f"Eval suite:\n{json.dumps(suite, indent=2, sort_keys=True)}\n\n"
        f"Current content:\n```\n{current_content}\n```\n\n"
        "Return an improved replacement content block for this target."
    )


def build_review_prompt(
    *,
    target_title: str,
    target_type: str,
    current_content: str,
    candidate_content: str,
    invariant_constraints: dict[str, Any],
) -> str:
    return (
        f"Target title:\n{target_title}\n\n"
        f"Target type:\n{target_type}\n\n"
        f"Invariant constraints:\n{json.dumps(invariant_constraints, indent=2, sort_keys=True)}\n\n"
        f"Current content:\n```\n{current_content}\n```\n\n"
        f"Candidate content:\n```\n{candidate_content}\n```\n\n"
        "Approve only if the candidate preserves the target purpose and respects the constraints."
    )


def suite_summary(suite: Any) -> dict[str, Any]:
    return {
        "name": suite.name,
        "target_type": suite.compatible_target_type,
        "scoring_method": suite.scoring_method,
        "aggregation": suite.aggregation,
        "pass_threshold": suite.pass_threshold,
        "repetitions": suite.repetitions,
        "cases": suite.cases,
    }


def evaluate_criteria(output_text: str, criteria: list[dict[str, Any]]) -> tuple[float, bool, list[dict[str, Any]]]:
    if not criteria:
        return 1.0, True, []

    normalized = output_text
    lowered = output_text.lower()
    results: list[dict[str, Any]] = []
    passes = 0
    for criterion in criteria:
        kind = str(criterion.get("type", "")).strip()
        passed = False
        if kind == "contains_all":
            values = [str(item).lower() for item in criterion.get("values", [])]
            passed = bool(values) and all(item in lowered for item in values)
        elif kind == "contains_any":
            values = [str(item).lower() for item in criterion.get("values", [])]
            passed = bool(values) and any(item in lowered for item in values)
        elif kind == "not_contains_any":
            values = [str(item).lower() for item in criterion.get("values", [])]
            passed = bool(values) and all(item not in lowered for item in values)
        elif kind == "max_length":
            passed = len(normalized) <= int(criterion.get("value", 0))
        elif kind == "min_length":
            passed = len(normalized) >= int(criterion.get("value", 0))
        elif kind == "regex":
            pattern = str(criterion.get("pattern", ""))
            passed = bool(pattern) and re.search(pattern, normalized, flags=re.MULTILINE) is not None
        elif kind == "starts_with":
            value = str(criterion.get("value", ""))
            passed = normalized.startswith(value)
        elif kind == "ends_with":
            value = str(criterion.get("value", ""))
            passed = normalized.endswith(value)
        results.append({"type": kind, "passed": passed, "criterion": criterion})
        if passed:
            passes += 1
    score = passes / len(criteria)
    return score, passes == len(criteria), results


def aggregate_scores(scores: list[float], aggregation: str) -> float:
    if not scores:
        return 0.0
    normalized = aggregation.strip().lower()
    if normalized == "median":
        return float(statistics.median(scores))
    if normalized == "max":
        return max(scores)
    if normalized == "pass_rate":
        return sum(scores) / len(scores)
    return sum(scores) / len(scores)


class SkillOptimizerMode(ModeAdapter):
    name = "skill_optimizer"
    label = "Skill Optimizer"
    description = "Generic prompt/skill optimization mode with LLM mutation, invariant review, and repeated eval suites."

    def bootstrap_proposal(self, *, ledger: Any) -> ActionProposal | None:
        return None

    def build_proposals(self, *, ledger: Any, claim: Any) -> list[ActionProposal]:
        targets = [item for item in ledger.targets_for_claim(claim.id) if item.mode == self.name]
        if not targets:
            return []
        target = sorted(targets, key=lambda item: item.updated_at)[-1]
        suites = [item for item in ledger.eval_suites_for_claim(claim.id) if item.target_id == target.id]
        candidates = [
            item for item in ledger.mutation_candidates_for_claim(claim.id) if item.target_id == target.id
        ]
        eval_runs = [item for item in ledger.eval_runs_for_claim(claim.id) if item.target_id == target.id]
        candidate_stats = self._candidate_stats(candidates=candidates, eval_runs=eval_runs)
        evaluated_candidate_ids = {item["candidate"].id for item in candidate_stats}
        best_candidate = max(candidate_stats, key=lambda item: item["score"], default=None)
        approved_unevaluated = [
            item
            for item in candidates
            if item.review_status == ReviewStatus.APPROVED and item.id not in evaluated_candidate_ids
        ]
        metrics = ledger.claim_metrics(claim.id)
        proposals: list[ActionProposal] = []

        if not suites:
            proposals.append(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.ANALYZE_FAILURE,
                    expected_information_gain=0.6,
                    priority="next",
                    reason="The target exists, but no eval suite is registered yet.",
                    executor=ActionExecutor.SKILL_OPTIMIZER,
                    mode=self.name,
                    stage="setup",
                    command_hint=(
                        "Register an eval suite with: PYTHONPATH=src python -m aieq_core.cli "
                        f"eval register <ledger> --mode {self.name} --claim-id {claim.id} --target-id {target.id} --suite-file <suite.json>"
                    ),
                )
            )
            return proposals

        if not candidates or all(item.review_status != ReviewStatus.APPROVED for item in candidates):
            score = clamp(0.45 * claim.novelty + 0.25 * claim.falsifiability + 0.30 * metrics["uncertainty"])
            proposals.append(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.DESIGN_MUTATION,
                    expected_information_gain=score,
                    priority=self._priority(score),
                    reason="No approved mutation candidate exists yet for this target.",
                    executor=ActionExecutor.SKILL_OPTIMIZER,
                    mode=self.name,
                    stage="mutation",
                    command_hint=(
                        f"PYTHONPATH=src python -m aieq_core.cli run-next {ledger.path} --mode {self.name}"
                    ),
                )
            )
            return proposals

        if approved_unevaluated:
            score = clamp(
                0.40 * metrics["uncertainty"]
                + 0.25 * claim.novelty
                + 0.20 * claim.falsifiability
                + 0.15
            )
            proposals.append(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.RUN_EVAL,
                    expected_information_gain=score,
                    priority=self._priority(score),
                    reason="An approved mutation candidate is waiting for repeated eval runs.",
                    executor=ActionExecutor.SKILL_OPTIMIZER,
                    mode=self.name,
                    stage="evaluation",
                    command_hint=(
                        f"PYTHONPATH=src python -m aieq_core.cli run-next {ledger.path} --mode {self.name}"
                    ),
                )
            )

        if best_candidate is not None and target.promoted_candidate_id != best_candidate["candidate"].id:
            suite_threshold = max((suite.pass_threshold for suite in suites), default=1.0)
            if best_candidate["score"] >= suite_threshold:
                score = clamp(0.35 + 0.45 * best_candidate["score"] + 0.20 * (1.0 - metrics["uncertainty"]))
                proposals.append(
                    ActionProposal(
                        claim_id=claim.id,
                        claim_title=claim.title,
                        action_type=ActionType.PROMOTE_WINNER,
                        expected_information_gain=score,
                        priority=self._priority(score),
                        reason="The best evaluated candidate cleared the configured threshold and can be promoted.",
                        executor=ActionExecutor.SKILL_OPTIMIZER,
                        mode=self.name,
                        stage="promotion",
                        command_hint=(
                            "Promote the current best candidate with: "
                            f"PYTHONPATH=src python -m aieq_core.cli promote-winner {ledger.path} --claim-id {claim.id}"
                        ),
                    )
                )

        if metrics["optimization_stagnation_candidate_count"] >= 1 and candidate_stats:
            score = clamp(
                0.30
                + 0.30 * metrics["uncertainty"]
                + 0.20 * claim.novelty
                + 0.20 * clamp(metrics["optimization_stagnation_candidate_count"] / 3.0)
            )
            proposals.append(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.ANALYZE_FAILURE,
                    expected_information_gain=score,
                    priority=self._priority(score),
                    reason="Recent evaluated candidates are not materially improving the score.",
                    executor=ActionExecutor.SKILL_OPTIMIZER,
                    mode=self.name,
                    stage="analysis",
                    command_hint=(
                        f"PYTHONPATH=src python -m aieq_core.cli run-next {ledger.path} --mode {self.name}"
                    ),
                )
            )

        if not proposals:
            score = clamp(0.35 + 0.35 * claim.novelty + 0.30 * metrics["uncertainty"])
            proposals.append(
                ActionProposal(
                    claim_id=claim.id,
                    claim_title=claim.title,
                    action_type=ActionType.DESIGN_MUTATION,
                    expected_information_gain=score,
                    priority=self._priority(score),
                    reason="The next best move is to propose another candidate mutation.",
                    executor=ActionExecutor.SKILL_OPTIMIZER,
                    mode=self.name,
                    stage="mutation",
                    command_hint=(
                        f"PYTHONPATH=src python -m aieq_core.cli run-next {ledger.path} --mode {self.name}"
                    ),
                )
            )

        return proposals

    def doctor(self, *, config: Any, ledger_path: str | None = None) -> dict[str, Any]:
        key_present = bool(config.env.get("OPENAI_API_KEY"))
        available = key_present
        blocked_by = [] if available else ["Missing OPENAI_API_KEY for skill optimization."]
        payload = {
            "mode": self.name,
            "runtime": {
                "mutation_model": config.skill_mutation_model,
                "review_model": config.skill_review_model,
                "eval_model": config.skill_eval_model,
            },
            "capabilities": {
                "design_mutation": {
                    "action": "design_mutation",
                    "executor": "skill_optimizer",
                    "available": available,
                    "blocked_by": blocked_by,
                    "model": config.skill_mutation_model,
                },
                "run_eval": {
                    "action": "run_eval",
                    "executor": "skill_optimizer",
                    "available": available,
                    "blocked_by": blocked_by,
                    "model": config.skill_eval_model,
                },
                "promote_winner": {
                    "action": "promote_winner",
                    "executor": "skill_optimizer",
                    "available": True,
                    "blocked_by": [],
                },
                "analyze_failure": {
                    "action": "analyze_failure",
                    "executor": "skill_optimizer",
                    "available": True,
                    "blocked_by": [],
                },
            },
        }
        if ledger_path:
            payload["ledger_path"] = ledger_path
        return payload

    def execute_action(
        self,
        *,
        orchestrator: Any,
        ledger: Any,
        proposal: ActionProposal,
        decision_id: str,
        data_description: str,
        data_description_file: str,
    ) -> dict[str, Any]:
        if action_matches(proposal.action_type, ActionType.DESIGN_MUTATION):
            return self._execute_design_mutation(
                orchestrator=orchestrator,
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        if action_matches(proposal.action_type, ActionType.RUN_EVAL):
            return self._execute_run_eval(
                orchestrator=orchestrator,
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        if action_matches(proposal.action_type, ActionType.PROMOTE_WINNER):
            return self._execute_promote_winner(
                orchestrator=orchestrator,
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        if action_matches(proposal.action_type, ActionType.ANALYZE_FAILURE):
            return self._execute_analyze_failure(
                orchestrator=orchestrator,
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        raise orchestrator.unsupported_action_error(proposal.action_type.value)

    def _execute_design_mutation(
        self,
        *,
        orchestrator: Any,
        ledger: Any,
        proposal: ActionProposal,
        decision_id: str,
    ) -> dict[str, Any]:
        claim = ledger.get_claim(proposal.claim_id)
        target = self._latest_target(ledger=ledger, claim_id=claim.id)
        suite = self._primary_suite(ledger=ledger, claim_id=claim.id, target_id=target.id)
        execution_dir = orchestrator.config.execution_dir(decision_id)
        client = self._client(orchestrator)
        current_content = self._base_content_for_target(ledger=ledger, target=target)
        draft = client.mutate(
            target_title=target.title,
            target_type=target.target_type,
            current_content=current_content,
            invariant_constraints=target.invariant_constraints,
            suite=suite_summary(suite),
        )
        review = client.review(
            target_title=target.title,
            target_type=target.target_type,
            current_content=current_content,
            candidate_content=draft.content,
            invariant_constraints=target.invariant_constraints,
        )
        prompt_path = execution_dir / "prompt-mutation.prompt.txt"
        response_path = execution_dir / "prompt-mutation.response.json"
        candidate_path = execution_dir / "prompt-mutation.candidate.txt"
        review_prompt_path = execution_dir / "prompt-mutation.review.prompt.txt"
        review_response_path = execution_dir / "prompt-mutation.review.response.json"
        baseline_path = execution_dir / "prompt-mutation.original.txt"
        prompt_path.write_text(draft.prompt, encoding="utf-8")
        response_path.write_text(json.dumps(draft.raw_response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate_path.write_text(draft.content, encoding="utf-8")
        review_prompt_path.write_text(review.prompt, encoding="utf-8")
        review_response_path.write_text(json.dumps(review.raw_response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        baseline_path.write_text(current_content, encoding="utf-8")
        candidate = ledger.add_mutation_candidate(
            claim_id=claim.id,
            target_id=target.id,
            summary=draft.summary,
            content=draft.content,
            source_type="skill_optimizer",
            source_ref=draft.response_id,
            source_path=str(candidate_path),
            review_status=ReviewStatus.APPROVED if review.approved else ReviewStatus.REJECTED,
            review_notes=review.summary,
            artifact_paths=[
                str(prompt_path),
                str(response_path),
                str(candidate_path),
                str(review_prompt_path),
                str(review_response_path),
                str(baseline_path),
            ],
            metadata={
                "mode": self.name,
                "target_title": target.title,
                "suite_id": suite.id,
                "review": {
                    "approved": review.approved,
                    "summary": review.summary,
                    "blockers": review.blockers,
                    "warnings": review.warnings,
                    "model": review.model,
                },
                "mutation": {
                    "model": draft.model,
                    "usage": draft.usage,
                },
            },
        )
        execution_status = ExecutionStatus.SUCCEEDED if review.approved else ExecutionStatus.SKIPPED
        execution = ledger.record_execution(
            decision_id=decision_id,
            claim_id=claim.id,
            claim_title=claim.title,
            action_type=proposal.action_type,
            executor=proposal.executor,
            mode=self.name,
            status=execution_status,
            notes=review.summary,
            artifact_quality=0.85 if review.approved else 0.45,
            artifact_paths=candidate.artifact_paths,
            metadata={
                "mode": self.name,
                "target_id": target.id,
                "suite_id": suite.id,
                "candidate_id": candidate.id,
            },
        )
        return {
            "ok": bool(review.approved),
            "action": proposal.action_type.value,
            "mode": self.name,
            "candidate": {
                "id": candidate.id,
                "summary": candidate.summary,
                "review_status": candidate.review_status.value,
                "review_notes": candidate.review_notes,
                "source_path": candidate.source_path,
            },
            "execution": execution.metadata | {"id": execution.id, "status": execution.status.value},
        }

    def _execute_run_eval(
        self,
        *,
        orchestrator: Any,
        ledger: Any,
        proposal: ActionProposal,
        decision_id: str,
    ) -> dict[str, Any]:
        claim = ledger.get_claim(proposal.claim_id)
        target = self._latest_target(ledger=ledger, claim_id=claim.id)
        suite = self._primary_suite(ledger=ledger, claim_id=claim.id, target_id=target.id)
        candidate = self._next_candidate_for_eval(ledger=ledger, claim_id=claim.id, target_id=target.id)
        execution_dir = orchestrator.config.execution_dir(decision_id)
        client = self._client(orchestrator)
        run_scores: list[float] = []
        pass_flags: list[bool] = []
        artifact_paths: list[str] = []
        total_runtime_seconds = 0.0
        total_cost = 0.0
        case_outputs: list[dict[str, Any]] = []
        for case in suite.cases:
            case_id = str(case.get("id", "")).strip() or f"case-{len(case_outputs) + 1}"
            case_input = str(case.get("input", "")).strip()
            criteria = case.get("criteria", [])
            for run_index in range(1, suite.repetitions + 1):
                response = client.evaluate_case(candidate_content=candidate.content, case_input=case_input)
                score, passed, criterion_results = evaluate_criteria(response.output_text, criteria)
                output_path = execution_dir / f"eval-{case_id}-run-{run_index}.json"
                payload = {
                    "case_id": case_id,
                    "run_index": run_index,
                    "input": case_input,
                    "output_text": response.output_text,
                    "score": score,
                    "passed": passed,
                    "criterion_results": criterion_results,
                    "model": response.model,
                    "response_id": response.response_id,
                    "usage": response.usage,
                }
                output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                artifact_paths.append(str(output_path))
                ledger.record_eval_run(
                    claim_id=claim.id,
                    target_id=target.id,
                    suite_id=suite.id,
                    candidate_id=candidate.id,
                    case_id=case_id,
                    run_index=run_index,
                    score=score,
                    passed=passed,
                    raw_output=response.output_text,
                    runtime_seconds=float(response.usage.get("total_tokens", 0)) if False else None,
                    cost_estimate_usd=None,
                    artifact_paths=[str(output_path)],
                    metadata={
                        "criterion_results": criterion_results,
                        "response_id": response.response_id,
                        "usage": response.usage,
                    },
                )
                run_scores.append(score)
                pass_flags.append(passed)
                total_cost += 0.0
                case_outputs.append(payload)
        aggregate_score = aggregate_scores(run_scores, suite.aggregation)
        pass_rate = sum(1 for item in pass_flags if item) / len(pass_flags) if pass_flags else 0.0
        previous_best = self._previous_best_score(
            ledger=ledger,
            claim_id=claim.id,
            target_id=target.id,
            excluding_candidate_id=candidate.id,
        )
        delta = aggregate_score - previous_best if previous_best is not None else 0.0
        if previous_best is None:
            direction = (
                EvidenceDirection.SUPPORT if aggregate_score >= suite.pass_threshold else EvidenceDirection.INCONCLUSIVE
            )
        elif aggregate_score > previous_best + 1e-6:
            direction = EvidenceDirection.SUPPORT
        elif aggregate_score < previous_best - 1e-6:
            direction = EvidenceDirection.CONTRADICT
        else:
            direction = EvidenceDirection.INCONCLUSIVE
        summary = (
            f"Skill eval for candidate {candidate.id}: aggregate_score={aggregate_score:.3f}, "
            f"pass_rate={pass_rate:.3f}, threshold={suite.pass_threshold:.3f}, "
            f"delta_vs_previous_best={delta:+.3f}."
        )
        evidence = ledger.add_evidence(
            claim_id=claim.id,
            summary=summary,
            direction=direction,
            strength=clamp(0.30 + 0.50 * aggregate_score + 0.20 * pass_rate),
            confidence=clamp(0.45 + 0.10 * len(suite.cases) + 0.10 * suite.repetitions),
            source_type="skill_optimizer",
            source_ref=candidate.id,
            artifact_paths=artifact_paths,
            metadata={
                "mode": self.name,
                "stage": "eval_summary",
                "target_id": target.id,
                "suite_id": suite.id,
                "candidate_id": candidate.id,
                "aggregate_score": aggregate_score,
                "pass_rate": pass_rate,
                "previous_best_score": previous_best,
                "delta_vs_previous_best": delta,
                "case_output_count": len(case_outputs),
            },
        )
        execution = ledger.record_execution(
            decision_id=decision_id,
            claim_id=claim.id,
            claim_title=claim.title,
            action_type=proposal.action_type,
            executor=proposal.executor,
            mode=self.name,
            status=ExecutionStatus.SUCCEEDED,
            notes=summary,
            runtime_seconds=round(total_runtime_seconds, 3),
            cost_estimate_usd=round(total_cost, 6),
            artifact_quality=clamp(0.40 + 0.40 * aggregate_score + 0.20 * pass_rate),
            artifact_paths=artifact_paths,
            metadata={
                "mode": self.name,
                "target_id": target.id,
                "suite_id": suite.id,
                "candidate_id": candidate.id,
                "aggregate_score": aggregate_score,
                "pass_rate": pass_rate,
                "evidence_id": evidence.id,
            },
        )
        evidence.metadata["execution_id"] = execution.id
        ledger.save()
        return {
            "ok": True,
            "action": proposal.action_type.value,
            "mode": self.name,
            "candidate_id": candidate.id,
            "aggregate_score": round(aggregate_score, 6),
            "pass_rate": round(pass_rate, 6),
            "threshold": suite.pass_threshold,
            "execution": {"id": execution.id, "status": execution.status.value},
        }

    def _execute_promote_winner(
        self,
        *,
        orchestrator: Any,
        ledger: Any,
        proposal: ActionProposal,
        decision_id: str,
    ) -> dict[str, Any]:
        claim = ledger.get_claim(proposal.claim_id)
        target = self._latest_target(ledger=ledger, claim_id=claim.id)
        suite = self._primary_suite(ledger=ledger, claim_id=claim.id, target_id=target.id)
        candidate = self._best_candidate(ledger=ledger, claim_id=claim.id, target_id=target.id)
        if candidate is None:
            raise SkillOptimizerError("No evaluated candidate is available to promote.")
        snapshot_dir = orchestrator.config.execution_dir(decision_id)
        snapshot_path = snapshot_dir / "promoted-candidate.txt"
        snapshot_path.write_text(candidate["candidate"].content, encoding="utf-8")
        ledger.promote_candidate(target_id=target.id, candidate_id=candidate["candidate"].id)
        summary = (
            f"Promoted candidate {candidate['candidate'].id} with aggregate_score={candidate['score']:.3f} "
            f"against threshold={suite.pass_threshold:.3f}."
        )
        ledger.add_evidence(
            claim_id=claim.id,
            summary=summary,
            direction=EvidenceDirection.SUPPORT,
            strength=clamp(0.40 + 0.50 * candidate["score"]),
            confidence=0.8,
            source_type="skill_optimizer",
            source_ref=candidate["candidate"].id,
            artifact_paths=[str(snapshot_path)],
            metadata={
                "mode": self.name,
                "stage": "promotion",
                "target_id": target.id,
                "suite_id": suite.id,
                "candidate_id": candidate["candidate"].id,
                "aggregate_score": candidate["score"],
            },
        )
        execution = ledger.record_execution(
            decision_id=decision_id,
            claim_id=claim.id,
            claim_title=claim.title,
            action_type=proposal.action_type,
            executor=proposal.executor,
            mode=self.name,
            status=ExecutionStatus.SUCCEEDED,
            notes=summary,
            artifact_quality=clamp(0.55 + 0.35 * candidate["score"]),
            artifact_paths=[str(snapshot_path)],
            metadata={
                "mode": self.name,
                "target_id": target.id,
                "suite_id": suite.id,
                "candidate_id": candidate["candidate"].id,
                "aggregate_score": candidate["score"],
            },
        )
        return {
            "ok": True,
            "action": proposal.action_type.value,
            "mode": self.name,
            "promoted_candidate_id": candidate["candidate"].id,
            "snapshot_path": str(snapshot_path),
            "execution": {"id": execution.id, "status": execution.status.value},
        }

    def _execute_analyze_failure(
        self,
        *,
        orchestrator: Any,
        ledger: Any,
        proposal: ActionProposal,
        decision_id: str,
    ) -> dict[str, Any]:
        claim = ledger.get_claim(proposal.claim_id)
        target = self._latest_target(ledger=ledger, claim_id=claim.id)
        stats = self._candidate_stats(
            candidates=[
                item for item in ledger.mutation_candidates_for_claim(claim.id) if item.target_id == target.id
            ],
            eval_runs=[item for item in ledger.eval_runs_for_claim(claim.id) if item.target_id == target.id],
        )
        report = {
            "claim_id": claim.id,
            "target_id": target.id,
            "candidate_count": len(stats),
            "best_score": max((item["score"] for item in stats), default=0.0),
            "stagnation_candidate_count": ledger.claim_metrics(claim.id)["optimization_stagnation_candidate_count"],
            "candidates": [
                {
                    "candidate_id": item["candidate"].id,
                    "summary": item["candidate"].summary,
                    "score": item["score"],
                    "pass_rate": item["pass_rate"],
                    "run_count": item["run_count"],
                }
                for item in stats
            ],
        }
        execution_dir = orchestrator.config.execution_dir(decision_id)
        report_path = execution_dir / "analysis-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        execution = ledger.record_execution(
            decision_id=decision_id,
            claim_id=claim.id,
            claim_title=claim.title,
            action_type=proposal.action_type,
            executor=proposal.executor,
            mode=self.name,
            status=ExecutionStatus.SUCCEEDED,
            notes="Generated a skill-optimizer failure analysis report.",
            artifact_quality=0.6,
            artifact_paths=[str(report_path)],
            metadata={"mode": self.name, "target_id": target.id},
        )
        return {
            "ok": True,
            "action": proposal.action_type.value,
            "mode": self.name,
            "report_path": str(report_path),
            "execution": {"id": execution.id, "status": execution.status.value},
        }

    def _client(self, orchestrator: Any) -> OpenAISkillOptimizerClient:
        api_key = orchestrator.config.env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise orchestrator.unsupported_action_error("skill_optimizer requires OPENAI_API_KEY.")
        return OpenAISkillOptimizerClient(
            api_key=api_key,
            mutation_model=orchestrator.config.skill_mutation_model,
            review_model=orchestrator.config.skill_review_model,
            eval_model=orchestrator.config.skill_eval_model,
            timeout_seconds=orchestrator.config.skill_timeout_seconds,
            base_url=orchestrator.config.env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    @staticmethod
    def _latest_target(*, ledger: Any, claim_id: str) -> Any:
        targets = [item for item in ledger.targets_for_claim(claim_id) if item.mode == "skill_optimizer"]
        if not targets:
            raise SkillOptimizerError("No skill-optimizer target is registered for this claim.")
        return sorted(targets, key=lambda item: item.updated_at)[-1]

    @staticmethod
    def _primary_suite(*, ledger: Any, claim_id: str, target_id: str) -> Any:
        suites = [item for item in ledger.eval_suites_for_claim(claim_id) if item.target_id == target_id]
        if not suites:
            raise SkillOptimizerError("No eval suite is registered for this target.")
        return sorted(suites, key=lambda item: item.updated_at)[-1]

    def _next_candidate_for_eval(self, *, ledger: Any, claim_id: str, target_id: str) -> Any:
        candidates = [
            item
            for item in ledger.mutation_candidates_for_claim(claim_id)
            if item.target_id == target_id and item.review_status == ReviewStatus.APPROVED
        ]
        if not candidates:
            raise SkillOptimizerError("No approved candidate is available for evaluation.")
        evaluated_ids = {
            item.candidate_id
            for item in ledger.eval_runs_for_claim(claim_id)
            if item.target_id == target_id
        }
        pending = [item for item in candidates if item.id not in evaluated_ids]
        if pending:
            return sorted(pending, key=lambda item: item.updated_at)[-1]
        return sorted(candidates, key=lambda item: item.updated_at)[-1]

    def _best_candidate(self, *, ledger: Any, claim_id: str, target_id: str) -> dict[str, Any] | None:
        candidates = [item for item in ledger.mutation_candidates_for_claim(claim_id) if item.target_id == target_id]
        eval_runs = [item for item in ledger.eval_runs_for_claim(claim_id) if item.target_id == target_id]
        stats = self._candidate_stats(candidates=candidates, eval_runs=eval_runs)
        if not stats:
            return None
        return max(stats, key=lambda item: item["score"])

    @staticmethod
    def _base_content_for_target(*, ledger: Any, target: Any) -> str:
        if target.promoted_candidate_id:
            try:
                candidate = ledger.get_mutation_candidate(target.promoted_candidate_id)
                return candidate.content
            except KeyError:
                pass
        return target.content

    def _previous_best_score(
        self,
        *,
        ledger: Any,
        claim_id: str,
        target_id: str,
        excluding_candidate_id: str,
    ) -> float | None:
        best = self._best_candidate(ledger=ledger, claim_id=claim_id, target_id=target_id)
        if best is None:
            return None
        if best["candidate"].id == excluding_candidate_id:
            stats = [
                item
                for item in self._candidate_stats(
                    candidates=[
                        candidate
                        for candidate in ledger.mutation_candidates_for_claim(claim_id)
                        if candidate.target_id == target_id and candidate.id != excluding_candidate_id
                    ],
                    eval_runs=[
                        eval_run
                        for eval_run in ledger.eval_runs_for_claim(claim_id)
                        if eval_run.target_id == target_id and eval_run.candidate_id != excluding_candidate_id
                    ],
                )
            ]
            if not stats:
                return None
            return max(item["score"] for item in stats)
        return best["score"]

    @staticmethod
    def _candidate_stats(*, candidates: list[Any], eval_runs: list[Any]) -> list[dict[str, Any]]:
        by_candidate: dict[str, list[Any]] = {}
        for eval_run in eval_runs:
            by_candidate.setdefault(eval_run.candidate_id, []).append(eval_run)
        stats: list[dict[str, Any]] = []
        for candidate in candidates:
            runs = by_candidate.get(candidate.id, [])
            if not runs:
                continue
            score = sum(item.score for item in runs) / len(runs)
            pass_rate = sum(1 for item in runs if item.passed) / len(runs)
            stats.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "pass_rate": pass_rate,
                    "run_count": len(runs),
                }
            )
        return stats

    @staticmethod
    def _priority(score: float) -> str:
        if score >= 0.75:
            return "now"
        if score >= 0.55:
            return "next"
        return "watch"
