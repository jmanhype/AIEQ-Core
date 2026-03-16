from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .models import (
    Assumption,
    Artifact,
    ArtifactTarget,
    ArtifactKind,
    Attack,
    AttackStatus,
    Claim,
    ClaimStatus,
    DecisionRecord,
    Evidence,
    EvidenceDirection,
    EvalRun,
    EvalSuite,
    ExecutionRecord,
    ExecutionStatus,
    MutationCandidate,
    ReviewStatus,
    ActionExecutor,
    ActionProposal,
    ActionType,
    clamp,
    serialize_dataclass,
    utc_now,
)

SCHEMA_VERSION = 4


class EpistemicLedger:
    """Persistent claim/evidence graph for automated research workflows."""

    def __init__(
        self,
        path: str | Path,
        *,
        claims: dict[str, Claim] | None = None,
        assumptions: dict[str, Assumption] | None = None,
        evidence: dict[str, Evidence] | None = None,
        attacks: dict[str, Attack] | None = None,
        artifacts: dict[str, Artifact] | None = None,
        targets: dict[str, ArtifactTarget] | None = None,
        eval_suites: dict[str, EvalSuite] | None = None,
        mutation_candidates: dict[str, MutationCandidate] | None = None,
        eval_runs: dict[str, EvalRun] | None = None,
        decisions: dict[str, DecisionRecord] | None = None,
        executions: dict[str, ExecutionRecord] | None = None,
    ) -> None:
        self.path = Path(path)
        self.claims = claims or {}
        self.assumptions = assumptions or {}
        self.evidence = evidence or {}
        self.attacks = attacks or {}
        self.artifacts = artifacts or {}
        self.targets = targets or {}
        self.eval_suites = eval_suites or {}
        self.mutation_candidates = mutation_candidates or {}
        self.eval_runs = eval_runs or {}
        self.decisions = decisions or {}
        self.executions = executions or {}
        self._rebuild_indexes()
        self.refresh_all()

    @classmethod
    def load(cls, path: str | Path) -> "EpistemicLedger":
        ledger_path = Path(path)
        if not ledger_path.exists():
            return cls(ledger_path)

        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        claims = {
            raw["id"]: Claim(
                **{
                    **raw,
                    "status": ClaimStatus(raw.get("status", ClaimStatus.PROPOSED.value)),
                }
            )
            for raw in payload.get("claims", [])
        }
        assumptions = {
            raw["id"]: Assumption(**raw)
            for raw in payload.get("assumptions", [])
        }
        evidence = {
            raw["id"]: Evidence(
                **{
                    **raw,
                    "direction": EvidenceDirection(
                        raw.get("direction", EvidenceDirection.INCONCLUSIVE.value)
                    ),
                }
            )
            for raw in payload.get("evidence", [])
        }
        attacks = {
            raw["id"]: Attack(
                **{
                    **raw,
                    "status": AttackStatus(raw.get("status", AttackStatus.OPEN.value)),
                }
            )
            for raw in payload.get("attacks", [])
        }
        artifacts = {
            raw["id"]: Artifact(
                **{
                    **raw,
                    "kind": ArtifactKind(raw.get("kind", ArtifactKind.METHOD.value)),
                }
            )
            for raw in payload.get("artifacts", [])
        }
        targets = {
            raw["id"]: ArtifactTarget(**raw)
            for raw in payload.get("targets", [])
        }
        eval_suites = {
            raw["id"]: EvalSuite(**raw)
            for raw in payload.get("eval_suites", [])
        }
        mutation_candidates = {
            raw["id"]: MutationCandidate(
                **{
                    **raw,
                    "review_status": ReviewStatus(
                        raw.get("review_status", ReviewStatus.PENDING.value)
                    ),
                }
            )
            for raw in payload.get("mutation_candidates", [])
        }
        eval_runs = {
            raw["id"]: EvalRun(**raw)
            for raw in payload.get("eval_runs", [])
        }
        decisions = {
            raw["id"]: DecisionRecord(
                **{
                    **raw,
                    "action_type": ActionType(raw["action_type"]),
                    "executor": ActionExecutor(raw["executor"]),
                }
            )
            for raw in payload.get("decisions", [])
        }
        executions = {
            raw["id"]: ExecutionRecord(
                **{
                    **raw,
                    "action_type": ActionType(raw["action_type"]),
                    "executor": ActionExecutor(raw["executor"]),
                    "status": ExecutionStatus(raw["status"]),
                }
            )
            for raw in payload.get("executions", [])
        }
        return cls(
            ledger_path,
            claims=claims,
            assumptions=assumptions,
            evidence=evidence,
            attacks=attacks,
            artifacts=artifacts,
            targets=targets,
            eval_suites=eval_suites,
            mutation_candidates=mutation_candidates,
            eval_runs=eval_runs,
            decisions=decisions,
            executions=executions,
        )

    def save(self) -> Path:
        self.refresh_all()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": utc_now(),
            "claims": [serialize_dataclass(claim) for claim in self.list_claims()],
            "assumptions": [
                serialize_dataclass(assumption)
                for assumption in sorted(
                    self.assumptions.values(), key=lambda item: item.created_at
                )
            ],
            "evidence": [
                serialize_dataclass(item)
                for item in sorted(self.evidence.values(), key=lambda item: item.created_at)
            ],
            "attacks": [
                serialize_dataclass(item)
                for item in sorted(self.attacks.values(), key=lambda item: item.created_at)
            ],
            "artifacts": [
                serialize_dataclass(item)
                for item in sorted(self.artifacts.values(), key=lambda item: item.created_at)
            ],
            "targets": [
                serialize_dataclass(item)
                for item in sorted(self.targets.values(), key=lambda item: item.created_at)
            ],
            "eval_suites": [
                serialize_dataclass(item)
                for item in sorted(self.eval_suites.values(), key=lambda item: item.created_at)
            ],
            "mutation_candidates": [
                serialize_dataclass(item)
                for item in sorted(
                    self.mutation_candidates.values(), key=lambda item: item.created_at
                )
            ],
            "eval_runs": [
                serialize_dataclass(item)
                for item in sorted(self.eval_runs.values(), key=lambda item: item.created_at)
            ],
            "decisions": [
                serialize_dataclass(item)
                for item in sorted(self.decisions.values(), key=lambda item: item.created_at)
            ],
            "executions": [
                serialize_dataclass(item)
                for item in sorted(self.executions.values(), key=lambda item: item.created_at)
            ],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.path

    def list_claims(self) -> list[Claim]:
        return sorted(self.claims.values(), key=lambda item: item.created_at)

    def claim_snapshot(self, claim_id: str) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        return {
            "claim": serialize_dataclass(claim),
            "metrics": self.claim_metrics(claim_id),
            "assumptions": [
                serialize_dataclass(item) for item in self.assumptions_for_claim(claim_id)
            ],
            "evidence": [
                serialize_dataclass(item) for item in self.evidence_for_claim(claim_id)
            ],
            "attacks": [serialize_dataclass(item) for item in self.attacks_for_claim(claim_id)],
            "artifacts": [
                serialize_dataclass(item) for item in self.artifacts_for_claim(claim_id)
            ],
            "targets": [serialize_dataclass(item) for item in self.targets_for_claim(claim_id)],
            "eval_suites": [
                serialize_dataclass(item) for item in self.eval_suites_for_claim(claim_id)
            ],
            "mutation_candidates": [
                serialize_dataclass(item)
                for item in self.mutation_candidates_for_claim(claim_id)
            ],
            "eval_runs": [serialize_dataclass(item) for item in self.eval_runs_for_claim(claim_id)],
            "decisions": [
                serialize_dataclass(item) for item in self.decisions_for_claim(claim_id)
            ],
            "executions": [
                serialize_dataclass(item) for item in self.executions_for_claim(claim_id)
            ],
        }

    def summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for claim in self.list_claims():
            metrics = self.claim_metrics(claim.id)
            rows.append(
                {
                    "claim_id": claim.id,
                    "title": claim.title,
                    "mode": str(claim.metadata.get("mode", "")).strip() or "ml_research",
                    "status": claim.status.value,
                    "confidence": round(claim.confidence, 3),
                    "uncertainty": round(metrics["uncertainty"], 3),
                    "evidence_count": metrics["evidence_count"],
                    "open_attack_count": metrics["open_attack_count"],
                    "artifact_count": metrics["artifact_count"],
                    "target_count": metrics["target_count"],
                    "eval_suite_count": metrics["eval_suite_count"],
                    "mutation_candidate_count": metrics["mutation_candidate_count"],
                    "eval_run_count": metrics["eval_run_count"],
                    "decision_count": metrics["decision_count"],
                    "execution_count": metrics["execution_count"],
                    "novelty": round(claim.novelty, 3),
                    "falsifiability": round(claim.falsifiability, 3),
                }
            )
        return rows

    def add_claim(
        self,
        *,
        title: str,
        statement: str,
        novelty: float = 0.5,
        falsifiability: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        claim_id: str | None = None,
    ) -> Claim:
        claim = Claim(
            id=claim_id or self._generate_id("claim"),
            title=title,
            statement=statement,
            novelty=novelty,
            falsifiability=falsifiability,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.claims[claim.id] = claim
        self.save()
        return claim

    def add_assumption(
        self,
        *,
        claim_id: str,
        text: str,
        rationale: str = "",
        risk: float = 0.5,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        assumption_id: str | None = None,
    ) -> Assumption:
        self.get_claim(claim_id)
        assumption = Assumption(
            id=assumption_id or self._generate_id("asm"),
            claim_id=claim_id,
            text=text,
            rationale=rationale,
            risk=risk,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.assumptions[assumption.id] = assumption
        self.save()
        return assumption

    def add_evidence(
        self,
        *,
        claim_id: str,
        summary: str,
        direction: EvidenceDirection | str = EvidenceDirection.INCONCLUSIVE,
        strength: float = 0.5,
        confidence: float = 0.5,
        source_type: str = "manual",
        source_ref: str = "",
        artifact_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> Evidence:
        self.get_claim(claim_id)
        evidence = Evidence(
            id=evidence_id or self._generate_id("ev"),
            claim_id=claim_id,
            summary=summary,
            direction=(
                direction
                if isinstance(direction, EvidenceDirection)
                else EvidenceDirection(direction)
            ),
            strength=strength,
            confidence=confidence,
            source_type=source_type,
            source_ref=source_ref,
            artifact_paths=artifact_paths or [],
            metadata=metadata or {},
        )
        self.evidence[evidence.id] = evidence
        self.save()
        return evidence

    def add_attack(
        self,
        *,
        claim_id: str,
        description: str,
        target_kind: str = "claim",
        target_id: str = "",
        severity: float = 0.5,
        status: AttackStatus | str = AttackStatus.OPEN,
        resolution: str = "",
        metadata: dict[str, Any] | None = None,
        attack_id: str | None = None,
    ) -> Attack:
        self.get_claim(claim_id)
        attack = Attack(
            id=attack_id or self._generate_id("atk"),
            claim_id=claim_id,
            description=description,
            target_kind=target_kind,
            target_id=target_id,
            severity=severity,
            status=status if isinstance(status, AttackStatus) else AttackStatus(status),
            resolution=resolution,
            metadata=metadata or {},
        )
        self.attacks[attack.id] = attack
        self.save()
        return attack

    def add_artifact(
        self,
        *,
        claim_id: str,
        kind: ArtifactKind | str,
        title: str,
        content: str = "",
        source_type: str = "manual",
        source_ref: str = "",
        source_path: str = "",
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> Artifact:
        self.get_claim(claim_id)
        artifact = Artifact(
            id=artifact_id or self._generate_id("art"),
            claim_id=claim_id,
            kind=kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind),
            title=title,
            content=content,
            source_type=source_type,
            source_ref=source_ref,
            source_path=source_path,
            metadata=metadata or {},
        )
        self.artifacts[artifact.id] = artifact
        self.save()
        return artifact

    def register_target(
        self,
        *,
        claim_id: str,
        mode: str,
        target_type: str,
        title: str,
        content: str = "",
        source_type: str = "manual",
        source_ref: str = "",
        source_path: str = "",
        mutable_fields: list[str] | None = None,
        invariant_constraints: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        target_id: str | None = None,
    ) -> ArtifactTarget:
        self.get_claim(claim_id)
        target = ArtifactTarget(
            id=target_id or self._generate_id("tgt"),
            claim_id=claim_id,
            mode=mode,
            target_type=target_type,
            title=title,
            content=content,
            source_type=source_type,
            source_ref=source_ref,
            source_path=source_path,
            mutable_fields=mutable_fields or [],
            invariant_constraints=invariant_constraints or {},
            metadata=metadata or {},
        )
        self.targets[target.id] = target
        self.save()
        return target

    def register_eval_suite(
        self,
        *,
        claim_id: str,
        target_id: str,
        name: str,
        compatible_target_type: str,
        scoring_method: str = "binary",
        aggregation: str = "average",
        pass_threshold: float = 1.0,
        repetitions: int = 1,
        cases: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        suite_id: str | None = None,
    ) -> EvalSuite:
        self.get_claim(claim_id)
        self.get_target(target_id)
        suite = EvalSuite(
            id=suite_id or self._generate_id("eval"),
            claim_id=claim_id,
            target_id=target_id,
            name=name,
            compatible_target_type=compatible_target_type,
            scoring_method=scoring_method,
            aggregation=aggregation,
            pass_threshold=pass_threshold,
            repetitions=repetitions,
            cases=cases or [],
            metadata=metadata or {},
        )
        self.eval_suites[suite.id] = suite
        self.save()
        return suite

    def add_mutation_candidate(
        self,
        *,
        claim_id: str,
        target_id: str,
        summary: str,
        content: str,
        source_type: str = "manual",
        source_ref: str = "",
        source_path: str = "",
        parent_candidate_id: str = "",
        review_status: ReviewStatus | str = ReviewStatus.PENDING,
        review_notes: str = "",
        artifact_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        candidate_id: str | None = None,
    ) -> MutationCandidate:
        self.get_claim(claim_id)
        self.get_target(target_id)
        candidate = MutationCandidate(
            id=candidate_id or self._generate_id("cand"),
            claim_id=claim_id,
            target_id=target_id,
            parent_candidate_id=parent_candidate_id,
            summary=summary,
            content=content,
            source_type=source_type,
            source_ref=source_ref,
            source_path=source_path,
            review_status=(
                review_status
                if isinstance(review_status, ReviewStatus)
                else ReviewStatus(review_status)
            ),
            review_notes=review_notes,
            artifact_paths=artifact_paths or [],
            metadata=metadata or {},
        )
        self.mutation_candidates[candidate.id] = candidate
        self.save()
        return candidate

    def record_eval_run(
        self,
        *,
        claim_id: str,
        target_id: str,
        suite_id: str,
        candidate_id: str,
        case_id: str,
        run_index: int,
        score: float,
        passed: bool,
        raw_output: str = "",
        runtime_seconds: float | None = None,
        cost_estimate_usd: float | None = None,
        artifact_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        eval_run_id: str | None = None,
    ) -> EvalRun:
        self.get_claim(claim_id)
        self.get_target(target_id)
        self.get_eval_suite(suite_id)
        self.get_mutation_candidate(candidate_id)
        eval_run = EvalRun(
            id=eval_run_id or self._generate_id("erun"),
            claim_id=claim_id,
            target_id=target_id,
            suite_id=suite_id,
            candidate_id=candidate_id,
            case_id=case_id,
            run_index=run_index,
            score=score,
            passed=passed,
            raw_output=raw_output,
            runtime_seconds=runtime_seconds,
            cost_estimate_usd=cost_estimate_usd,
            artifact_paths=artifact_paths or [],
            metadata=metadata or {},
        )
        self.eval_runs[eval_run.id] = eval_run
        self.save()
        return eval_run

    def promote_candidate(
        self,
        *,
        target_id: str,
        candidate_id: str,
    ) -> ArtifactTarget:
        target = self.get_target(target_id)
        self.get_mutation_candidate(candidate_id)
        target.promoted_candidate_id = candidate_id
        target.updated_at = utc_now()
        self.save()
        return target

    def upsert_artifact(
        self,
        *,
        claim_id: str,
        kind: ArtifactKind | str,
        title: str,
        content: str = "",
        source_type: str = "manual",
        source_ref: str = "",
        source_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        normalized_kind = kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind)
        existing = next(
            (
                item
                for item in self.artifacts_for_claim(claim_id)
                if item.kind == normalized_kind and item.source_path == source_path
            ),
            None,
        )
        if existing is None:
            return self.add_artifact(
                claim_id=claim_id,
                kind=normalized_kind,
                title=title,
                content=content,
                source_type=source_type,
                source_ref=source_ref,
                source_path=source_path,
                metadata=metadata,
            )

        existing.title = title
        existing.content = content
        existing.source_type = source_type
        existing.source_ref = source_ref
        existing.source_path = source_path
        existing.updated_at = utc_now()
        existing.metadata = metadata or {}
        self.save()
        return existing

    def assumptions_for_claim(self, claim_id: str) -> list[Assumption]:
        return [
            self.assumptions[assumption_id]
            for assumption_id in self.get_claim(claim_id).assumption_ids
            if assumption_id in self.assumptions
        ]

    def evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        return [
            self.evidence[evidence_id]
            for evidence_id in self.get_claim(claim_id).evidence_ids
            if evidence_id in self.evidence
        ]

    def attacks_for_claim(self, claim_id: str) -> list[Attack]:
        return [
            self.attacks[attack_id]
            for attack_id in self.get_claim(claim_id).attack_ids
            if attack_id in self.attacks
        ]

    def artifacts_for_claim(self, claim_id: str) -> list[Artifact]:
        return [
            self.artifacts[artifact_id]
            for artifact_id in self.get_claim(claim_id).artifact_ids
            if artifact_id in self.artifacts
        ]

    def targets_for_claim(self, claim_id: str) -> list[ArtifactTarget]:
        return [
            self.targets[target_id]
            for target_id in self.get_claim(claim_id).target_ids
            if target_id in self.targets
        ]

    def eval_suites_for_claim(self, claim_id: str) -> list[EvalSuite]:
        return [
            self.eval_suites[suite_id]
            for suite_id in self.get_claim(claim_id).eval_suite_ids
            if suite_id in self.eval_suites
        ]

    def mutation_candidates_for_claim(self, claim_id: str) -> list[MutationCandidate]:
        return [
            self.mutation_candidates[candidate_id]
            for candidate_id in self.get_claim(claim_id).mutation_candidate_ids
            if candidate_id in self.mutation_candidates
        ]

    def eval_runs_for_claim(self, claim_id: str) -> list[EvalRun]:
        return [
            self.eval_runs[eval_run_id]
            for eval_run_id in self.get_claim(claim_id).eval_run_ids
            if eval_run_id in self.eval_runs
        ]

    def decisions_for_claim(self, claim_id: str) -> list[DecisionRecord]:
        return [
            self.decisions[decision_id]
            for decision_id in self.get_claim(claim_id).decision_ids
            if decision_id in self.decisions
        ]

    def executions_for_claim(self, claim_id: str) -> list[ExecutionRecord]:
        return [
            self.executions[execution_id]
            for execution_id in self.get_claim(claim_id).execution_ids
            if execution_id in self.executions
        ]

    def get_claim(self, claim_id: str) -> Claim:
        try:
            return self.claims[claim_id]
        except KeyError as exc:
            raise KeyError(f"Unknown claim id: {claim_id}") from exc

    def get_decision(self, decision_id: str) -> DecisionRecord:
        try:
            return self.decisions[decision_id]
        except KeyError as exc:
            raise KeyError(f"Unknown decision id: {decision_id}") from exc

    def get_target(self, target_id: str) -> ArtifactTarget:
        try:
            return self.targets[target_id]
        except KeyError as exc:
            raise KeyError(f"Unknown target id: {target_id}") from exc

    def get_eval_suite(self, suite_id: str) -> EvalSuite:
        try:
            return self.eval_suites[suite_id]
        except KeyError as exc:
            raise KeyError(f"Unknown eval suite id: {suite_id}") from exc

    def get_mutation_candidate(self, candidate_id: str) -> MutationCandidate:
        try:
            return self.mutation_candidates[candidate_id]
        except KeyError as exc:
            raise KeyError(f"Unknown mutation candidate id: {candidate_id}") from exc

    def list_decisions(self) -> list[DecisionRecord]:
        return sorted(self.decisions.values(), key=lambda item: item.created_at)

    def list_executions(self) -> list[ExecutionRecord]:
        return sorted(self.executions.values(), key=lambda item: item.created_at)

    def record_decision(
        self,
        proposal: ActionProposal,
        *,
        metadata: dict[str, Any] | None = None,
        decision_id: str | None = None,
    ) -> DecisionRecord:
        record = DecisionRecord(
            id=decision_id or self._generate_id("dec"),
            claim_id=proposal.claim_id,
            claim_title=proposal.claim_title,
            action_type=proposal.action_type,
            executor=proposal.executor,
            mode=proposal.mode,
            stage=proposal.stage,
            priority=proposal.priority,
            expected_information_gain=proposal.expected_information_gain,
            reason=proposal.reason,
            command_hint=proposal.command_hint,
            metadata=metadata or {},
        )
        self.decisions[record.id] = record
        self.save()
        return record

    def record_execution(
        self,
        *,
        status: ExecutionStatus | str,
        decision_id: str = "",
        claim_id: str = "",
        claim_title: str = "",
        action_type: ActionType | str | None = None,
        executor: ActionExecutor | str = ActionExecutor.MANUAL,
        mode: str = "",
        notes: str = "",
        runtime_seconds: float | None = None,
        cost_estimate_usd: float | None = None,
        artifact_quality: float | None = None,
        artifact_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> ExecutionRecord:
        if decision_id:
            decision = self.get_decision(decision_id)
            if not claim_id:
                claim_id = decision.claim_id
            if not claim_title:
                claim_title = decision.claim_title
            if action_type is None:
                action_type = decision.action_type
            if executor in {"", ActionExecutor.MANUAL, "manual"}:
                executor = decision.executor
            if not mode:
                mode = decision.mode

        if action_type is None:
            raise ValueError("record_execution requires either decision_id or action_type")
        if claim_id and claim_id not in self.claims:
            raise KeyError(f"Unknown claim id: {claim_id}")
        if claim_id and not claim_title:
            claim_title = self.get_claim(claim_id).title

        record = ExecutionRecord(
            id=execution_id or self._generate_id("exe"),
            decision_id=decision_id,
            claim_id=claim_id,
            claim_title=claim_title,
            action_type=action_type if isinstance(action_type, ActionType) else ActionType(action_type),
            executor=executor if isinstance(executor, ActionExecutor) else ActionExecutor(executor),
            status=status if isinstance(status, ExecutionStatus) else ExecutionStatus(status),
            mode=mode,
            notes=notes,
            runtime_seconds=runtime_seconds,
            cost_estimate_usd=cost_estimate_usd,
            artifact_quality=artifact_quality,
            artifact_paths=artifact_paths or [],
            metadata=metadata or {},
        )
        self.executions[record.id] = record
        self.save()
        return record

    def refresh_all(self) -> None:
        self._rebuild_indexes()
        for claim_id in self.claims:
            self.refresh_claim(claim_id)

    def refresh_claim(self, claim_id: str) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        metrics = self.claim_metrics(claim_id)

        if claim.status != ClaimStatus.ARCHIVED:
            claim.status = self._derive_status(claim, metrics)
        claim.confidence = metrics["belief"]
        claim.updated_at = utc_now()
        return metrics

    def claim_metrics(self, claim_id: str) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        assumptions = self.assumptions_for_claim(claim_id)
        evidence = self.evidence_for_claim(claim_id)
        attacks = self.attacks_for_claim(claim_id)
        artifacts = self.artifacts_for_claim(claim_id)
        targets = self.targets_for_claim(claim_id)
        eval_suites = self.eval_suites_for_claim(claim_id)
        mutation_candidates = self.mutation_candidates_for_claim(claim_id)
        eval_runs = self.eval_runs_for_claim(claim_id)
        decisions = self.decisions_for_claim(claim_id)
        executions = self.executions_for_claim(claim_id)
        autoresearch_meta = claim.metadata.get("autoresearch", {})
        if not isinstance(autoresearch_meta, dict):
            autoresearch_meta = {}
        autoresearch_series = autoresearch_meta.get("series", {})
        if not isinstance(autoresearch_series, dict):
            autoresearch_series = {}
        autoresearch_aggregate = autoresearch_meta.get("aggregate_series", {})
        if not isinstance(autoresearch_aggregate, dict):
            autoresearch_aggregate = {}

        def _series_float(key: str) -> float:
            value = autoresearch_series.get(key)
            return float(value) if value is not None else 0.0

        def _series_int(key: str) -> int:
            value = autoresearch_series.get(key)
            return int(value) if value is not None else 0

        def _aggregate_float(key: str) -> float:
            value = autoresearch_aggregate.get(key)
            return float(value) if value is not None else 0.0

        def _aggregate_int(key: str) -> int:
            value = autoresearch_aggregate.get(key)
            return int(value) if value is not None else 0

        known_artifact_qualities = [
            item.artifact_quality
            for item in executions
            if item.artifact_quality is not None
        ]
        total_runtime_seconds = sum(item.runtime_seconds or 0.0 for item in executions)
        failed_runtime_seconds = sum(
            item.runtime_seconds or 0.0
            for item in executions
            if item.status == ExecutionStatus.FAILED
        )
        total_cost_usd = sum(item.cost_estimate_usd or 0.0 for item in executions)
        failed_cost_usd = sum(
            item.cost_estimate_usd or 0.0
            for item in executions
            if item.status == ExecutionStatus.FAILED
        )

        support_score = sum(
            item.strength * item.confidence
            for item in evidence
            if item.direction == EvidenceDirection.SUPPORT
        )
        contradict_score = sum(
            item.strength * item.confidence
            for item in evidence
            if item.direction == EvidenceDirection.CONTRADICT
        )
        inconclusive_score = sum(
            item.strength * item.confidence
            for item in evidence
            if item.direction == EvidenceDirection.INCONCLUSIVE
        )
        evidence_weight = support_score + contradict_score + inconclusive_score

        if evidence_weight == 0:
            belief = 0.0
            uncertainty = 1.0
        else:
            belief = clamp(
                (support_score + 0.5 * inconclusive_score) / evidence_weight
            )
            uncertainty = clamp(
                1.0 - abs(support_score - contradict_score) / evidence_weight
            )

        open_attacks = [item for item in attacks if item.status == AttackStatus.OPEN]
        open_attack_load = clamp(sum(item.severity for item in open_attacks) / 2.0)
        average_assumption_risk = (
            sum(item.risk for item in assumptions) / len(assumptions) if assumptions else 0.0
        )
        highest_risk_assumption = (
            max(assumptions, key=lambda item: item.risk) if assumptions else None
        )
        method_artifact_count = len(
            [item for item in artifacts if item.kind == ArtifactKind.METHOD]
        )
        paper_artifact_count = len(
            [item for item in artifacts if item.kind == ArtifactKind.PAPER]
        )
        candidate_scores: dict[str, list[float]] = {}
        candidate_passes: dict[str, list[bool]] = {}
        for eval_run in eval_runs:
            candidate_scores.setdefault(eval_run.candidate_id, []).append(eval_run.score)
            candidate_passes.setdefault(eval_run.candidate_id, []).append(eval_run.passed)

        best_candidate_id = ""
        best_candidate_score = 0.0
        candidate_average_pass_rate = 0.0
        promoted_candidate_count = len(
            [item for item in targets if item.promoted_candidate_id.strip()]
        )
        threshold_met = False
        evaluated_candidates = []
        for candidate in mutation_candidates:
            scores = candidate_scores.get(candidate.id, [])
            passes = candidate_passes.get(candidate.id, [])
            if not scores:
                continue
            average_score = sum(scores) / len(scores)
            pass_rate = (
                sum(1 for item in passes if item) / len(passes) if passes else 0.0
            )
            if average_score > best_candidate_score:
                best_candidate_score = average_score
                best_candidate_id = candidate.id
            candidate_average_pass_rate += pass_rate
            evaluated_candidates.append((candidate, average_score))

        candidate_average_pass_rate = (
            candidate_average_pass_rate / len(evaluated_candidates)
            if evaluated_candidates
            else 0.0
        )
        stagnation_candidate_count = 0
        running_best = -1.0
        last_improvement_index = -1
        for index, (_, average_score) in enumerate(
            sorted(evaluated_candidates, key=lambda item: item[0].created_at)
        ):
            if average_score > running_best + 1e-9:
                running_best = average_score
                last_improvement_index = index
        if evaluated_candidates:
            stagnation_candidate_count = max(0, len(evaluated_candidates) - last_improvement_index - 1)
        for suite in eval_suites:
            if best_candidate_score >= suite.pass_threshold:
                threshold_met = True
                break

        return {
            "belief": round(belief, 6),
            "uncertainty": round(uncertainty, 6),
            "support_score": round(support_score, 6),
            "contradict_score": round(contradict_score, 6),
            "inconclusive_score": round(inconclusive_score, 6),
            "evidence_weight": round(evidence_weight, 6),
            "evidence_count": len(evidence),
            "open_attack_count": len(open_attacks),
            "open_attack_load": round(open_attack_load, 6),
            "artifact_count": len(artifacts),
            "method_artifact_count": method_artifact_count,
            "paper_artifact_count": paper_artifact_count,
            "target_count": len(targets),
            "eval_suite_count": len(eval_suites),
            "mutation_candidate_count": len(mutation_candidates),
            "eval_run_count": len(eval_runs),
            "reviewed_candidate_count": len(
                [
                    item
                    for item in mutation_candidates
                    if item.review_status in {ReviewStatus.APPROVED, ReviewStatus.REJECTED}
                ]
            ),
            "approved_candidate_count": len(
                [item for item in mutation_candidates if item.review_status == ReviewStatus.APPROVED]
            ),
            "promoted_candidate_count": promoted_candidate_count,
            "optimization_best_candidate_id": best_candidate_id,
            "optimization_best_score": round(best_candidate_score, 6),
            "optimization_average_pass_rate": round(candidate_average_pass_rate, 6),
            "optimization_threshold_met": threshold_met,
            "optimization_stagnation_candidate_count": stagnation_candidate_count,
            "assumption_risk": round(average_assumption_risk, 6),
            "decision_count": len(decisions),
            "execution_count": len(executions),
            "failed_execution_count": len(
                [item for item in executions if item.status == ExecutionStatus.FAILED]
            ),
            "successful_execution_count": len(
                [item for item in executions if item.status == ExecutionStatus.SUCCEEDED]
            ),
            "total_runtime_seconds": round(total_runtime_seconds, 3),
            "failed_runtime_seconds": round(failed_runtime_seconds, 3),
            "total_cost_usd": round(total_cost_usd, 6),
            "failed_cost_usd": round(failed_cost_usd, 6),
            "average_artifact_quality": round(
                (
                    sum(known_artifact_qualities) / len(known_artifact_qualities)
                    if known_artifact_qualities
                    else 0.0
                ),
                6,
            ),
            "highest_risk_assumption_id": (
                highest_risk_assumption.id if highest_risk_assumption else ""
            ),
            "highest_risk_assumption_risk": (
                round(highest_risk_assumption.risk, 6)
                if highest_risk_assumption
                else 0.0
            ),
            "autoresearch_series_run_count": _series_int("total_runs"),
            "autoresearch_series_keep_rate": round(_series_float("keep_rate"), 6),
            "autoresearch_series_crash_rate": round(_series_float("crash_rate"), 6),
            "autoresearch_series_frontier_improvement_count": _series_int(
                "frontier_improvement_count"
            ),
            "autoresearch_series_stagnation_run_count": _series_int("stagnation_run_count"),
            "autoresearch_series_best_improvement_bpb": round(
                _series_float("best_improvement_bpb"), 6
            ),
            "autoresearch_series_average_memory_gb": round(
                _series_float("average_memory_gb"), 3
            ),
            "autoresearch_branch_count": _aggregate_int("branch_count"),
            "autoresearch_active_branch_count": _aggregate_int("active_branch_count"),
            "autoresearch_plateau_branch_count": _aggregate_int("plateau_branch_count"),
            "autoresearch_total_run_count_all_branches": _aggregate_int("total_runs"),
            "autoresearch_best_branch": str(
                autoresearch_aggregate.get("preferred_branch", "")
            ).strip(),
            "autoresearch_aggregate_keep_rate": round(_aggregate_float("keep_rate"), 6),
            "autoresearch_aggregate_crash_rate": round(_aggregate_float("crash_rate"), 6),
        }

    def _derive_status(self, claim: Claim, metrics: dict[str, Any]) -> ClaimStatus:
        if metrics["contradict_score"] >= max(0.7, metrics["support_score"] * 1.25):
            return ClaimStatus.FALSIFIED
        if (
            metrics["support_score"] >= max(0.7, metrics["contradict_score"] * 1.5)
            and metrics["open_attack_load"] < 0.25
        ):
            return ClaimStatus.SUPPORTED
        if metrics["evidence_count"] > 0 or metrics["open_attack_count"] > 0:
            if metrics["contradict_score"] > 0 or metrics["open_attack_count"] > 0:
                return ClaimStatus.CONTESTED
            return ClaimStatus.ACTIVE
        return ClaimStatus.PROPOSED

    def _rebuild_indexes(self) -> None:
        for claim in self.claims.values():
            claim.assumption_ids = []
            claim.evidence_ids = []
            claim.attack_ids = []
            claim.artifact_ids = []
            claim.target_ids = []
            claim.eval_suite_ids = []
            claim.mutation_candidate_ids = []
            claim.eval_run_ids = []
            claim.decision_ids = []
            claim.execution_ids = []

        for assumption in self.assumptions.values():
            if assumption.claim_id in self.claims:
                self.claims[assumption.claim_id].assumption_ids.append(assumption.id)

        for evidence in self.evidence.values():
            if evidence.claim_id in self.claims:
                self.claims[evidence.claim_id].evidence_ids.append(evidence.id)

        for attack in self.attacks.values():
            if attack.claim_id in self.claims:
                self.claims[attack.claim_id].attack_ids.append(attack.id)

        for artifact in self.artifacts.values():
            if artifact.claim_id in self.claims:
                self.claims[artifact.claim_id].artifact_ids.append(artifact.id)

        for target in self.targets.values():
            if target.claim_id in self.claims:
                self.claims[target.claim_id].target_ids.append(target.id)

        for suite in self.eval_suites.values():
            if suite.claim_id in self.claims:
                self.claims[suite.claim_id].eval_suite_ids.append(suite.id)

        for candidate in self.mutation_candidates.values():
            if candidate.claim_id in self.claims:
                self.claims[candidate.claim_id].mutation_candidate_ids.append(candidate.id)

        for eval_run in self.eval_runs.values():
            if eval_run.claim_id in self.claims:
                self.claims[eval_run.claim_id].eval_run_ids.append(eval_run.id)

        for decision in self.decisions.values():
            if decision.claim_id in self.claims:
                self.claims[decision.claim_id].decision_ids.append(decision.id)

        for execution in self.executions.values():
            if execution.claim_id in self.claims:
                self.claims[execution.claim_id].execution_ids.append(execution.id)

    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"
