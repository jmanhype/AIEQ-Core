from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    FALSIFIED = "falsified"
    ARCHIVED = "archived"


class EvidenceDirection(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    INCONCLUSIVE = "inconclusive"


class AttackStatus(str, Enum):
    OPEN = "open"
    ADDRESSED = "addressed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ActionType(str, Enum):
    PROPOSE_HYPOTHESIS = "propose_hypothesis"
    DESIGN_MUTATION = "design_mutation"
    RUN_EVAL = "run_eval"
    ANALYZE_FAILURE = "analyze_failure"
    PROMOTE_WINNER = "promote_winner"
    SYNTHESIZE_REPORT = "synthesize_report"
    GENERATE_IDEA = "generate_idea"
    GENERATE_METHOD = "generate_method"
    RUN_EXPERIMENT = "run_experiment"
    CHALLENGE_ASSUMPTION = "challenge_assumption"
    TRIAGE_ATTACK = "triage_attack"
    REPRODUCE_RESULT = "reproduce_result"
    COLLECT_COUNTEREVIDENCE = "collect_counterevidence"
    SYNTHESIZE_PAPER = "synthesize_paper"


class ActionExecutor(str, Enum):
    AIEQ_CORE = "aieq_core"
    DENARIO = "denario"
    AUTORESEARCH = "autoresearch"
    SKILL_OPTIMIZER = "skill_optimizer"
    MANUAL = "manual"


class ExecutionStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ArtifactKind(str, Enum):
    METHOD = "method"
    PAPER = "paper"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    MATERIALIZED = "materialized"
    DISMISSED = "dismissed"


@dataclass(slots=True)
class Claim:
    id: str
    title: str
    statement: str
    status: ClaimStatus = ClaimStatus.PROPOSED
    novelty: float = 0.5
    falsifiability: float = 0.5
    confidence: float = 0.0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    tags: list[str] = field(default_factory=list)
    assumption_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    attack_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    execution_ids: list[str] = field(default_factory=list)
    target_ids: list[str] = field(default_factory=list)
    eval_suite_ids: list[str] = field(default_factory=list)
    mutation_candidate_ids: list[str] = field(default_factory=list)
    eval_run_ids: list[str] = field(default_factory=list)
    input_ids: list[str] = field(default_factory=list)
    hypothesis_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.novelty = clamp(float(self.novelty))
        self.falsifiability = clamp(float(self.falsifiability))
        self.confidence = clamp(float(self.confidence))


@dataclass(slots=True)
class Assumption:
    id: str
    claim_id: str
    text: str
    rationale: str = ""
    risk: float = 0.5
    created_at: str = field(default_factory=utc_now)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.risk = clamp(float(self.risk))


@dataclass(slots=True)
class Evidence:
    id: str
    claim_id: str
    summary: str
    direction: EvidenceDirection = EvidenceDirection.INCONCLUSIVE
    strength: float = 0.5
    confidence: float = 0.5
    source_type: str = "manual"
    source_ref: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.strength = clamp(float(self.strength))
        self.confidence = clamp(float(self.confidence))


@dataclass(slots=True)
class Attack:
    id: str
    claim_id: str
    description: str
    target_kind: str = "claim"
    target_id: str = ""
    severity: float = 0.5
    status: AttackStatus = AttackStatus.OPEN
    created_at: str = field(default_factory=utc_now)
    resolution: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.severity = clamp(float(self.severity))


@dataclass(slots=True)
class Artifact:
    id: str
    claim_id: str
    kind: ArtifactKind = ArtifactKind.METHOD
    title: str = ""
    content: str = ""
    source_type: str = "manual"
    source_ref: str = ""
    source_path: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactTarget:
    id: str
    claim_id: str
    mode: str
    target_type: str
    title: str
    content: str = ""
    source_type: str = "manual"
    source_ref: str = ""
    source_path: str = ""
    mutable_fields: list[str] = field(default_factory=list)
    invariant_constraints: dict[str, Any] = field(default_factory=dict)
    promoted_candidate_id: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InputArtifact:
    id: str
    title: str
    input_type: str
    content: str
    source_type: str = "manual"
    source_ref: str = ""
    source_path: str = ""
    summary: str = ""
    linked_claim_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InnovationHypothesis:
    id: str
    input_id: str
    title: str
    statement: str
    summary: str = ""
    rationale: str = ""
    recommended_mode: str = ""
    target_type: str = ""
    target_title: str = ""
    target_source_strategy: str = ""
    mutable_fields: list[str] = field(default_factory=list)
    suggested_constraints: list[str] = field(default_factory=list)
    eval_outline: list[str] = field(default_factory=list)
    leverage: float = 0.5
    testability: float = 0.5
    novelty: float = 0.5
    optimization_readiness: float = 0.5
    overall_score: float = 0.0
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    materialized_claim_id: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.leverage = clamp(float(self.leverage))
        self.testability = clamp(float(self.testability))
        self.novelty = clamp(float(self.novelty))
        self.optimization_readiness = clamp(float(self.optimization_readiness))
        self.overall_score = clamp(float(self.overall_score))


@dataclass(slots=True)
class EvalSuite:
    id: str
    claim_id: str
    target_id: str
    name: str
    compatible_target_type: str
    scoring_method: str = "binary"
    aggregation: str = "average"
    pass_threshold: float = 1.0
    repetitions: int = 1
    cases: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pass_threshold = clamp(float(self.pass_threshold))
        self.repetitions = max(1, int(self.repetitions))


@dataclass(slots=True)
class MutationCandidate:
    id: str
    claim_id: str
    target_id: str
    parent_candidate_id: str = ""
    summary: str = ""
    content: str = ""
    source_type: str = "manual"
    source_ref: str = ""
    source_path: str = ""
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_notes: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    artifact_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRun:
    id: str
    claim_id: str
    target_id: str
    suite_id: str
    candidate_id: str
    case_id: str
    run_index: int
    score: float
    passed: bool
    raw_output: str = ""
    runtime_seconds: float | None = None
    cost_estimate_usd: float | None = None
    artifact_paths: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_index = max(0, int(self.run_index))
        self.score = clamp(float(self.score))
        if self.runtime_seconds is not None:
            self.runtime_seconds = max(0.0, float(self.runtime_seconds))
        if self.cost_estimate_usd is not None:
            self.cost_estimate_usd = max(0.0, float(self.cost_estimate_usd))


@dataclass(slots=True)
class ActionProposal:
    claim_id: str
    claim_title: str
    action_type: ActionType
    expected_information_gain: float
    priority: str
    reason: str
    executor: ActionExecutor = ActionExecutor.AIEQ_CORE
    mode: str = ""
    stage: str = ""
    command_hint: str = ""

    def __post_init__(self) -> None:
        self.expected_information_gain = clamp(float(self.expected_information_gain))


@dataclass(slots=True)
class ControllerDecision:
    queue_state: str
    summary: str
    primary_action: ActionProposal
    backlog: list[ActionProposal] = field(default_factory=list)


@dataclass(slots=True)
class DecisionRecord:
    id: str
    claim_id: str
    claim_title: str
    action_type: ActionType
    executor: ActionExecutor
    stage: str
    priority: str
    expected_information_gain: float
    reason: str
    mode: str = ""
    command_hint: str = ""
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.expected_information_gain = clamp(float(self.expected_information_gain))


@dataclass(slots=True)
class ExecutionRecord:
    id: str
    decision_id: str
    claim_id: str
    claim_title: str
    action_type: ActionType
    executor: ActionExecutor
    status: ExecutionStatus
    mode: str = ""
    notes: str = ""
    runtime_seconds: float | None = None
    cost_estimate_usd: float | None = None
    artifact_quality: float | None = None
    artifact_paths: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.runtime_seconds is not None:
            self.runtime_seconds = max(0.0, float(self.runtime_seconds))
        if self.cost_estimate_usd is not None:
            self.cost_estimate_usd = max(0.0, float(self.cost_estimate_usd))
        if self.artifact_quality is not None:
            self.artifact_quality = clamp(float(self.artifact_quality))


ACTION_COMPATIBILITY_MAP: dict[ActionType, ActionType] = {
    ActionType.GENERATE_IDEA: ActionType.PROPOSE_HYPOTHESIS,
    ActionType.GENERATE_METHOD: ActionType.DESIGN_MUTATION,
    ActionType.RUN_EXPERIMENT: ActionType.RUN_EVAL,
    ActionType.REPRODUCE_RESULT: ActionType.RUN_EVAL,
    ActionType.SYNTHESIZE_PAPER: ActionType.SYNTHESIZE_REPORT,
}


def canonical_action_type(action_type: ActionType | str) -> ActionType:
    value = action_type if isinstance(action_type, ActionType) else ActionType(action_type)
    return ACTION_COMPATIBILITY_MAP.get(value, value)


def action_matches(action_type: ActionType | str, *candidates: ActionType | str) -> bool:
    canonical = canonical_action_type(action_type)
    return any(canonical == canonical_action_type(item) for item in candidates)


def serialize_dataclass(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: serialize_dataclass(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [serialize_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_dataclass(item) for key, item in value.items()}
    return value
