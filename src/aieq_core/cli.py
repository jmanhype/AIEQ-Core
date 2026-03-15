from __future__ import annotations

import argparse
import json
from typing import Sequence

from .adapters.autoresearch import AutoresearchAdapter
from .adapters.denario import DenarioAdapter
from .controller import ResearchController
from .ledger import EpistemicLedger
from .models import (
    ActionExecutor,
    ActionType,
    AttackStatus,
    EvidenceDirection,
    ExecutionStatus,
    serialize_dataclass,
)
from .policy import ExpectedInformationGainPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aieq-core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create an empty ledger file")
    init_parser.add_argument("ledger", help="Path to the ledger JSON file")

    claim_parser = subparsers.add_parser("add-claim", help="Add a claim")
    claim_parser.add_argument("ledger", help="Path to the ledger JSON file")
    claim_parser.add_argument("--title", required=True)
    claim_parser.add_argument("--statement", required=True)
    claim_parser.add_argument("--novelty", type=float, default=0.5)
    claim_parser.add_argument("--falsifiability", type=float, default=0.5)
    claim_parser.add_argument("--tag", action="append", default=[])

    assumption_parser = subparsers.add_parser("add-assumption", help="Add an assumption")
    assumption_parser.add_argument("ledger", help="Path to the ledger JSON file")
    assumption_parser.add_argument("--claim-id", required=True)
    assumption_parser.add_argument("--text", required=True)
    assumption_parser.add_argument("--rationale", default="")
    assumption_parser.add_argument("--risk", type=float, default=0.5)
    assumption_parser.add_argument("--tag", action="append", default=[])

    evidence_parser = subparsers.add_parser("add-evidence", help="Add evidence")
    evidence_parser.add_argument("ledger", help="Path to the ledger JSON file")
    evidence_parser.add_argument("--claim-id", required=True)
    evidence_parser.add_argument("--summary", required=True)
    evidence_parser.add_argument(
        "--direction",
        choices=[item.value for item in EvidenceDirection],
        default=EvidenceDirection.INCONCLUSIVE.value,
    )
    evidence_parser.add_argument("--strength", type=float, default=0.5)
    evidence_parser.add_argument("--confidence", type=float, default=0.5)
    evidence_parser.add_argument("--source-type", default="manual")
    evidence_parser.add_argument("--source-ref", default="")
    evidence_parser.add_argument("--artifact", action="append", default=[])

    attack_parser = subparsers.add_parser("add-attack", help="Add an attack")
    attack_parser.add_argument("ledger", help="Path to the ledger JSON file")
    attack_parser.add_argument("--claim-id", required=True)
    attack_parser.add_argument("--description", required=True)
    attack_parser.add_argument("--target-kind", default="claim")
    attack_parser.add_argument("--target-id", default="")
    attack_parser.add_argument("--severity", type=float, default=0.5)
    attack_parser.add_argument(
        "--status",
        choices=[item.value for item in AttackStatus],
        default=AttackStatus.OPEN.value,
    )
    attack_parser.add_argument("--resolution", default="")

    show_parser = subparsers.add_parser("show", help="Show summary or one claim snapshot")
    show_parser.add_argument("ledger", help="Path to the ledger JSON file")
    show_parser.add_argument("--claim-id")

    rank_parser = subparsers.add_parser(
        "rank-actions", help="Rank next actions by expected information gain"
    )
    rank_parser.add_argument("ledger", help="Path to the ledger JSON file")
    rank_parser.add_argument("--limit", type=int, default=10)

    import_autoresearch = subparsers.add_parser(
        "import-autoresearch-run",
        help="Parse an autoresearch run log and attach it as ledger evidence",
    )
    import_autoresearch.add_argument("ledger", help="Path to the ledger JSON file")
    import_autoresearch.add_argument("--claim-id", required=True)
    import_autoresearch.add_argument("--run-log", required=True)
    import_autoresearch.add_argument("--commit", default="")
    import_autoresearch.add_argument("--branch", default="")
    import_autoresearch.add_argument("--description", default="")
    import_autoresearch.add_argument("--status", default="")
    import_autoresearch.add_argument("--baseline-bpb", type=float)
    import_autoresearch.add_argument("--tolerance", type=float, default=1e-4)
    import_autoresearch.add_argument("--artifact", action="append", default=[])
    import_autoresearch.add_argument("--decision-id", default="")
    import_autoresearch.add_argument(
        "--execution-status",
        choices=[item.value for item in ExecutionStatus],
    )
    import_autoresearch.add_argument("--cost-usd", type=float)
    import_autoresearch.add_argument("--artifact-quality", type=float)

    import_autoresearch_results = subparsers.add_parser(
        "import-autoresearch-results",
        help="Import an autoresearch results.tsv history and summarize it for the controller",
    )
    import_autoresearch_results.add_argument("ledger", help="Path to the ledger JSON file")
    import_autoresearch_results.add_argument("--claim-id", required=True)
    import_autoresearch_results.add_argument("--results-tsv", required=True)
    import_autoresearch_results.add_argument("--branch", default="")
    import_autoresearch_results.add_argument("--baseline-bpb", type=float)
    import_autoresearch_results.add_argument("--tolerance", type=float, default=1e-4)

    import_denario = subparsers.add_parser(
        "import-denario-project",
        help="Import a Denario project directory into the ledger",
    )
    import_denario.add_argument("ledger", help="Path to the ledger JSON file")
    import_denario.add_argument("--project-dir", required=True)
    import_denario.add_argument("--claim-id")
    import_denario.add_argument("--novelty", type=float, default=0.7)
    import_denario.add_argument("--falsifiability", type=float, default=0.6)
    import_denario.add_argument("--tag", action="append", default=[])
    import_denario.add_argument(
        "--results-direction",
        choices=[item.value for item in EvidenceDirection],
        default=EvidenceDirection.INCONCLUSIVE.value,
    )
    import_denario.add_argument("--results-strength", type=float, default=0.65)
    import_denario.add_argument("--results-confidence", type=float, default=0.7)
    import_denario.add_argument("--literature-severity", type=float, default=0.6)
    import_denario.add_argument("--referee-severity", type=float, default=0.75)
    import_denario.add_argument("--decision-id", default="")
    import_denario.add_argument(
        "--execution-status",
        choices=[item.value for item in ExecutionStatus],
    )
    import_denario.add_argument("--runtime-seconds", type=float)
    import_denario.add_argument("--cost-usd", type=float)
    import_denario.add_argument("--artifact-quality", type=float)

    decide_parser = subparsers.add_parser(
        "decide-next",
        help="Run the controller and decide the next research action",
    )
    decide_parser.add_argument("ledger", help="Path to the ledger JSON file")
    decide_parser.add_argument("--backlog-limit", type=int, default=5)
    decide_parser.add_argument(
        "--record",
        action="store_true",
        help="Persist the primary controller decision into ledger history",
    )

    execution_parser = subparsers.add_parser(
        "record-execution",
        help="Record the outcome of a previously planned or manual action",
    )
    execution_parser.add_argument("ledger", help="Path to the ledger JSON file")
    execution_parser.add_argument("--decision-id", default="")
    execution_parser.add_argument("--claim-id", default="")
    execution_parser.add_argument("--claim-title", default="")
    execution_parser.add_argument(
        "--action-type",
        choices=[item.value for item in ActionType],
    )
    execution_parser.add_argument(
        "--executor",
        choices=[item.value for item in ActionExecutor],
        default=ActionExecutor.MANUAL.value,
    )
    execution_parser.add_argument(
        "--status",
        choices=[item.value for item in ExecutionStatus],
        required=True,
    )
    execution_parser.add_argument("--notes", default="")
    execution_parser.add_argument("--runtime-seconds", type=float)
    execution_parser.add_argument("--cost-usd", type=float)
    execution_parser.add_argument("--artifact-quality", type=float)
    execution_parser.add_argument("--artifact", action="append", default=[])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        ledger = EpistemicLedger.load(args.ledger)
        ledger.save()
        _emit({"ledger_path": str(ledger.path), "claims": 0})
        return 0

    ledger = EpistemicLedger.load(args.ledger)

    if args.command == "add-claim":
        claim = ledger.add_claim(
            title=args.title,
            statement=args.statement,
            novelty=args.novelty,
            falsifiability=args.falsifiability,
            tags=args.tag,
        )
        _emit(serialize_dataclass(claim))
        return 0

    if args.command == "add-assumption":
        assumption = ledger.add_assumption(
            claim_id=args.claim_id,
            text=args.text,
            rationale=args.rationale,
            risk=args.risk,
            tags=args.tag,
        )
        _emit(serialize_dataclass(assumption))
        return 0

    if args.command == "add-evidence":
        evidence = ledger.add_evidence(
            claim_id=args.claim_id,
            summary=args.summary,
            direction=args.direction,
            strength=args.strength,
            confidence=args.confidence,
            source_type=args.source_type,
            source_ref=args.source_ref,
            artifact_paths=args.artifact,
        )
        _emit(serialize_dataclass(evidence))
        return 0

    if args.command == "add-attack":
        attack = ledger.add_attack(
            claim_id=args.claim_id,
            description=args.description,
            target_kind=args.target_kind,
            target_id=args.target_id,
            severity=args.severity,
            status=args.status,
            resolution=args.resolution,
        )
        _emit(serialize_dataclass(attack))
        return 0

    if args.command == "show":
        if args.claim_id:
            _emit(ledger.claim_snapshot(args.claim_id))
        else:
            _emit({"claims": ledger.summary_rows()})
        return 0

    if args.command == "rank-actions":
        policy = ExpectedInformationGainPolicy()
        proposals = policy.rank_actions(ledger, limit=args.limit)
        _emit({"actions": [serialize_dataclass(item) for item in proposals]})
        return 0

    if args.command == "import-autoresearch-run":
        evidence = AutoresearchAdapter.import_run(
            ledger=ledger,
            claim_id=args.claim_id,
            run_log_path=args.run_log,
            commit=args.commit,
            branch=args.branch,
            description=args.description,
            baseline_bpb=args.baseline_bpb,
            status=args.status,
            tolerance=args.tolerance,
            artifact_paths=args.artifact,
            decision_id=args.decision_id,
            execution_status=args.execution_status,
            cost_estimate_usd=args.cost_usd,
            artifact_quality=args.artifact_quality,
        )
        payload = {"evidence": serialize_dataclass(evidence)}
        execution_id = evidence.metadata.get("execution_id", "")
        if execution_id:
            payload["execution"] = serialize_dataclass(ledger.executions[execution_id])
        _emit(payload)
        return 0

    if args.command == "import-autoresearch-results":
        imported = AutoresearchAdapter.import_results_tsv(
            ledger=ledger,
            claim_id=args.claim_id,
            results_tsv_path=args.results_tsv,
            branch=args.branch,
            baseline_bpb=args.baseline_bpb,
            tolerance=args.tolerance,
        )
        _emit(imported)
        return 0

    if args.command == "import-denario-project":
        imported = DenarioAdapter.import_project(
            ledger=ledger,
            project_dir=args.project_dir,
            claim_id=args.claim_id,
            novelty=args.novelty,
            falsifiability=args.falsifiability,
            tags=args.tag,
            results_direction=args.results_direction,
            results_strength=args.results_strength,
            results_confidence=args.results_confidence,
            literature_severity=args.literature_severity,
            referee_severity=args.referee_severity,
            decision_id=args.decision_id,
            execution_status=args.execution_status,
            runtime_seconds=args.runtime_seconds,
            cost_estimate_usd=args.cost_usd,
            artifact_quality=args.artifact_quality,
        )
        _emit(imported)
        return 0

    if args.command == "decide-next":
        decision = ResearchController().decide(
            ledger,
            backlog_limit=args.backlog_limit,
        )
        payload = {"decision": serialize_dataclass(decision)}
        if args.record:
            record = ledger.record_decision(
                decision.primary_action,
                metadata={
                    "queue_state": decision.queue_state,
                    "summary": decision.summary,
                    "backlog": [serialize_dataclass(item) for item in decision.backlog],
                },
            )
            payload["decision_record"] = serialize_dataclass(record)
        _emit(payload)
        return 0

    if args.command == "record-execution":
        execution = ledger.record_execution(
            decision_id=args.decision_id,
            claim_id=args.claim_id,
            claim_title=args.claim_title,
            action_type=args.action_type,
            executor=args.executor,
            status=args.status,
            notes=args.notes,
            runtime_seconds=args.runtime_seconds,
            cost_estimate_usd=args.cost_usd,
            artifact_quality=args.artifact_quality,
            artifact_paths=args.artifact,
        )
        _emit(serialize_dataclass(execution))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
