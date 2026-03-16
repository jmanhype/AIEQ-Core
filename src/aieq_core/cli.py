from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters.autoresearch import AutoresearchAdapter
from .adapters.denario import DenarioAdapter
from .controller import ResearchController
from .ledger import EpistemicLedger
from .modes import default_mode_registry
from .modes.skill_optimizer import SkillOptimizerMode
from .models import (
    ActionExecutor,
    ActionType,
    AttackStatus,
    EvidenceDirection,
    ExecutionStatus,
    serialize_dataclass,
)
from .orchestrator import ResearchOrchestrator
from .policy import ExpectedInformationGainPolicy
from .runtime import RuntimeConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aieq-core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mode_parser = subparsers.add_parser("mode", help="Inspect registered innovation modes")
    mode_subparsers = mode_parser.add_subparsers(dest="mode_command", required=True)
    mode_subparsers.add_parser("list", help="List available modes")

    target_parser = subparsers.add_parser("target", help="Manage optimization targets")
    target_subparsers = target_parser.add_subparsers(dest="target_command", required=True)
    target_register = target_subparsers.add_parser("register", help="Register a mutable target")
    target_register.add_argument("ledger", help="Path to the ledger JSON file")
    target_register.add_argument("--mode", required=True)
    target_register.add_argument("--title", required=True)
    target_register.add_argument("--statement", default="")
    target_register.add_argument("--claim-id", default="")
    target_register.add_argument("--target-type", default="prompt_template")
    target_register.add_argument("--source-file", default="")
    target_register.add_argument("--content", default="")
    target_register.add_argument("--mutable-field", action="append", default=[])
    target_register.add_argument("--constraint-file", default="")
    target_register.add_argument("--novelty", type=float, default=0.6)
    target_register.add_argument("--falsifiability", type=float, default=0.7)
    target_register.add_argument("--tag", action="append", default=[])

    eval_parser = subparsers.add_parser("eval", help="Manage eval suites")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_register = eval_subparsers.add_parser("register", help="Register an eval suite")
    eval_register.add_argument("ledger", help="Path to the ledger JSON file")
    eval_register.add_argument("--mode", required=True)
    eval_register.add_argument("--claim-id", required=True)
    eval_register.add_argument("--target-id", required=True)
    eval_register.add_argument("--suite-file", required=True)

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
    rank_parser.add_argument("--mode", default="")

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
    decide_parser.add_argument("--mode", default="")
    decide_parser.add_argument(
        "--record",
        action="store_true",
        help="Persist the primary controller decision into ledger history",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect runtime readiness for Denario/autoresearch execution from this repo",
    )
    doctor_parser.add_argument("--ledger", default="")
    doctor_parser.add_argument("--mode", default="")
    doctor_parser.add_argument("--env-file", default="")

    run_next_parser = subparsers.add_parser(
        "run-next",
        help="Decide and execute the next supported action from this repo",
    )
    run_next_parser.add_argument("ledger", help="Path to the ledger JSON file")
    run_next_parser.add_argument("--mode", default="")
    run_next_parser.add_argument("--backlog-limit", type=int, default=5)
    run_next_parser.add_argument("--dry-run", action="store_true")
    run_next_parser.add_argument("--data-description", default="")
    run_next_parser.add_argument("--data-description-file", default="")
    run_next_parser.add_argument("--env-file", default="")

    run_loop_parser = subparsers.add_parser(
        "run-loop",
        help="Run multiple run-next iterations against the same ledger",
    )
    run_loop_parser.add_argument("ledger", help="Path to the ledger JSON file")
    run_loop_parser.add_argument("--mode", default="")
    run_loop_parser.add_argument("--iterations", type=int, default=3)
    run_loop_parser.add_argument("--backlog-limit", type=int, default=5)
    run_loop_parser.add_argument("--data-description", default="")
    run_loop_parser.add_argument("--data-description-file", default="")
    run_loop_parser.add_argument("--env-file", default="")

    promote_parser = subparsers.add_parser(
        "promote-winner",
        help="Promote the best evaluated candidate for a skill-optimizer claim",
    )
    promote_parser.add_argument("ledger", help="Path to the ledger JSON file")
    promote_parser.add_argument("--claim-id", required=True)

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
    execution_parser.add_argument("--mode", default="")
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

    if args.command == "mode":
        registry = default_mode_registry()
        if args.mode_command == "list":
            _emit({"modes": [serialize_dataclass(item) for item in registry.list_modes()]})
            return 0

    if args.command == "target":
        ledger = EpistemicLedger.load(args.ledger)
        if args.target_command == "register":
            if args.mode != "skill_optimizer":
                parser.error("target register currently supports --mode skill_optimizer only.")
            content = _resolve_content(source_file=args.source_file, content=args.content)
            constraints = _load_json_file(args.constraint_file) if args.constraint_file else {}
            claim = (
                ledger.get_claim(args.claim_id)
                if args.claim_id
                else ledger.add_claim(
                    title=args.title,
                    statement=args.statement
                    or f"Optimize `{args.title}` against the registered eval suite.",
                    novelty=args.novelty,
                    falsifiability=args.falsifiability,
                    tags=args.tag,
                    metadata={"mode": args.mode},
                )
            )
            claim.metadata["mode"] = args.mode
            ledger.save()
            target = ledger.register_target(
                claim_id=claim.id,
                mode=args.mode,
                target_type=args.target_type,
                title=args.title,
                content=content,
                source_type="file" if args.source_file else "inline",
                source_path=str(Path(args.source_file).expanduser().resolve()) if args.source_file else "",
                mutable_fields=args.mutable_field,
                invariant_constraints=constraints,
            )
            _emit({"claim": serialize_dataclass(claim), "target": serialize_dataclass(target)})
            return 0

    if args.command == "eval":
        ledger = EpistemicLedger.load(args.ledger)
        if args.eval_command == "register":
            if args.mode != "skill_optimizer":
                parser.error("eval register currently supports --mode skill_optimizer only.")
            suite_payload = _load_json_file(args.suite_file)
            suite = ledger.register_eval_suite(
                claim_id=args.claim_id,
                target_id=args.target_id,
                name=str(suite_payload.get("name", "eval-suite")).strip() or "eval-suite",
                compatible_target_type=str(
                    suite_payload.get("compatible_target_type", "prompt_template")
                ).strip()
                or "prompt_template",
                scoring_method=str(suite_payload.get("scoring_method", "binary")).strip() or "binary",
                aggregation=str(suite_payload.get("aggregation", "average")).strip() or "average",
                pass_threshold=float(suite_payload.get("pass_threshold", 1.0)),
                repetitions=int(suite_payload.get("repetitions", 1)),
                cases=list(suite_payload.get("cases", [])),
                metadata={
                    key: value
                    for key, value in suite_payload.items()
                    if key
                    not in {
                        "name",
                        "compatible_target_type",
                        "scoring_method",
                        "aggregation",
                        "pass_threshold",
                        "repetitions",
                        "cases",
                    }
                },
            )
            claim = ledger.get_claim(args.claim_id)
            claim.metadata["mode"] = args.mode
            ledger.save()
            _emit({"eval_suite": serialize_dataclass(suite)})
            return 0

    if args.command == "init":
        ledger = EpistemicLedger.load(args.ledger)
        ledger.save()
        _emit({"ledger_path": str(ledger.path), "claims": 0})
        return 0

    if args.command == "doctor":
        config = RuntimeConfig.load(env_file=args.env_file or None)
        orchestrator = ResearchOrchestrator(config=config)
        payload = orchestrator.doctor(ledger_path=args.ledger or None, mode=args.mode)
        _emit(payload)
        return 0

    if args.command == "run-next":
        config = RuntimeConfig.load(env_file=args.env_file or None)
        orchestrator = ResearchOrchestrator(config=config)
        payload = orchestrator.run_next(
            args.ledger,
            backlog_limit=args.backlog_limit,
            dry_run=args.dry_run,
            data_description=args.data_description,
            data_description_file=args.data_description_file,
            mode=args.mode,
        )
        _emit(payload)
        return 0 if payload.get("ok", True) else 1

    if args.command == "run-loop":
        config = RuntimeConfig.load(env_file=args.env_file or None)
        orchestrator = ResearchOrchestrator(config=config)
        runs = []
        final_ok = True
        for _ in range(max(1, args.iterations)):
            payload = orchestrator.run_next(
                args.ledger,
                backlog_limit=args.backlog_limit,
                dry_run=False,
                data_description=args.data_description,
                data_description_file=args.data_description_file,
                mode=args.mode,
            )
            runs.append(payload)
            final_ok = final_ok and bool(payload.get("ok", True))
        _emit({"iterations": len(runs), "ok": final_ok, "runs": runs})
        return 0 if final_ok else 1

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
        decision = ResearchController().decide(
            ledger,
            backlog_limit=args.limit,
            mode_hint=args.mode,
        )
        _emit(
            {
                "actions": [
                    serialize_dataclass(item)
                    for item in [decision.primary_action, *decision.backlog][: args.limit]
                ]
            }
        )
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
            mode_hint=args.mode,
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
            mode=args.mode,
            status=args.status,
            notes=args.notes,
            runtime_seconds=args.runtime_seconds,
            cost_estimate_usd=args.cost_usd,
            artifact_quality=args.artifact_quality,
            artifact_paths=args.artifact,
        )
        _emit(serialize_dataclass(execution))
        return 0

    if args.command == "promote-winner":
        claim = ledger.get_claim(args.claim_id)
        if str(claim.metadata.get("mode", "")).strip() != "skill_optimizer":
            parser.error("promote-winner currently supports skill_optimizer claims only.")
        mode = SkillOptimizerMode()
        target = mode._latest_target(ledger=ledger, claim_id=claim.id)
        best = mode._best_candidate(ledger=ledger, claim_id=claim.id, target_id=target.id)
        if best is None:
            parser.error("No evaluated candidate exists for this claim.")
        ledger.promote_candidate(target_id=target.id, candidate_id=best["candidate"].id)
        _emit(
            {
                "claim_id": claim.id,
                "target_id": target.id,
                "promoted_candidate_id": best["candidate"].id,
                "aggregate_score": round(best["score"], 6),
            }
        )
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _resolve_content(*, source_file: str, content: str) -> str:
    if source_file:
        return Path(source_file).expanduser().read_text(encoding="utf-8")
    return content


def _load_json_file(path: str) -> dict[str, object]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _emit(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
