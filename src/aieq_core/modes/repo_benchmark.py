from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from typing import Any
import shutil

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
from .skill_optimizer import SkillOptimizerError, SkillOptimizerMode, aggregate_scores


class RepoBenchmarkError(SkillOptimizerError):
    """Raised when repo-benchmark execution cannot proceed."""


class RepoBenchmarkMode(SkillOptimizerMode):
    name = "repo_benchmark"
    label = "Repo Benchmark"
    description = (
        "Optimize repo artifacts with LLM mutation/review and evaluate them by running "
        "script-backed benchmark commands against the checked-out repository."
    )
    executor = ActionExecutor.REPO_BENCHMARK

    def doctor(self, *, config: Any, ledger_path: str | None = None) -> dict[str, Any]:
        key_present = bool(config.env.get("OPENAI_API_KEY"))
        python_cmd = self._python_command(config)
        available = key_present and bool(python_cmd)
        blocked_by: list[str] = []
        if not key_present:
            blocked_by.append("Missing OPENAI_API_KEY for repo benchmark mutation/review.")
        if not python_cmd:
            blocked_by.append("No python launcher found for repo benchmark commands.")
        payload = {
            "mode": self.name,
            "runtime": {
                "mutation_model": config.skill_mutation_model,
                "review_model": config.skill_review_model,
                "python_command": python_cmd,
            },
            "capabilities": {
                "design_mutation": {
                    "action": "design_mutation",
                    "executor": self.executor.value,
                    "available": available,
                    "blocked_by": blocked_by,
                    "model": config.skill_mutation_model,
                },
                "run_eval": {
                    "action": "run_eval",
                    "executor": self.executor.value,
                    "available": available,
                    "blocked_by": blocked_by,
                },
                "promote_winner": {
                    "action": "promote_winner",
                    "executor": self.executor.value,
                    "available": True,
                    "blocked_by": [],
                },
                "analyze_failure": {
                    "action": "analyze_failure",
                    "executor": self.executor.value,
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
        if action_matches(proposal.action_type, ActionType.RUN_EVAL):
            return self._execute_run_eval(
                orchestrator=orchestrator,
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        return super().execute_action(
            orchestrator=orchestrator,
            ledger=ledger,
            proposal=proposal,
            decision_id=decision_id,
            data_description=data_description,
            data_description_file=data_description_file,
        )

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
        source_path = Path(target.source_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise RepoBenchmarkError(
                f"Repo benchmark target needs a real file source_path, got: {target.source_path or '<missing>'}."
            )
        repo_root = self._repo_root_for_target(ledger=ledger, target=target)
        execution_dir = orchestrator.config.execution_dir(decision_id)
        original_disk_content = source_path.read_text(encoding="utf-8")
        baseline_content = self._base_content_for_target(ledger=ledger, target=target)

        run_scores: list[float] = []
        pass_flags: list[bool] = []
        artifact_paths: list[str] = []
        case_outputs: list[dict[str, Any]] = []
        baseline_metrics_by_case: dict[str, dict[str, float]] = {}
        total_runtime_seconds = 0.0
        failures: list[str] = []

        try:
            for case in suite.cases:
                case_id = str(case.get("id", "")).strip() or f"case-{len(case_outputs) + 1}"
                role = self._case_role(case)
                checks = list(case.get("checks") or [])
                repetitions = 1 if role == "baseline" else suite.repetitions
                for run_index in range(1, repetitions + 1):
                    applied_content = baseline_content if role == "baseline" else candidate.content
                    source_path.write_text(applied_content, encoding="utf-8")
                    started = time.monotonic()
                    result = self._run_case_command(
                        orchestrator=orchestrator,
                        case=case,
                        repo_root=repo_root,
                    )
                    runtime_seconds = round(time.monotonic() - started, 3)
                    total_runtime_seconds += runtime_seconds
                    metrics = self._parse_case_metrics(case=case, result=result, repo_root=repo_root)
                    baseline_key = str(case.get("baseline_case_id", "")).strip() or "baseline"
                    baseline_metrics = baseline_metrics_by_case.get(baseline_key, {})
                    if role == "baseline":
                        baseline_metrics_by_case[case_id] = metrics
                        score = 1.0
                        passed = True
                        check_results: list[dict[str, Any]] = []
                        include_in_aggregate = False
                    else:
                        score, passed, check_results = self._evaluate_checks(
                            checks=checks,
                            metrics=metrics,
                            baseline_metrics=baseline_metrics,
                        )
                        include_in_aggregate = True
                        run_scores.append(score)
                        pass_flags.append(passed)

                    payload = {
                        "case_id": case_id,
                        "role": role,
                        "run_index": run_index,
                        "command": self._case_command_string(case),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "metrics": metrics,
                        "score": score,
                        "passed": passed,
                        "checks": check_results,
                        "runtime_seconds": runtime_seconds,
                        "include_in_aggregate": include_in_aggregate,
                    }
                    output_path = execution_dir / f"benchmark-{case_id}-run-{run_index}.json"
                    output_path.write_text(
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
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
                        raw_output=json.dumps(
                            {"stdout": result.stdout, "stderr": result.stderr, "metrics": metrics},
                            sort_keys=True,
                        ),
                        runtime_seconds=runtime_seconds,
                        cost_estimate_usd=None,
                        artifact_paths=[str(output_path)],
                        metadata={
                            "mode": self.name,
                            "role": role,
                            "include_in_aggregate": include_in_aggregate,
                            "metrics": metrics,
                            "checks": check_results,
                            "returncode": result.returncode,
                        },
                    )
                    case_outputs.append(payload)
                    if result.returncode != 0:
                        failures.append(
                            f"{case_id} run {run_index} exited with code {result.returncode}."
                        )
        finally:
            source_path.write_text(original_disk_content, encoding="utf-8")

        if not run_scores:
            raise RepoBenchmarkError(
                "Repo benchmark suite did not produce any candidate-scored runs. "
                "Add candidate cases with `checks` and executable `command` fields."
            )

        aggregate_score = self._aggregate_scores(run_scores=run_scores, suite=suite)
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
        failure_suffix = f" Failures: {' '.join(failures)}" if failures else ""
        summary = (
            f"Repo benchmark eval for candidate {candidate.id}: aggregate_score={aggregate_score:.3f}, "
            f"pass_rate={pass_rate:.3f}, threshold={suite.pass_threshold:.3f}, "
            f"delta_vs_previous_best={delta:+.3f}.{failure_suffix}"
        )
        evidence = ledger.add_evidence(
            claim_id=claim.id,
            summary=summary,
            direction=direction,
            strength=clamp(0.30 + 0.50 * aggregate_score + 0.20 * pass_rate),
            confidence=clamp(0.45 + 0.10 * len(run_scores)),
            source_type=self.name,
            source_ref=candidate.id,
            artifact_paths=artifact_paths,
            metadata={
                "mode": self.name,
                "stage": "benchmark_eval_summary",
                "target_id": target.id,
                "suite_id": suite.id,
                "candidate_id": candidate.id,
                "aggregate_score": aggregate_score,
                "pass_rate": pass_rate,
                "previous_best_score": previous_best,
                "delta_vs_previous_best": delta,
                "case_output_count": len(case_outputs),
                "baseline_metrics": baseline_metrics_by_case,
                "failures": failures,
            },
        )
        execution = ledger.record_execution(
            decision_id=decision_id,
            claim_id=claim.id,
            claim_title=claim.title,
            action_type=proposal.action_type,
            executor=proposal.executor,
            mode=self.name,
            status=ExecutionStatus.SUCCEEDED if not failures else ExecutionStatus.FAILED,
            notes=summary,
            runtime_seconds=round(total_runtime_seconds, 3),
            cost_estimate_usd=None,
            artifact_quality=clamp(0.35 + 0.45 * aggregate_score + 0.20 * pass_rate),
            artifact_paths=artifact_paths,
            metadata={
                "mode": self.name,
                "target_id": target.id,
                "suite_id": suite.id,
                "candidate_id": candidate.id,
                "aggregate_score": aggregate_score,
                "pass_rate": pass_rate,
                "evidence_id": evidence.id,
                "baseline_metrics": baseline_metrics_by_case,
                "failures": failures,
            },
        )
        evidence.metadata["execution_id"] = execution.id
        ledger.save()
        return {
            "ok": not failures,
            "action": proposal.action_type.value,
            "mode": self.name,
            "candidate_id": candidate.id,
            "aggregate_score": round(aggregate_score, 6),
            "pass_rate": round(pass_rate, 6),
            "threshold": suite.pass_threshold,
            "repo_root": str(repo_root),
            "target_path": str(source_path),
            "execution": {"id": execution.id, "status": execution.status.value},
        }

    def _repo_root_for_target(self, *, ledger: Any, target: Any) -> Path:
        input_id = str(target.metadata.get("input_id", "")).strip()
        if input_id:
            try:
                item = ledger.get_input(input_id)
                source_path = str(item.source_path or "").strip()
                if source_path:
                    candidate = Path(source_path).expanduser().resolve()
                    if candidate.exists() and candidate.is_dir():
                        return candidate
            except KeyError:
                pass
        if target.source_path:
            return Path(target.source_path).expanduser().resolve().parent
        raise RepoBenchmarkError("Repo benchmark target has no resolvable repository root.")

    def _run_case_command(
        self,
        *,
        orchestrator: Any,
        case: dict[str, Any],
        repo_root: Path,
    ) -> Any:
        command_text = self._case_command_string(case)
        if not command_text:
            raise RepoBenchmarkError("Repo benchmark case is missing a `command` field.")
        cwd_value = str(case.get("cwd", "")).strip()
        cwd = repo_root / cwd_value if cwd_value else repo_root
        cwd = cwd.expanduser().resolve()
        if not cwd.exists() or not cwd.is_dir():
            raise RepoBenchmarkError(f"Repo benchmark cwd does not exist: {cwd}")
        shell_mode = bool(case.get("shell", False))
        args = ["/bin/sh", "-lc", command_text] if shell_mode else shlex.split(command_text)
        env = orchestrator.config.subprocess_env()
        extra_env = case.get("env", {})
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                env[str(key)] = str(value)
        env.setdefault("PYTHONPATH", str(repo_root))
        timeout_seconds = float(case.get("timeout_seconds", 600))
        from ..orchestrator import ExternalCommand

        return orchestrator.command_runner(
            ExternalCommand(
                args=args,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
                capture_output=True,
            )
        )

    def _parse_case_metrics(
        self,
        *,
        case: dict[str, Any],
        result: Any,
        repo_root: Path,
    ) -> dict[str, float]:
        parser = str(case.get("parser", "json_stdout")).strip() or "json_stdout"
        if parser == "json_stdout":
            payload = result.stdout
        elif parser == "json_file":
            output_file = str(case.get("output_file", "")).strip()
            if not output_file:
                raise RepoBenchmarkError("json_file parser requires `output_file`.")
            path = (repo_root / output_file).expanduser().resolve()
            if not path.exists():
                raise RepoBenchmarkError(f"Repo benchmark metrics file does not exist: {path}")
            payload = path.read_text(encoding="utf-8")
        else:
            raise RepoBenchmarkError(f"Unsupported repo benchmark parser: {parser}")

        try:
            parsed = json.loads(payload or "{}")
        except json.JSONDecodeError as exc:
            raise RepoBenchmarkError("Repo benchmark command did not emit valid JSON metrics.") from exc
        if not isinstance(parsed, dict):
            raise RepoBenchmarkError("Repo benchmark metrics payload must be a JSON object.")

        metrics: dict[str, float] = {}
        for key, value in parsed.items():
            if isinstance(value, bool):
                metrics[str(key)] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                metrics[str(key)] = float(value)
        return metrics

    def _evaluate_checks(
        self,
        *,
        checks: list[dict[str, Any]],
        metrics: dict[str, float],
        baseline_metrics: dict[str, float],
    ) -> tuple[float, bool, list[dict[str, Any]]]:
        if not checks:
            return 1.0, True, []
        results: list[dict[str, Any]] = []
        passed_count = 0
        for check in checks:
            check_type = str(check.get("type", "")).strip() or str(check.get("comparison", "")).strip()
            metric_key = str(check.get("key", "")).strip()
            actual = metrics.get(metric_key)
            baseline_key = str(check.get("baseline_key", "")).strip() or metric_key
            baseline = baseline_metrics.get(baseline_key)
            threshold = check.get("value", check.get("threshold"))
            passed = self._evaluate_check(
                check_type=check_type,
                actual=actual,
                baseline=baseline,
                threshold=threshold,
            )
            results.append(
                {
                    "type": check_type,
                    "key": metric_key,
                    "actual": actual,
                    "baseline_key": baseline_key,
                    "baseline": baseline,
                    "threshold": threshold,
                    "passed": passed,
                }
            )
            if passed:
                passed_count += 1
        score = passed_count / len(checks)
        return score, passed_count == len(checks), results

    @staticmethod
    def _evaluate_check(
        *,
        check_type: str,
        actual: float | None,
        baseline: float | None,
        threshold: Any,
    ) -> bool:
        if actual is None:
            return False
        kind = check_type.strip().lower()
        threshold_value = float(threshold) if threshold is not None else 0.0
        if kind == "gte":
            return actual >= threshold_value
        if kind == "lte":
            return actual <= threshold_value
        if kind == "gt":
            return actual > threshold_value
        if kind == "lt":
            return actual < threshold_value
        if kind == "eq":
            return abs(actual - threshold_value) <= 1e-9
        if baseline is None:
            return False
        if kind == "baseline_delta_gte":
            return (actual - baseline) >= threshold_value
        if kind == "baseline_delta_lte":
            return (actual - baseline) <= threshold_value
        if kind == "baseline_ratio_lte":
            return False if baseline == 0 else (actual / baseline) <= threshold_value
        if kind == "baseline_ratio_gte":
            return False if baseline == 0 else (actual / baseline) >= threshold_value
        if kind == "baseline_reduction_gte":
            return False if baseline == 0 else ((baseline - actual) / baseline) >= threshold_value
        return False

    @staticmethod
    def _case_command_string(case: dict[str, Any]) -> str:
        value = case.get("command", "")
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return str(value).strip()

    @staticmethod
    def _case_role(case: dict[str, Any]) -> str:
        explicit = str(case.get("role", "")).strip().lower()
        if explicit in {"baseline", "candidate"}:
            return explicit
        case_id = str(case.get("id", "")).strip().lower()
        return "baseline" if case_id == "baseline" else "candidate"

    def _aggregate_scores(self, *, run_scores: list[float], suite: Any) -> float:
        return aggregate_scores(run_scores, suite.aggregation)

    @staticmethod
    def _python_command(config: Any) -> str:
        launcher = config.env.get("AIEQ_REPO_BENCHMARK_PYTHON", "").strip()
        if launcher:
            return launcher
        for candidate in ("python3", "python"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return ""

    @staticmethod
    def _candidate_stats(*, candidates: list[Any], eval_runs: list[Any]) -> list[dict[str, Any]]:
        filtered_runs = [
            item for item in eval_runs if item.metadata.get("include_in_aggregate", True)
        ]
        return SkillOptimizerMode._candidate_stats(candidates=candidates, eval_runs=filtered_runs)
