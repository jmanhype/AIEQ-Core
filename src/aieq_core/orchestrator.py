from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any, Callable

from .adapters.autoresearch import AutoresearchAdapter
from .adapters.denario import DenarioAdapter
from .controller import ResearchController
from .ledger import EpistemicLedger
from .method_bridge import (
    MethodBridgeDraft,
    MethodBridgeError,
    MethodBridgeReview,
    OpenAIMethodBridge,
)
from .modes import default_mode_registry
from .models import ActionProposal, ActionType, ExecutionStatus, serialize_dataclass
from .runtime import RuntimeConfig, capability_key_for_action, doctor_report


@dataclass(slots=True)
class ExternalCommand:
    args: list[str]
    cwd: Path
    env: dict[str, str]
    timeout_seconds: float | None = None
    capture_output: bool = True
    stdout_path: str = ""
    stderr_path: str = ""
    combine_output: bool = False


@dataclass(slots=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class UnsupportedAutomatedActionError(RuntimeError):
    """Raised when the controller requests an action not yet automated."""


CommandRunner = Callable[[ExternalCommand], CommandResult]


class ResearchOrchestrator:
    """Single entrypoint for capability checks and guarded external execution."""

    def __init__(
        self,
        *,
        config: RuntimeConfig | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config or RuntimeConfig.load()
        self.command_runner = command_runner or self._default_command_runner
        self.mode_registry = default_mode_registry()

    def doctor(
        self,
        *,
        ledger_path: str | Path | None = None,
        mode: str = "",
    ) -> dict[str, Any]:
        if mode:
            return self.mode_registry.get(mode).doctor(
                config=self.config,
                ledger_path=str(ledger_path) if ledger_path is not None else None,
            )

        payload = doctor_report(self.config, ledger_path=ledger_path)
        payload["modes"] = [serialize_dataclass(item) for item in self.mode_registry.list_modes()]
        payload["mode_reports"] = {
            descriptor.name: self.mode_registry.get(descriptor.name).doctor(
                config=self.config,
                ledger_path=str(ledger_path) if ledger_path is not None else None,
            )
            for descriptor in self.mode_registry.list_modes()
            if descriptor.name != "ml_research"
        }
        return payload

    def run_next(
        self,
        ledger_path: str | Path,
        *,
        backlog_limit: int = 5,
        dry_run: bool = False,
        data_description: str = "",
        data_description_file: str = "",
        mode: str = "",
    ) -> dict[str, Any]:
        self.config.ensure_runtime_dirs()
        ledger = EpistemicLedger.load(ledger_path)
        decision = ResearchController(mode_registry=self.mode_registry).decide(
            ledger,
            backlog_limit=backlog_limit,
            mode_hint=mode,
        )
        payload: dict[str, Any] = {
            "ok": True,
            "decision": serialize_dataclass(decision),
        }

        if dry_run:
            payload["mode"] = "dry_run"
            payload["doctor"] = self.doctor(ledger_path=ledger_path, mode=mode)
            return payload

        decision_record = ledger.record_decision(
            decision.primary_action,
            metadata={
                "queue_state": decision.queue_state,
                "summary": decision.summary,
                "backlog": [serialize_dataclass(item) for item in decision.backlog],
                "runner": "orchestrator",
                "mode": decision.primary_action.mode or mode,
            },
        )
        payload["decision_record"] = serialize_dataclass(decision_record)

        try:
            result = self._execute_action(
                ledger=ledger,
                proposal=decision.primary_action,
                decision_id=decision_record.id,
                data_description=data_description,
                data_description_file=data_description_file,
                mode_hint=mode,
            )
            payload["result"] = result
            payload["ok"] = bool(result.get("ok", True))
        except UnsupportedAutomatedActionError as exc:
            execution = ledger.record_execution(
                decision_id=decision_record.id,
                status=ExecutionStatus.SKIPPED,
                notes=str(exc),
                metadata={"runner": "orchestrator", "status": "unsupported"},
            )
            payload["skipped"] = True
            payload["execution"] = serialize_dataclass(execution)
        except Exception as exc:  # pragma: no cover - defensive fallback
            execution = ledger.record_execution(
                decision_id=decision_record.id,
                status=ExecutionStatus.FAILED,
                notes=str(exc),
                metadata={"runner": "orchestrator", "status": "exception"},
            )
            payload["ok"] = False
            payload["error"] = str(exc)
            payload["execution"] = serialize_dataclass(execution)

        payload["follow_up_decision"] = serialize_dataclass(
            ResearchController(mode_registry=self.mode_registry).decide(
                ledger,
                backlog_limit=backlog_limit,
                mode_hint=mode,
            )
        )
        return payload

    def _execute_action(
        self,
        *,
        ledger: EpistemicLedger,
        proposal: ActionProposal,
        decision_id: str,
        data_description: str,
        data_description_file: str,
        mode_hint: str,
    ) -> dict[str, Any]:
        if proposal.claim_id:
            claim = ledger.get_claim(proposal.claim_id)
            adapter = self.mode_registry.for_claim(claim)
        else:
            adapter = self.mode_registry.get(proposal.mode or mode_hint or "ml_research")
        return adapter.execute_action(
            orchestrator=self,
            ledger=ledger,
            proposal=proposal,
            decision_id=decision_id,
            data_description=data_description,
            data_description_file=data_description_file,
        )

    @staticmethod
    def unsupported_action_error(message: str) -> UnsupportedAutomatedActionError:
        return UnsupportedAutomatedActionError(message)

    def _execute_generate_idea(
        self,
        *,
        ledger: EpistemicLedger,
        proposal: ActionProposal,
        decision_id: str,
        data_description: str,
        data_description_file: str,
    ) -> dict[str, Any]:
        execution_dir = self.config.execution_dir(decision_id)
        project_dir = self.config.denario_project_dir_for_decision(
            decision_id=decision_id,
            claim_id=proposal.claim_id,
            claim_title=proposal.claim_title,
        )
        project_dir.parent.mkdir(parents=True, exist_ok=True)

        data_input = self._resolve_data_description_input(
            project_dir=project_dir,
            data_description=data_description,
            data_description_file=data_description_file,
        )
        spec = {
            "task": "generate_idea",
            "project_dir": str(project_dir),
            "mode": self.config.denario_mode,
            "idea_llm": self.config.denario_idea_llm,
            "data_description_input": data_input,
        }
        command, spec_path = self._build_denario_command(spec=spec, execution_dir=execution_dir)

        started = time.monotonic()
        result = self.command_runner(command)
        runtime_seconds = round(time.monotonic() - started, 3)
        stdout_log, stderr_log = self._persist_command_logs(
            execution_dir=execution_dir,
            stdout_text=result.stdout,
            stderr_text=result.stderr,
            prefix="denario",
        )

        if result.returncode != 0:
            execution = ledger.record_execution(
                decision_id=decision_id,
                status=ExecutionStatus.FAILED,
                notes="Denario generate_idea failed.",
                runtime_seconds=runtime_seconds,
                artifact_paths=[stdout_log, stderr_log, str(spec_path)],
                metadata={"runner": "orchestrator", "project_dir": str(project_dir)},
            )
            return {
                "ok": False,
                "action": "generate_idea",
                "project_dir": str(project_dir),
                "spec_path": str(spec_path),
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "execution": serialize_dataclass(execution),
            }

        imported = DenarioAdapter.import_project(
            ledger=ledger,
            project_dir=project_dir,
            decision_id=decision_id,
            execution_status=ExecutionStatus.SUCCEEDED,
            runtime_seconds=runtime_seconds,
        )
        return {
            "ok": True,
            "action": "generate_idea",
            "project_dir": str(project_dir),
            "spec_path": str(spec_path),
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "imported": imported,
        }

    def _execute_generate_method(
        self,
        *,
        ledger: EpistemicLedger,
        proposal: ActionProposal,
        decision_id: str,
    ) -> dict[str, Any]:
        claim = ledger.get_claim(proposal.claim_id)
        denario_meta = claim.metadata.get("denario", {})
        project_dir = Path(str(denario_meta.get("project_dir", "")).strip())
        if not project_dir.exists():
            raise UnsupportedAutomatedActionError(
                f"Claim {claim.id} has no usable Denario project directory."
            )

        execution_dir = self.config.execution_dir(decision_id)
        spec = {
            "task": "generate_method",
            "project_dir": str(project_dir),
            "mode": self.config.denario_mode,
            "method_llm": self.config.denario_method_llm,
        }
        command, spec_path = self._build_denario_command(spec=spec, execution_dir=execution_dir)

        started = time.monotonic()
        result = self.command_runner(command)
        runtime_seconds = round(time.monotonic() - started, 3)
        stdout_log, stderr_log = self._persist_command_logs(
            execution_dir=execution_dir,
            stdout_text=result.stdout,
            stderr_text=result.stderr,
            prefix="denario",
        )

        if result.returncode != 0:
            execution = ledger.record_execution(
                decision_id=decision_id,
                status=ExecutionStatus.FAILED,
                notes="Denario generate_method failed.",
                runtime_seconds=runtime_seconds,
                artifact_paths=[stdout_log, stderr_log, str(spec_path)],
                metadata={"runner": "orchestrator", "project_dir": str(project_dir)},
            )
            return {
                "ok": False,
                "action": "generate_method",
                "project_dir": str(project_dir),
                "spec_path": str(spec_path),
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "execution": serialize_dataclass(execution),
            }

        imported = DenarioAdapter.import_project(
            ledger=ledger,
            project_dir=project_dir,
            claim_id=claim.id,
            decision_id=decision_id,
            execution_status=ExecutionStatus.SUCCEEDED,
            runtime_seconds=runtime_seconds,
        )
        return {
            "ok": True,
            "action": "generate_method",
            "project_dir": str(project_dir),
            "spec_path": str(spec_path),
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "imported": imported,
        }

    def _execute_synthesize_paper(
        self,
        *,
        ledger: EpistemicLedger,
        proposal: ActionProposal,
        decision_id: str,
    ) -> dict[str, Any]:
        claim = ledger.get_claim(proposal.claim_id)
        denario_meta = claim.metadata.get("denario", {})
        project_dir = Path(str(denario_meta.get("project_dir", "")).strip())
        if not project_dir.exists():
            raise UnsupportedAutomatedActionError(
                f"Claim {claim.id} has no usable Denario project directory."
            )

        execution_dir = self.config.execution_dir(decision_id)
        spec = {
            "task": "synthesize_paper",
            "project_dir": str(project_dir),
            "paper_llm": self.config.denario_paper_llm,
            "journal": self.config.denario_paper_journal,
        }
        command, spec_path = self._build_denario_command(spec=spec, execution_dir=execution_dir)

        started = time.monotonic()
        result = self.command_runner(command)
        runtime_seconds = round(time.monotonic() - started, 3)
        stdout_log, stderr_log = self._persist_command_logs(
            execution_dir=execution_dir,
            stdout_text=result.stdout,
            stderr_text=result.stderr,
            prefix="denario",
        )

        if result.returncode != 0:
            execution = ledger.record_execution(
                decision_id=decision_id,
                status=ExecutionStatus.FAILED,
                notes="Denario synthesize_paper failed.",
                runtime_seconds=runtime_seconds,
                artifact_paths=[stdout_log, stderr_log, str(spec_path)],
                metadata={"runner": "orchestrator", "project_dir": str(project_dir)},
            )
            return {
                "ok": False,
                "action": "synthesize_paper",
                "project_dir": str(project_dir),
                "spec_path": str(spec_path),
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "execution": serialize_dataclass(execution),
            }

        imported = DenarioAdapter.import_project(
            ledger=ledger,
            project_dir=project_dir,
            claim_id=claim.id,
            decision_id=decision_id,
            execution_status=ExecutionStatus.SUCCEEDED,
            runtime_seconds=runtime_seconds,
        )
        return {
            "ok": True,
            "action": "synthesize_paper",
            "project_dir": str(project_dir),
            "spec_path": str(spec_path),
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
            "imported": imported,
        }

    def _execute_autoresearch(
        self,
        *,
        ledger: EpistemicLedger,
        proposal: ActionProposal,
        decision_id: str,
    ) -> dict[str, Any]:
        claim = ledger.get_claim(proposal.claim_id)
        branch = self._resolve_autoresearch_branch(claim)
        execution_dir = self.config.execution_dir(decision_id)
        remote_host = self.config.autoresearch_remote_host if self.config.use_remote_autoresearch() else ""
        bridge_metadata: dict[str, Any] | None = None
        bridge_artifact_paths: list[str] = []
        original_train_py = ""
        final_run_log = execution_dir / "autoresearch-run.log"
        total_runtime_seconds = 0.0

        method_artifact = self._latest_method_artifact_for_claim(ledger=ledger, claim_id=claim.id)
        try:
            if method_artifact is not None:
                original_train_py = self._read_autoresearch_train_py()
                bridge = self._build_method_bridge(
                    claim=claim,
                    method_artifact=method_artifact,
                    current_train_py=original_train_py,
                )
                bridge_attempts: list[dict[str, Any]] = []
                result: CommandResult | None = None
                runtime_error_excerpt = ""
                max_attempts = 2

                for attempt_index in range(1, max_attempts + 1):
                    review = self._review_method_bridge(
                        claim=claim,
                        method_artifact=method_artifact,
                        current_train_py=original_train_py,
                        bridge=bridge,
                    )
                    attempt_metadata, attempt_artifacts = self._persist_method_bridge_artifacts(
                        execution_dir=execution_dir,
                        method_artifact=method_artifact,
                        original_train_py=original_train_py,
                        bridge=bridge,
                        review=review,
                        attempt_index=attempt_index,
                        runtime_error_excerpt=runtime_error_excerpt,
                    )
                    bridge_attempts.append(attempt_metadata)
                    bridge_artifact_paths.extend(attempt_artifacts)

                    if not review.approved:
                        bridge_metadata = self._combine_bridge_attempts(bridge_attempts)
                        blocker_suffix = (
                            f" Blockers: {'; '.join(review.blockers)}."
                            if review.blockers
                            else ""
                        )
                        notes = (
                            "Bridge review rejected execution before launch: "
                            f"{review.summary}.{blocker_suffix}"
                        )
                        execution = ledger.record_execution(
                            decision_id=decision_id,
                            status=ExecutionStatus.SKIPPED,
                            notes=notes,
                            artifact_paths=bridge_artifact_paths,
                            metadata={
                                "runner": "orchestrator",
                                "status": "bridge_review_rejected",
                                "bridge": bridge_metadata or {},
                            },
                        )
                        return {
                            "ok": False,
                            "action": proposal.action_type.value,
                            "branch": branch,
                            "mode": "remote" if remote_host else "local",
                            "host": remote_host,
                            "blocked": "bridge_review_rejected",
                            "bridge": bridge_metadata,
                            "execution": serialize_dataclass(execution),
                        }

                    self._write_autoresearch_train_py(bridge.train_py)

                    attempt_run_log = self._attempt_run_log_path(
                        execution_dir=execution_dir,
                        attempt_index=attempt_index,
                    )
                    result, attempt_runtime_seconds = self._run_autoresearch_once(
                        remote_host=remote_host,
                        run_log=attempt_run_log,
                    )
                    total_runtime_seconds += attempt_runtime_seconds
                    bridge_artifact_paths.append(str(attempt_run_log))
                    final_run_log = attempt_run_log

                    if result.returncode == 0 or attempt_index == max_attempts:
                        break

                    runtime_error_excerpt = self._runtime_failure_excerpt(attempt_run_log)
                    bridge = self._repair_method_bridge_after_runtime_failure(
                        claim=claim,
                        method_artifact=method_artifact,
                        failed_bridge=bridge,
                        runtime_error_excerpt=runtime_error_excerpt,
                    )

                if result is None:
                    raise RuntimeError("Autoresearch bridge loop failed to execute any attempts.")
                bridge_metadata = self._combine_bridge_attempts(bridge_attempts)
            else:
                result, total_runtime_seconds = self._run_autoresearch_once(
                    remote_host=remote_host,
                    run_log=final_run_log,
                )
        finally:
            if original_train_py:
                self._write_autoresearch_train_py(original_train_py)

        runtime_seconds = round(total_runtime_seconds, 3)
        commit = self._autoresearch_revision()
        description = f"{proposal.action_type.value} for {claim.title}"
        baseline_bpb = self._baseline_bpb_for_branch(claim=claim, branch=branch)
        results_tsv_path = self._results_tsv_path_for_claim(claim=claim, branch=branch)

        if result.returncode != 0:
            self._append_results_row(
                results_tsv_path,
                {
                    "commit": commit or "local",
                    "val_bpb": "0.000000",
                    "memory_gb": "0.0",
                    "status": "crash",
                    "description": description,
                },
            )
            imported_run = serialize_dataclass(
                self._annotate_autoresearch_import(
                    ledger=ledger,
                    evidence=AutoresearchAdapter.import_run(
                        ledger=ledger,
                        claim_id=claim.id,
                        run_log_path=final_run_log,
                        commit=commit,
                        branch=branch,
                        description=description,
                        status="crash",
                        artifact_paths=bridge_artifact_paths,
                        decision_id=decision_id,
                        execution_status=ExecutionStatus.FAILED,
                    ),
                    bridge_metadata=bridge_metadata,
                )
            )
            imported_series = AutoresearchAdapter.import_results_tsv(
                ledger=ledger,
                claim_id=claim.id,
                results_tsv_path=results_tsv_path,
                branch=branch,
                baseline_bpb=baseline_bpb,
            )
            execution_id = imported_run.get("metadata", {}).get("execution_id", "")
            execution = (
                serialize_dataclass(ledger.executions[execution_id])
                if execution_id and execution_id in ledger.executions
                else None
            )
            return {
                "ok": False,
                "action": proposal.action_type.value,
                "branch": branch,
                "mode": "remote" if remote_host else "local",
                "host": remote_host,
                "run_log_path": str(final_run_log),
                "results_tsv_path": str(results_tsv_path),
                "bridge": bridge_metadata,
                "imported_run": imported_run,
                "imported_series": imported_series,
                "execution": execution,
            }

        parsed_run = AutoresearchAdapter.parse_run_log(final_run_log)
        status = self._infer_autoresearch_status(
            baseline_bpb=baseline_bpb,
            val_bpb=parsed_run.val_bpb,
        )
        self._append_results_row(
            results_tsv_path,
            {
                "commit": commit or "local",
                "val_bpb": f"{parsed_run.val_bpb:.6f}",
                "memory_gb": f"{parsed_run.peak_vram_gb:.1f}",
                "status": status,
                "description": description,
            },
        )
        imported_run = serialize_dataclass(
            self._annotate_autoresearch_import(
                ledger=ledger,
                evidence=AutoresearchAdapter.import_run(
                    ledger=ledger,
                    claim_id=claim.id,
                    run_log_path=final_run_log,
                    commit=commit,
                    branch=branch,
                    description=description,
                    status=status,
                    baseline_bpb=baseline_bpb,
                    artifact_paths=bridge_artifact_paths,
                    decision_id=decision_id,
                    execution_status=ExecutionStatus.SUCCEEDED,
                ),
                bridge_metadata=bridge_metadata,
            )
        )
        imported_series = AutoresearchAdapter.import_results_tsv(
            ledger=ledger,
            claim_id=claim.id,
            results_tsv_path=results_tsv_path,
            branch=branch,
            baseline_bpb=baseline_bpb,
        )
        execution_id = imported_run.get("metadata", {}).get("execution_id", "")
        execution = (
            serialize_dataclass(ledger.executions[execution_id])
            if execution_id and execution_id in ledger.executions
            else None
        )
        return {
            "ok": True,
            "action": proposal.action_type.value,
            "branch": branch,
            "mode": "remote" if remote_host else "local",
            "host": remote_host,
            "commit": commit,
            "status": status,
            "runtime_seconds": runtime_seconds,
            "run_log_path": str(final_run_log),
            "results_tsv_path": str(results_tsv_path),
            "bridge": bridge_metadata,
            "imported_run": imported_run,
            "imported_series": imported_series,
            "execution": execution,
        }

    def _build_denario_command(
        self,
        *,
        spec: dict[str, Any],
        execution_dir: Path,
    ) -> tuple[ExternalCommand, Path]:
        launcher = self.config.launcher_for_project(self.config.denario_repo)
        if not launcher.available:
            raise UnsupportedAutomatedActionError(launcher.detail)

        spec_path = execution_dir / "denario-spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        task_script = Path(__file__).with_name("_denario_task.py")
        command = ExternalCommand(
            args=[*launcher.command_prefix, str(task_script), str(spec_path)],
            cwd=self.config.denario_repo,
            env=self.config.subprocess_env(),
            timeout_seconds=self.config.denario_timeout_seconds,
            capture_output=True,
        )
        return command, spec_path

    def _resolve_data_description_input(
        self,
        *,
        project_dir: Path,
        data_description: str,
        data_description_file: str,
    ) -> str:
        if data_description_file:
            return str(Path(data_description_file).expanduser().resolve())
        if data_description:
            return data_description

        configured = self.config.default_data_description_path()
        if configured is not None and configured.exists():
            return str(configured)

        existing = project_dir / "input_files" / "data_description.md"
        if existing.exists():
            return str(existing)

        raise UnsupportedAutomatedActionError(
            "generate_idea requires AIEQ_DATA_DESCRIPTION_FILE, --data-description-file, "
            "or --data-description."
        )

    def _resolve_autoresearch_branch(self, claim: Any) -> str:
        autoresearch_meta = claim.metadata.get("autoresearch", {})
        if not isinstance(autoresearch_meta, dict):
            autoresearch_meta = {}
        branch = str(autoresearch_meta.get("branch", "")).strip()
        return branch or self.config.default_autoresearch_branch

    def _baseline_bpb_for_branch(self, *, claim: Any, branch: str) -> float | None:
        autoresearch_meta = claim.metadata.get("autoresearch", {})
        if not isinstance(autoresearch_meta, dict):
            return None

        series_by_branch = autoresearch_meta.get("series_by_branch", {})
        if isinstance(series_by_branch, dict):
            entry = series_by_branch.get(branch, {})
            if isinstance(entry, dict):
                series = entry.get("series", {})
                if isinstance(series, dict):
                    value = series.get("best_val_bpb")
                    if value is not None:
                        return float(value)

        series = autoresearch_meta.get("series", {})
        if isinstance(series, dict) and str(autoresearch_meta.get("branch", "")).strip() == branch:
            value = series.get("best_val_bpb")
            if value is not None:
                return float(value)
        return None

    def _results_tsv_path_for_claim(self, *, claim: Any, branch: str) -> Path:
        autoresearch_meta = claim.metadata.get("autoresearch", {})
        if isinstance(autoresearch_meta, dict):
            series_by_branch = autoresearch_meta.get("series_by_branch", {})
            if isinstance(series_by_branch, dict):
                entry = series_by_branch.get(branch, {})
                if isinstance(entry, dict):
                    stored_path = str(entry.get("results_tsv_path", "")).strip()
                    if stored_path:
                        return Path(stored_path)

            if str(autoresearch_meta.get("branch", "")).strip() == branch:
                stored_path = str(autoresearch_meta.get("results_tsv_path", "")).strip()
                if stored_path:
                    return Path(stored_path)

        return self.config.autoresearch_results_tsv_for_claim(claim_id=claim.id, branch=branch)

    @staticmethod
    def _infer_autoresearch_status(*, baseline_bpb: float | None, val_bpb: float) -> str:
        if baseline_bpb is None:
            return "keep"
        return "keep" if val_bpb < baseline_bpb - 1e-4 else "discard"

    def _autoresearch_command(self) -> list[str]:
        if self.config.use_remote_autoresearch():
            remote_repo = self.config.remote_shell_path(self.config.autoresearch_remote_repo)
            remote_script = f"set -e; cd {remote_repo}; uv run train.py"
            return self.config.remote_ssh_command(f"bash -lc {shlex.quote(remote_script)}")

        launcher = self.config.launcher_for_project(self.config.autoresearch_repo)
        if not launcher.available:
            raise UnsupportedAutomatedActionError(launcher.detail)
        return [*launcher.command_prefix, "train.py"]

    def _latest_method_artifact_for_claim(
        self,
        *,
        ledger: EpistemicLedger,
        claim_id: str,
    ) -> Any | None:
        method_artifacts = [
            artifact
            for artifact in ledger.artifacts_for_claim(claim_id)
            if artifact.kind.value == "method"
        ]
        if not method_artifacts:
            return None
        return sorted(method_artifacts, key=lambda item: item.updated_at)[-1]

    def _build_method_bridge(
        self,
        *,
        claim: Any,
        method_artifact: Any,
        current_train_py: str,
    ) -> MethodBridgeDraft:
        if not self.config.method_bridge_enabled:
            raise UnsupportedAutomatedActionError(
                "Method bridge is disabled for run_experiment."
            )

        api_key = self.config.env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise UnsupportedAutomatedActionError(
                "Method bridge requires OPENAI_API_KEY."
            )

        method_text = self._resolve_method_text(method_artifact)
        bridge = self._method_bridge_client(api_key=api_key)
        try:
            return bridge.generate(
                claim_title=claim.title,
                claim_statement=claim.statement,
                method_text=method_text,
                current_train_py=current_train_py,
            )
        except MethodBridgeError as exc:
            raise UnsupportedAutomatedActionError(str(exc)) from exc

    def _review_method_bridge(
        self,
        *,
        claim: Any,
        method_artifact: Any,
        current_train_py: str,
        bridge: MethodBridgeDraft,
    ) -> MethodBridgeReview:
        if not self.config.method_bridge_enabled:
            raise UnsupportedAutomatedActionError(
                "Method bridge is disabled for run_experiment."
            )

        api_key = self.config.env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise UnsupportedAutomatedActionError(
                "Method bridge requires OPENAI_API_KEY."
            )

        method_text = self._resolve_method_text(method_artifact)
        reviewer = self._method_bridge_client(api_key=api_key)
        try:
            return reviewer.review(
                claim_title=claim.title,
                claim_statement=claim.statement,
                method_text=method_text,
                current_train_py=current_train_py,
                generated_train_py=bridge.train_py,
                generated_summary=bridge.summary,
            )
        except MethodBridgeError as exc:
            raise UnsupportedAutomatedActionError(str(exc)) from exc

    def _method_bridge_client(self, *, api_key: str) -> OpenAIMethodBridge:
        return OpenAIMethodBridge(
            api_key=api_key,
            model=self.config.method_bridge_model,
            timeout_seconds=self.config.method_bridge_timeout_seconds,
            base_url=self.config.env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def _repair_method_bridge_after_runtime_failure(
        self,
        *,
        claim: Any,
        method_artifact: Any,
        failed_bridge: MethodBridgeDraft,
        runtime_error_excerpt: str,
    ) -> MethodBridgeDraft:
        if not self.config.method_bridge_enabled:
            raise UnsupportedAutomatedActionError(
                "Method bridge is disabled for run_experiment."
            )

        api_key = self.config.env.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise UnsupportedAutomatedActionError(
                "Method bridge requires OPENAI_API_KEY."
            )

        method_text = self._resolve_method_text(method_artifact)
        bridge = self._method_bridge_client(api_key=api_key)
        try:
            return bridge.repair_runtime_failure(
                claim_title=claim.title,
                claim_statement=claim.statement,
                method_text=method_text,
                previous_train_py=failed_bridge.train_py,
                runtime_error=runtime_error_excerpt,
                previous_summary=failed_bridge.summary,
            )
        except MethodBridgeError as exc:
            raise UnsupportedAutomatedActionError(str(exc)) from exc

    @staticmethod
    def _resolve_method_text(method_artifact: Any) -> str:
        source_path = str(method_artifact.source_path).strip()
        if source_path:
            path = Path(source_path)
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        return str(method_artifact.content).strip()

    def _persist_method_bridge_artifacts(
        self,
        *,
        execution_dir: Path,
        method_artifact: Any,
        original_train_py: str,
        bridge: MethodBridgeDraft,
        review: MethodBridgeReview | None = None,
        attempt_index: int = 1,
        runtime_error_excerpt: str = "",
    ) -> tuple[dict[str, Any], list[str]]:
        suffix = "" if attempt_index == 1 else f".attempt-{attempt_index}"
        prompt_path = execution_dir / f"train-bridge{suffix}.prompt.txt"
        response_path = execution_dir / f"train-bridge{suffix}.response.json"
        original_path = execution_dir / f"train-bridge{suffix}.original.py"
        bridged_path = execution_dir / f"train-bridge{suffix}.generated.py"
        review_prompt_path = execution_dir / f"train-bridge{suffix}.review.prompt.txt"
        review_response_path = execution_dir / f"train-bridge{suffix}.review.response.json"
        runtime_error_path = execution_dir / f"train-bridge{suffix}.runtime-error.txt"

        prompt_path.write_text(bridge.prompt, encoding="utf-8")
        response_path.write_text(
            json.dumps(bridge.raw_response, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        original_path.write_text(original_train_py, encoding="utf-8")
        bridged_path.write_text(bridge.train_py, encoding="utf-8")
        artifact_paths = [
            str(prompt_path),
            str(response_path),
            str(original_path),
            str(bridged_path),
        ]
        if review is not None:
            review_prompt_path.write_text(review.prompt, encoding="utf-8")
            review_response_path.write_text(
                json.dumps(review.raw_response, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifact_paths.extend([str(review_prompt_path), str(review_response_path)])
        if runtime_error_excerpt:
            runtime_error_path.write_text(runtime_error_excerpt, encoding="utf-8")
            artifact_paths.append(str(runtime_error_path))

        metadata = {
            "applied": True,
            "attempt": attempt_index,
            "model": bridge.model,
            "response_id": bridge.response_id,
            "summary": bridge.summary,
            "usage": bridge.usage,
            "method_artifact_id": str(method_artifact.id),
            "method_source_path": str(method_artifact.source_path),
            "prompt_path": str(prompt_path),
            "response_path": str(response_path),
            "original_train_path": str(original_path),
            "generated_train_path": str(bridged_path),
            "repair_source": "runtime_failure" if runtime_error_excerpt else "initial",
        }
        if review is not None:
            metadata["review"] = {
                **review.as_metadata(),
                "prompt_path": str(review_prompt_path),
                "response_path": str(review_response_path),
            }
        if runtime_error_excerpt:
            metadata["runtime_error_path"] = str(runtime_error_path)
        return metadata, artifact_paths

    def _annotate_autoresearch_import(
        self,
        *,
        ledger: EpistemicLedger,
        evidence: Any,
        bridge_metadata: dict[str, Any] | None,
    ) -> Any:
        if not bridge_metadata:
            return evidence

        evidence.metadata["bridge"] = dict(bridge_metadata)
        execution_id = str(evidence.metadata.get("execution_id", "")).strip()
        if execution_id and execution_id in ledger.executions:
            ledger.executions[execution_id].metadata["bridge"] = dict(bridge_metadata)
        ledger.save()
        return evidence

    @staticmethod
    def _combine_bridge_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not attempts:
            return None
        combined = dict(attempts[-1])
        combined["attempt_count"] = len(attempts)
        combined["runtime_repair_applied"] = len(attempts) > 1
        combined["attempts"] = [dict(item) for item in attempts]
        return combined

    def _run_autoresearch_once(
        self,
        *,
        remote_host: str,
        run_log: Path,
    ) -> tuple[CommandResult, float]:
        started = time.monotonic()
        result = self.command_runner(
            ExternalCommand(
                args=self._autoresearch_command(),
                cwd=self.config.repo_root if remote_host else self.config.autoresearch_repo,
                env=self.config.subprocess_env(),
                timeout_seconds=self.config.autoresearch_timeout_seconds,
                capture_output=False,
                stdout_path=str(run_log),
                stderr_path=str(run_log),
                combine_output=True,
            )
        )
        return result, round(time.monotonic() - started, 3)

    @staticmethod
    def _attempt_run_log_path(
        *,
        execution_dir: Path,
        attempt_index: int,
    ) -> Path:
        if attempt_index == 1:
            return execution_dir / "autoresearch-run.log"
        return execution_dir / f"autoresearch-run.attempt-{attempt_index}.log"

    @staticmethod
    def _runtime_failure_excerpt(
        run_log: Path,
        *,
        line_limit: int = 120,
        char_limit: int = 12000,
    ) -> str:
        text = run_log.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        excerpt = "\n".join(lines[-line_limit:])
        if len(excerpt) > char_limit:
            excerpt = excerpt[-char_limit:]
        return excerpt.strip()

    def _read_autoresearch_train_py(self) -> str:
        if self.config.use_remote_autoresearch():
            remote_train = self._remote_autoresearch_train_path()
            script = (
                "import base64\n"
                "from pathlib import Path\n"
                f"data = Path({remote_train!r}).expanduser().read_bytes()\n"
                "print(base64.b64encode(data).decode('ascii'))\n"
            )
            completed = self._run_remote_python(script)
            encoded = completed.stdout.strip()
            return base64.b64decode(encoded.encode("ascii")).decode("utf-8")

        return (self.config.autoresearch_repo / "train.py").read_text(encoding="utf-8")

    def _write_autoresearch_train_py(self, source: str) -> None:
        if self.config.use_remote_autoresearch():
            remote_train = self._remote_autoresearch_train_path()
            encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
            script = (
                "import base64\n"
                "from pathlib import Path\n"
                f"path = Path({remote_train!r}).expanduser()\n"
                f"path.write_bytes(base64.b64decode({encoded!r}))\n"
            )
            self._run_remote_python(script)
            return

        (self.config.autoresearch_repo / "train.py").write_text(source, encoding="utf-8")

    def _run_remote_python(self, script: str) -> subprocess.CompletedProcess[str]:
        heredoc = f"python3 - <<'PY'\n{script}\nPY"
        completed = subprocess.run(
            self.config.remote_ssh_command(f"bash -lc {shlex.quote(heredoc)}"),
            cwd=self.config.repo_root,
            env=self.config.subprocess_env(),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "Remote Python helper failed.")
        return completed

    def _remote_autoresearch_train_path(self) -> str:
        return str(Path(self.config.autoresearch_remote_repo) / "train.py")

    def _autoresearch_revision(self) -> str:
        if self.config.use_remote_autoresearch():
            return self._git_short_revision_remote()
        return self._git_short_revision(self.config.autoresearch_repo)

    @staticmethod
    def _append_results_row(path: Path, row: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            if not file_exists:
                writer.writerow(["commit", "val_bpb", "memory_gb", "status", "description"])
            writer.writerow(
                [
                    row["commit"],
                    row["val_bpb"],
                    row["memory_gb"],
                    row["status"],
                    row["description"],
                ]
            )

    def _git_short_revision(self, repo_path: Path) -> str:
        launcher_env = self.config.subprocess_env()
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_path,
                env=launcher_env,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return ""

        revision = completed.stdout.strip()
        try:
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--", "train.py"],
                cwd=repo_path,
                env=launcher_env,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return revision

        return f"{revision}-dirty" if dirty.stdout.strip() else revision

    def _git_short_revision_remote(self) -> str:
        remote_repo = self.config.remote_shell_path(self.config.autoresearch_remote_repo)
        remote_script = f"set -e; cd {remote_repo}; git rev-parse --short HEAD; git status --porcelain -- train.py"
        try:
            completed = subprocess.run(
                self.config.remote_ssh_command(f"bash -lc {shlex.quote(remote_script)}"),
                cwd=self.config.repo_root,
                env=self.config.subprocess_env(),
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return ""

        if completed.returncode != 0:
            return ""

        lines = completed.stdout.splitlines()
        if not lines:
            return ""
        revision = lines[0].strip()
        dirty = any(line.strip() for line in lines[1:])
        return f"{revision}-dirty" if dirty else revision

    @staticmethod
    def _persist_command_logs(
        *,
        execution_dir: Path,
        stdout_text: str,
        stderr_text: str,
        prefix: str,
    ) -> tuple[str, str]:
        stdout_path = execution_dir / f"{prefix}.stdout.log"
        stderr_path = execution_dir / f"{prefix}.stderr.log"
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        return str(stdout_path), str(stderr_path)

    @staticmethod
    def _default_command_runner(command: ExternalCommand) -> CommandResult:
        if command.capture_output:
            completed = subprocess.run(
                command.args,
                cwd=command.cwd,
                env=command.env,
                timeout=command.timeout_seconds,
                check=False,
                capture_output=True,
                text=True,
            )
            return CommandResult(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        stdout_path = Path(command.stdout_path)
        stderr_path = Path(command.stderr_path or command.stdout_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        if command.combine_output:
            with stdout_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command.args,
                    cwd=command.cwd,
                    env=command.env,
                    timeout=command.timeout_seconds,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
                "w",
                encoding="utf-8",
            ) as stderr_handle:
                completed = subprocess.run(
                    command.args,
                    cwd=command.cwd,
                    env=command.env,
                    timeout=command.timeout_seconds,
                    check=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )

        return CommandResult(returncode=completed.returncode)
