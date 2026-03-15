from __future__ import annotations

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

    def doctor(self, *, ledger_path: str | Path | None = None) -> dict[str, Any]:
        return doctor_report(self.config, ledger_path=ledger_path)

    def run_next(
        self,
        ledger_path: str | Path,
        *,
        backlog_limit: int = 5,
        dry_run: bool = False,
        data_description: str = "",
        data_description_file: str = "",
    ) -> dict[str, Any]:
        self.config.ensure_runtime_dirs()
        ledger = EpistemicLedger.load(ledger_path)
        decision = ResearchController().decide(ledger, backlog_limit=backlog_limit)
        payload: dict[str, Any] = {
            "ok": True,
            "decision": serialize_dataclass(decision),
        }

        if dry_run:
            payload["mode"] = "dry_run"
            payload["doctor"] = self.doctor(ledger_path=ledger_path)
            return payload

        decision_record = ledger.record_decision(
            decision.primary_action,
            metadata={
                "queue_state": decision.queue_state,
                "summary": decision.summary,
                "backlog": [serialize_dataclass(item) for item in decision.backlog],
                "runner": "orchestrator",
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
            ResearchController().decide(ledger, backlog_limit=backlog_limit)
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
    ) -> dict[str, Any]:
        capability = self.doctor().get("capabilities", {}).get(
            capability_key_for_action(proposal.action_type),
            {},
        )
        if capability_key_for_action(proposal.action_type) == "manual_only":
            raise UnsupportedAutomatedActionError(
                f"{proposal.action_type.value} is not automated yet."
            )
        if not capability.get("available", False):
            blockers = capability.get("blocked_by", [])
            raise UnsupportedAutomatedActionError(
                f"{proposal.action_type.value} is blocked: {' '.join(blockers)}"
            )

        if proposal.action_type == ActionType.GENERATE_IDEA:
            return self._execute_generate_idea(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
                data_description=data_description,
                data_description_file=data_description_file,
            )
        if proposal.action_type == ActionType.GENERATE_METHOD:
            return self._execute_generate_method(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        if proposal.action_type == ActionType.SYNTHESIZE_PAPER:
            return self._execute_synthesize_paper(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )
        if proposal.action_type in {ActionType.RUN_EXPERIMENT, ActionType.REPRODUCE_RESULT}:
            return self._execute_autoresearch(
                ledger=ledger,
                proposal=proposal,
                decision_id=decision_id,
            )

        raise UnsupportedAutomatedActionError(
            f"{proposal.action_type.value} is not automated in the current execution plane."
        )

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
        run_log = execution_dir / "autoresearch-run.log"
        remote_host = self.config.autoresearch_remote_host if self.config.use_remote_autoresearch() else ""
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
        runtime_seconds = round(time.monotonic() - started, 3)

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
                AutoresearchAdapter.import_run(
                    ledger=ledger,
                    claim_id=claim.id,
                    run_log_path=run_log,
                    commit=commit,
                    branch=branch,
                    description=description,
                    status="crash",
                    decision_id=decision_id,
                    execution_status=ExecutionStatus.FAILED,
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
                "run_log_path": str(run_log),
                "results_tsv_path": str(results_tsv_path),
                "imported_run": imported_run,
                "imported_series": imported_series,
                "execution": execution,
            }

        parsed_run = AutoresearchAdapter.parse_run_log(run_log)
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
            AutoresearchAdapter.import_run(
                ledger=ledger,
                claim_id=claim.id,
                run_log_path=run_log,
                commit=commit,
                branch=branch,
                description=description,
                status=status,
                baseline_bpb=baseline_bpb,
                decision_id=decision_id,
                execution_status=ExecutionStatus.SUCCEEDED,
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
            "run_log_path": str(run_log),
            "results_tsv_path": str(results_tsv_path),
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
