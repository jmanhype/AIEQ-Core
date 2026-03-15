from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from ..ledger import EpistemicLedger
from ..models import (
    Evidence,
    EvidenceDirection,
    ExecutionStatus,
    clamp,
    serialize_dataclass,
)

KEY_MAP = {
    "val_bpb": "val_bpb",
    "training_seconds": "training_seconds",
    "total_seconds": "total_seconds",
    "peak_vram_mb": "peak_vram_mb",
    "mfu_percent": "mfu_percent",
    "total_tokens_M": "total_tokens_m",
    "num_steps": "num_steps",
    "num_params_M": "num_params_m",
    "depth": "depth",
}

INT_FIELDS = {"num_steps", "depth"}
RESULTS_TSV_FIELDS = ("commit", "val_bpb", "memory_gb", "status", "description")
SUMMARY_PATTERN = re.compile(r"^(?P<key>[A-Za-z_]+):\s+(?P<value>.+)$", re.MULTILINE)


@dataclass(slots=True)
class AutoresearchRun:
    val_bpb: float
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens_m: float
    num_steps: int
    num_params_m: float
    depth: int

    @property
    def peak_vram_gb(self) -> float:
        return self.peak_vram_mb / 1024.0

    def as_metadata(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class AutoresearchResultRow:
    row_index: int
    commit: str
    val_bpb: float
    memory_gb: float
    status: str
    description: str

    def as_metadata(self) -> dict[str, object]:
        return asdict(self)


class AutoresearchAdapter:
    """Converts `autoresearch` artifacts into ledger evidence and series summaries."""

    @classmethod
    def parse_run_log(cls, log_path: str | Path) -> AutoresearchRun:
        path = Path(log_path)
        return cls.parse_run_text(path.read_text(encoding="utf-8"), source_name=str(path))

    @classmethod
    def parse_run_text(cls, text: str, *, source_name: str = "<memory>") -> AutoresearchRun:
        values: dict[str, float | int] = {}
        for match in SUMMARY_PATTERN.finditer(text):
            raw_key = match.group("key")
            if raw_key not in KEY_MAP:
                continue
            key = KEY_MAP[raw_key]
            raw_value = match.group("value").strip()
            if key in INT_FIELDS:
                values[key] = int(float(raw_value))
            else:
                values[key] = float(raw_value)

        missing = [field for field in KEY_MAP.values() if field not in values]
        if missing:
            raise ValueError(
                f"Missing autoresearch summary fields in {source_name}: {', '.join(sorted(missing))}"
            )

        return AutoresearchRun(**values)

    @classmethod
    def parse_results_tsv(cls, results_tsv_path: str | Path) -> list[AutoresearchResultRow]:
        path = Path(results_tsv_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames or []
            missing = [field for field in RESULTS_TSV_FIELDS if field not in fieldnames]
            if missing:
                raise ValueError(
                    f"Missing autoresearch TSV fields in {path}: {', '.join(sorted(missing))}"
                )

            rows: list[AutoresearchResultRow] = []
            for row_index, raw in enumerate(reader, start=1):
                if not any((raw.get(field) or "").strip() for field in RESULTS_TSV_FIELDS):
                    continue
                rows.append(
                    AutoresearchResultRow(
                        row_index=row_index,
                        commit=(raw.get("commit") or "").strip(),
                        val_bpb=float((raw.get("val_bpb") or "0").strip()),
                        memory_gb=float((raw.get("memory_gb") or "0").strip()),
                        status=(raw.get("status") or "").strip().lower(),
                        description=(raw.get("description") or "").strip(),
                    )
                )
        return rows

    @classmethod
    def import_run(
        cls,
        *,
        ledger: EpistemicLedger,
        claim_id: str,
        run_log_path: str | Path,
        commit: str = "",
        branch: str = "",
        description: str = "",
        baseline_bpb: float | None = None,
        status: str = "",
        tolerance: float = 1e-4,
        artifact_paths: list[str] | None = None,
        decision_id: str = "",
        execution_status: ExecutionStatus | str | None = None,
        cost_estimate_usd: float | None = None,
        artifact_quality: float | None = None,
    ) -> Evidence:
        run_log = Path(run_log_path)
        artifacts = [str(run_log), *(artifact_paths or [])]

        if status.lower() == "crash":
            summary = description or "Autoresearch run crashed before producing comparable metrics."
            resolved_artifact_quality = (
                artifact_quality
                if artifact_quality is not None
                else cls.infer_artifact_quality(
                    commit=commit,
                    branch=branch,
                    artifacts=artifacts,
                    parsed_metrics=False,
                    status=status,
                )
            )
            evidence = ledger.add_evidence(
                claim_id=claim_id,
                summary=summary,
                direction=EvidenceDirection.CONTRADICT,
                strength=0.4,
                confidence=0.6,
                source_type="autoresearch",
                source_ref=commit or str(run_log),
                artifact_paths=artifacts,
                metadata={
                    "adapter": "autoresearch",
                    "branch": branch,
                    "commit": commit,
                    "status": status,
                    "baseline_bpb": baseline_bpb,
                    "description": description,
                    "run_log_path": str(run_log),
                },
            )
            cls._link_execution(
                ledger=ledger,
                evidence=evidence,
                decision_id=decision_id,
                execution_status=execution_status or cls.infer_execution_status(status=status),
                notes=summary,
                runtime_seconds=None,
                cost_estimate_usd=cost_estimate_usd,
                artifact_quality=resolved_artifact_quality,
                artifact_paths=artifacts,
            )
            return evidence

        run = cls.parse_run_log(run_log)
        direction = cls.infer_direction(
            run=run,
            baseline_bpb=baseline_bpb,
            status=status,
            tolerance=tolerance,
        )
        strength = cls.infer_strength(run=run, baseline_bpb=baseline_bpb, status=status)
        confidence = cls.infer_confidence(run=run, baseline_bpb=baseline_bpb)
        delta_bpb = None if baseline_bpb is None else baseline_bpb - run.val_bpb
        resolved_artifact_quality = (
            artifact_quality
            if artifact_quality is not None
            else cls.infer_artifact_quality(
                commit=commit,
                branch=branch,
                artifacts=artifacts,
                parsed_metrics=True,
                status=status,
            )
        )

        summary = cls.build_summary(
            run=run,
            description=description,
            commit=commit,
            status=status,
            delta_bpb=delta_bpb,
        )
        evidence = ledger.add_evidence(
            claim_id=claim_id,
            summary=summary,
            direction=direction,
            strength=strength,
            confidence=confidence,
            source_type="autoresearch",
            source_ref=commit or str(run_log),
            artifact_paths=artifacts,
            metadata={
                "adapter": "autoresearch",
                "branch": branch,
                "commit": commit,
                "status": status,
                "baseline_bpb": baseline_bpb,
                "delta_bpb": delta_bpb,
                "description": description,
                "run_log_path": str(run_log),
                "metrics": run.as_metadata(),
            },
        )
        cls._link_execution(
            ledger=ledger,
            evidence=evidence,
            decision_id=decision_id,
            execution_status=execution_status or cls.infer_execution_status(status=status),
            notes=summary,
            runtime_seconds=run.total_seconds,
            cost_estimate_usd=cost_estimate_usd,
            artifact_quality=resolved_artifact_quality,
            artifact_paths=artifacts,
        )
        return evidence

    @classmethod
    def import_results_tsv(
        cls,
        *,
        ledger: EpistemicLedger,
        claim_id: str,
        results_tsv_path: str | Path,
        branch: str = "",
        baseline_bpb: float | None = None,
        tolerance: float = 1e-4,
    ) -> dict[str, object]:
        claim = ledger.get_claim(claim_id)
        results_path = Path(results_tsv_path)
        rows = cls.parse_results_tsv(results_path)
        branch_series = cls.summarize_results_tsv(
            rows,
            baseline_bpb=baseline_bpb,
            tolerance=tolerance,
        )
        autoresearch_meta = claim.metadata.get("autoresearch", {})
        if not isinstance(autoresearch_meta, dict):
            autoresearch_meta = {}
        resolved_branch = cls._resolve_branch_key(
            autoresearch_meta=autoresearch_meta,
            results_tsv_path=results_path,
            requested_branch=branch,
        )
        series_by_branch = autoresearch_meta.get("series_by_branch", {})
        if not isinstance(series_by_branch, dict):
            series_by_branch = {}
        series_by_branch[resolved_branch] = {
            "results_tsv_path": str(results_path),
            "series": branch_series,
        }
        preferred_branch, preferred_entry = cls._choose_preferred_branch(series_by_branch)
        aggregate_series = cls.rollup_series_by_branch(
            series_by_branch,
            preferred_branch=preferred_branch,
        )
        evidence = cls._upsert_results_tsv_evidence(
            ledger=ledger,
            claim_id=claim_id,
            results_tsv_path=results_path,
            branch=resolved_branch,
            branch_key=resolved_branch,
            rows=rows,
            series=branch_series,
            tolerance=tolerance,
        )

        autoresearch_meta["branch"] = preferred_branch
        autoresearch_meta["results_tsv_path"] = str(
            preferred_entry.get("results_tsv_path", str(results_path))
        )
        autoresearch_meta["series"] = preferred_entry.get("series", branch_series)
        autoresearch_meta["aggregate_series"] = aggregate_series
        autoresearch_meta["series_by_branch"] = series_by_branch
        autoresearch_meta["last_imported_branch"] = resolved_branch
        autoresearch_meta["last_imported_results_tsv_path"] = str(results_path)
        claim.metadata["autoresearch"] = autoresearch_meta
        ledger.save()

        return {
            "claim_id": claim_id,
            "branch": resolved_branch,
            "results_tsv_path": str(results_path),
            "series": branch_series,
            "aggregate_series": aggregate_series,
            "evidence": serialize_dataclass(evidence),
        }

    @classmethod
    def rollup_series_by_branch(
        cls,
        series_by_branch: dict[str, dict[str, Any]],
        *,
        preferred_branch: str,
    ) -> dict[str, Any]:
        branch_count = len(series_by_branch)
        total_runs = 0
        keep_count = 0
        discard_count = 0
        crash_count = 0
        decided_count = 0
        frontier_improvement_count = 0
        active_branch_count = 0
        plateau_branch_count = 0
        weighted_memory_sum = 0.0
        weighted_memory_count = 0

        for entry in series_by_branch.values():
            series = entry.get("series", {})
            if not isinstance(series, dict):
                continue
            total_runs += int(series.get("total_runs") or 0)
            keep_count += int(series.get("keep_count") or 0)
            discard_count += int(series.get("discard_count") or 0)
            crash_count += int(series.get("crash_count") or 0)
            decided_count += int(series.get("decided_count") or 0)
            frontier_improvement_count += int(series.get("frontier_improvement_count") or 0)
            non_crash_count = int(series.get("total_runs") or 0) - int(series.get("crash_count") or 0)
            weighted_memory_sum += float(series.get("average_memory_gb") or 0.0) * max(
                non_crash_count,
                0,
            )
            weighted_memory_count += max(non_crash_count, 0)
            if cls._is_active_branch(series):
                active_branch_count += 1
            if cls._is_plateau_branch(series):
                plateau_branch_count += 1

        keep_rate = keep_count / decided_count if decided_count else 0.0
        crash_rate = crash_count / total_runs if total_runs else 0.0
        preferred_entry = series_by_branch.get(preferred_branch, {})
        preferred_series = preferred_entry.get("series", {})
        if not isinstance(preferred_series, dict):
            preferred_series = {}

        return {
            "branch_count": branch_count,
            "preferred_branch": preferred_branch,
            "preferred_results_tsv_path": str(
                preferred_entry.get("results_tsv_path", "")
            ).strip(),
            "active_branch_count": active_branch_count,
            "plateau_branch_count": plateau_branch_count,
            "total_runs": total_runs,
            "keep_count": keep_count,
            "discard_count": discard_count,
            "crash_count": crash_count,
            "decided_count": decided_count,
            "keep_rate": round(keep_rate, 6),
            "crash_rate": round(crash_rate, 6),
            "frontier_improvement_count": frontier_improvement_count,
            "average_memory_gb": round(
                weighted_memory_sum / weighted_memory_count if weighted_memory_count else 0.0,
                3,
            ),
            "best_improvement_bpb": round(
                float(preferred_series.get("best_improvement_bpb") or 0.0),
                6,
            ),
            "stagnation_run_count": int(preferred_series.get("stagnation_run_count") or 0),
        }

    @classmethod
    def summarize_results_tsv(
        cls,
        rows: list[AutoresearchResultRow],
        *,
        baseline_bpb: float | None,
        tolerance: float,
    ) -> dict[str, Any]:
        total_runs = len(rows)
        keep_count = sum(1 for row in rows if row.status == "keep")
        discard_count = sum(1 for row in rows if row.status == "discard")
        crash_count = sum(1 for row in rows if row.status == "crash")
        decided_count = keep_count + discard_count
        keep_rate = keep_count / decided_count if decided_count else 0.0
        crash_rate = crash_count / total_runs if total_runs else 0.0

        non_crash_rows = [row for row in rows if row.status != "crash"]
        first_non_crash = non_crash_rows[0] if non_crash_rows else None
        resolved_baseline_bpb = (
            baseline_bpb
            if baseline_bpb is not None
            else (first_non_crash.val_bpb if first_non_crash else None)
        )

        frontier_bpb = resolved_baseline_bpb
        frontier_improvement_count = 0
        last_improvement_row_index = 0
        if baseline_bpb is None and first_non_crash is not None:
            last_improvement_row_index = first_non_crash.row_index

        best_row: AutoresearchResultRow | None = None
        latest_non_crash: AutoresearchResultRow | None = None
        for row in rows:
            if row.status == "crash":
                continue
            latest_non_crash = row
            if best_row is None or row.val_bpb < best_row.val_bpb:
                best_row = row
            if frontier_bpb is None:
                frontier_bpb = row.val_bpb
                last_improvement_row_index = row.row_index
                continue
            if row.val_bpb < frontier_bpb - tolerance:
                frontier_improvement_count += 1
                frontier_bpb = row.val_bpb
                last_improvement_row_index = row.row_index

        stagnation_run_count = max(0, total_runs - last_improvement_row_index) if total_runs else 0
        average_memory_gb = (
            sum(row.memory_gb for row in non_crash_rows) / len(non_crash_rows)
            if non_crash_rows
            else 0.0
        )
        best_improvement_bpb = (
            resolved_baseline_bpb - best_row.val_bpb
            if resolved_baseline_bpb is not None and best_row is not None
            else 0.0
        )

        return {
            "total_runs": total_runs,
            "keep_count": keep_count,
            "discard_count": discard_count,
            "crash_count": crash_count,
            "decided_count": decided_count,
            "keep_rate": round(keep_rate, 6),
            "crash_rate": round(crash_rate, 6),
            "baseline_bpb": (
                round(resolved_baseline_bpb, 6)
                if resolved_baseline_bpb is not None
                else None
            ),
            "best_val_bpb": round(best_row.val_bpb, 6) if best_row else None,
            "best_improvement_bpb": round(best_improvement_bpb, 6),
            "latest_val_bpb": round(latest_non_crash.val_bpb, 6) if latest_non_crash else None,
            "average_memory_gb": round(average_memory_gb, 3),
            "frontier_improvement_count": frontier_improvement_count,
            "stagnation_run_count": stagnation_run_count,
            "last_improvement_row_index": last_improvement_row_index,
            "best_row_index": best_row.row_index if best_row else 0,
            "best_commit": best_row.commit if best_row else "",
            "best_description": best_row.description if best_row else "",
            "latest_status": rows[-1].status if rows else "",
        }

    @staticmethod
    def _resolve_branch_key(
        *,
        autoresearch_meta: dict[str, Any],
        results_tsv_path: Path,
        requested_branch: str,
    ) -> str:
        requested = requested_branch.strip()
        if requested:
            return requested

        results_path = str(results_tsv_path)
        series_by_branch = autoresearch_meta.get("series_by_branch", {})
        if isinstance(series_by_branch, dict):
            for branch_key, entry in series_by_branch.items():
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("results_tsv_path", "")).strip() == results_path:
                    return str(branch_key).strip() or "default"

        stored_path = str(autoresearch_meta.get("results_tsv_path", "")).strip()
        stored_branch = str(autoresearch_meta.get("branch", "")).strip()
        if stored_branch and stored_path == results_path:
            return stored_branch

        last_branch = str(autoresearch_meta.get("last_imported_branch", "")).strip()
        last_path = str(autoresearch_meta.get("last_imported_results_tsv_path", "")).strip()
        if last_branch and last_path == results_path:
            return last_branch

        return "default"

    @classmethod
    def _choose_preferred_branch(
        cls,
        series_by_branch: dict[str, dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        if not series_by_branch:
            return "default", {"results_tsv_path": "", "series": {}}

        preferred_branch = max(
            series_by_branch,
            key=lambda branch_key: cls._score_branch_entry(series_by_branch[branch_key]),
        )
        entry = series_by_branch[preferred_branch]
        return preferred_branch, entry if isinstance(entry, dict) else {"results_tsv_path": "", "series": {}}

    @staticmethod
    def _score_branch_entry(entry: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        series = entry.get("series", {})
        if not isinstance(series, dict):
            series = {}
        return (
            float(series.get("best_improvement_bpb") or 0.0),
            float(series.get("keep_rate") or 0.0),
            float(series.get("frontier_improvement_count") or 0.0),
            -float(series.get("stagnation_run_count") or 0.0),
            -float(series.get("crash_rate") or 0.0),
            float(series.get("total_runs") or 0.0),
        )

    @staticmethod
    def _is_active_branch(series: dict[str, Any]) -> bool:
        best_improvement = float(series.get("best_improvement_bpb") or 0.0)
        stagnation = int(series.get("stagnation_run_count") or 0)
        return best_improvement > 0.0005 and stagnation <= 3

    @staticmethod
    def _is_plateau_branch(series: dict[str, Any]) -> bool:
        total_runs = int(series.get("total_runs") or 0)
        stagnation = int(series.get("stagnation_run_count") or 0)
        best_improvement = float(series.get("best_improvement_bpb") or 0.0)
        return total_runs >= 4 and stagnation >= 4 and best_improvement <= 0.0005

    @staticmethod
    def infer_direction(
        *,
        run: AutoresearchRun,
        baseline_bpb: float | None,
        status: str,
        tolerance: float,
    ) -> EvidenceDirection:
        normalized_status = status.lower()
        if baseline_bpb is not None:
            delta = baseline_bpb - run.val_bpb
            if delta > tolerance:
                return EvidenceDirection.SUPPORT
            if delta < -tolerance:
                return EvidenceDirection.CONTRADICT
            return EvidenceDirection.INCONCLUSIVE
        if normalized_status == "keep":
            return EvidenceDirection.SUPPORT
        if normalized_status in {"discard", "crash"}:
            return EvidenceDirection.CONTRADICT
        return EvidenceDirection.INCONCLUSIVE

    @staticmethod
    def infer_strength(
        *, run: AutoresearchRun, baseline_bpb: float | None, status: str
    ) -> float:
        normalized_status = status.lower()
        if normalized_status == "crash":
            return 0.4

        delta_component = 0.0
        if baseline_bpb is not None:
            delta_component = clamp(abs(baseline_bpb - run.val_bpb) / 0.01)

        status_component = 0.15 if normalized_status == "keep" else 0.0
        duration_component = clamp(run.training_seconds / max(run.total_seconds, 1.0))
        return clamp(0.35 + 0.30 * delta_component + 0.20 * duration_component + status_component)

    @staticmethod
    def infer_confidence(*, run: AutoresearchRun, baseline_bpb: float | None) -> float:
        metric_completeness = 1.0 if run.num_steps > 0 and run.total_seconds > 0 else 0.6
        baseline_bonus = 0.15 if baseline_bpb is not None else 0.0
        return clamp(0.55 + 0.20 * metric_completeness + baseline_bonus)

    @staticmethod
    def infer_execution_status(*, status: str) -> ExecutionStatus:
        if status.lower() == "crash":
            return ExecutionStatus.FAILED
        return ExecutionStatus.SUCCEEDED

    @staticmethod
    def infer_series_direction(
        *, series: dict[str, Any], tolerance: float
    ) -> EvidenceDirection:
        best_improvement_bpb = float(series.get("best_improvement_bpb") or 0.0)
        total_runs = int(series.get("total_runs") or 0)
        keep_count = int(series.get("keep_count") or 0)
        crash_rate = float(series.get("crash_rate") or 0.0)
        if best_improvement_bpb > tolerance and keep_count > 0:
            return EvidenceDirection.SUPPORT
        if total_runs >= 3 and best_improvement_bpb <= tolerance and (
            crash_rate >= 0.34 or keep_count == 0
        ):
            return EvidenceDirection.CONTRADICT
        return EvidenceDirection.INCONCLUSIVE

    @staticmethod
    def infer_series_strength(*, series: dict[str, Any]) -> float:
        best_improvement_bpb = max(float(series.get("best_improvement_bpb") or 0.0), 0.0)
        keep_rate = float(series.get("keep_rate") or 0.0)
        crash_rate = float(series.get("crash_rate") or 0.0)
        improvement_component = clamp(best_improvement_bpb / 0.01)
        return clamp(
            0.30
            + 0.35 * improvement_component
            + 0.20 * keep_rate
            + 0.15 * (1.0 - crash_rate)
        )

    @staticmethod
    def infer_series_confidence(*, series: dict[str, Any]) -> float:
        total_runs = int(series.get("total_runs") or 0)
        crash_rate = float(series.get("crash_rate") or 0.0)
        has_baseline = series.get("baseline_bpb") is not None
        coverage = clamp(total_runs / 8.0)
        return clamp(
            0.45
            + 0.30 * coverage
            + 0.15 * (1.0 - crash_rate)
            + (0.10 if has_baseline else 0.0)
        )

    @staticmethod
    def infer_artifact_quality(
        *,
        commit: str,
        branch: str,
        artifacts: list[str],
        parsed_metrics: bool,
        status: str,
    ) -> float:
        quality = 0.15
        if parsed_metrics:
            quality += 0.35
        if commit:
            quality += 0.15
        if branch:
            quality += 0.10
        if len(artifacts) > 1:
            quality += 0.15
        if status.lower() != "crash":
            quality += 0.10
        return clamp(quality)

    @classmethod
    def build_results_tsv_summary(
        cls,
        *,
        results_tsv_path: str | Path,
        branch: str,
        series: dict[str, Any],
    ) -> str:
        path = Path(results_tsv_path)
        branch_fragment = f" on {branch}" if branch else ""
        best_val_bpb = series.get("best_val_bpb")
        if best_val_bpb is None:
            best_fragment = "best val_bpb=n/a"
        else:
            best_fragment = f"best val_bpb={float(best_val_bpb):.6f}"

        delta_fragment = ""
        if series.get("baseline_bpb") is not None:
            delta_fragment = (
                f", best delta_bpb={float(series.get('best_improvement_bpb') or 0.0):+.6f}"
            )

        summary = (
            f"Autoresearch series from {path.name}{branch_fragment}: "
            f"{int(series.get('total_runs') or 0)} runs, "
            f"keep={int(series.get('keep_count') or 0)}, "
            f"discard={int(series.get('discard_count') or 0)}, "
            f"crash={int(series.get('crash_count') or 0)}, "
            f"{best_fragment}{delta_fragment}, "
            f"stagnation={int(series.get('stagnation_run_count') or 0)} runs."
        )
        best_commit = str(series.get("best_commit") or "").strip()
        best_description = str(series.get("best_description") or "").strip()
        if best_commit or best_description:
            label = best_commit or f"row {int(series.get('best_row_index') or 0)}"
            detail = best_description or "best run"
            summary += f" Best run {label}: {detail}."
        return summary

    @staticmethod
    def build_summary(
        *,
        run: AutoresearchRun,
        description: str,
        commit: str,
        status: str,
        delta_bpb: float | None,
    ) -> str:
        prefix = description.strip() or "Autoresearch run"
        commit_fragment = f" @ {commit}" if commit else ""
        delta_fragment = (
            f", delta_bpb={delta_bpb:+.6f}" if delta_bpb is not None else ""
        )
        status_fragment = f", status={status}" if status else ""
        return (
            f"{prefix}{commit_fragment}: val_bpb={run.val_bpb:.6f}{delta_fragment}, "
            f"peak_vram_gb={run.peak_vram_gb:.1f}, depth={run.depth}, "
            f"steps={run.num_steps}{status_fragment}."
        )

    @classmethod
    def _upsert_results_tsv_evidence(
        cls,
        *,
        ledger: EpistemicLedger,
        claim_id: str,
        results_tsv_path: str | Path,
        branch: str,
        branch_key: str,
        rows: list[AutoresearchResultRow],
        series: dict[str, Any],
        tolerance: float,
    ) -> Evidence:
        results_path = str(results_tsv_path)
        metadata = {
            "adapter": "autoresearch",
            "stage": "results_tsv_series",
            "branch": branch,
            "branch_key": branch_key,
            "results_tsv_path": results_path,
            "row_count": len(rows),
            "rows": [row.as_metadata() for row in rows],
            "series": series,
        }
        summary = cls.build_results_tsv_summary(
            results_tsv_path=results_tsv_path,
            branch=branch,
            series=series,
        )
        direction = cls.infer_series_direction(series=series, tolerance=tolerance)
        strength = cls.infer_series_strength(series=series)
        confidence = cls.infer_series_confidence(series=series)

        existing = next(
            (
                item
                for item in ledger.evidence_for_claim(claim_id)
                if item.metadata.get("adapter") == "autoresearch"
                and item.metadata.get("stage") == "results_tsv_series"
                and (
                    item.metadata.get("branch_key") == branch_key
                    or item.metadata.get("results_tsv_path") == results_path
                )
            ),
            None,
        )
        if existing is not None:
            existing.summary = summary
            existing.direction = direction
            existing.strength = strength
            existing.confidence = confidence
            existing.source_type = "autoresearch"
            existing.source_ref = results_path
            existing.artifact_paths = [results_path]
            existing.metadata = metadata
            return existing

        evidence = Evidence(
            id=ledger._generate_id("ev"),
            claim_id=claim_id,
            summary=summary,
            direction=direction,
            strength=strength,
            confidence=confidence,
            source_type="autoresearch",
            source_ref=results_path,
            artifact_paths=[results_path],
            metadata=metadata,
        )
        ledger.evidence[evidence.id] = evidence
        return evidence

    @staticmethod
    def _link_execution(
        *,
        ledger: EpistemicLedger,
        evidence: Evidence,
        decision_id: str,
        execution_status: ExecutionStatus | str | None,
        notes: str,
        runtime_seconds: float | None,
        cost_estimate_usd: float | None,
        artifact_quality: float | None,
        artifact_paths: list[str],
    ) -> None:
        if not decision_id or execution_status is None:
            return

        execution = ledger.record_execution(
            decision_id=decision_id,
            claim_id=evidence.claim_id,
            status=execution_status,
            notes=notes,
            runtime_seconds=runtime_seconds,
            cost_estimate_usd=cost_estimate_usd,
            artifact_quality=artifact_quality,
            artifact_paths=artifact_paths,
            metadata={
                "adapter": "autoresearch",
                "evidence_id": evidence.id,
            },
        )
        evidence.metadata["decision_id"] = decision_id
        evidence.metadata["execution_id"] = execution.id
        ledger.save()
