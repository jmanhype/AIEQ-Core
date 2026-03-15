from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re

from ..ledger import EpistemicLedger
from ..models import (
    Attack,
    Artifact,
    ArtifactKind,
    Claim,
    Evidence,
    EvidenceDirection,
    ExecutionStatus,
    serialize_dataclass,
)

INPUT_FILES = "input_files"
PLOTS_FOLDER = "plots"
PAPER_FOLDER = "paper"
DESCRIPTION_FILE = "data_description.md"
IDEA_FILE = "idea.md"
METHOD_FILE = "methods.md"
RESULTS_FILE = "results.md"
LITERATURE_FILE = "literature.md"
REFEREE_FILE = "referee.md"


@dataclass(slots=True)
class DenarioProjectSnapshot:
    project_dir: str
    data_description: str = ""
    idea: str = ""
    method: str = ""
    results: str = ""
    literature: str = ""
    referee: str = ""
    plot_paths: list[str] = field(default_factory=list)
    paper_paths: list[str] = field(default_factory=list)
    input_file_paths: dict[str, str] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, object]:
        return asdict(self)


class DenarioAdapter:
    """Imports Denario project artifacts into the AIEQ ledger."""

    @classmethod
    def load_project(cls, project_dir: str | Path) -> DenarioProjectSnapshot:
        project_path = Path(project_dir)
        input_dir = project_path / INPUT_FILES
        plot_dir = input_dir / PLOTS_FOLDER
        paper_dir = project_path / PAPER_FOLDER

        input_file_paths = {
            "data_description": str(input_dir / DESCRIPTION_FILE),
            "idea": str(input_dir / IDEA_FILE),
            "method": str(input_dir / METHOD_FILE),
            "results": str(input_dir / RESULTS_FILE),
            "literature": str(input_dir / LITERATURE_FILE),
            "referee": str(input_dir / REFEREE_FILE),
        }

        snapshot = DenarioProjectSnapshot(
            project_dir=str(project_path),
            data_description=cls._read_text(input_dir / DESCRIPTION_FILE),
            idea=cls._read_text(input_dir / IDEA_FILE),
            method=cls._read_text(input_dir / METHOD_FILE),
            results=cls._read_text(input_dir / RESULTS_FILE),
            literature=cls._read_text(input_dir / LITERATURE_FILE),
            referee=cls._read_text(input_dir / REFEREE_FILE),
            plot_paths=cls._collect_files(plot_dir),
            paper_paths=cls._collect_files(paper_dir),
            input_file_paths=input_file_paths,
        )
        return snapshot

    @classmethod
    def import_project(
        cls,
        *,
        ledger: EpistemicLedger,
        project_dir: str | Path,
        claim_id: str | None = None,
        novelty: float = 0.7,
        falsifiability: float = 0.6,
        tags: list[str] | None = None,
        results_direction: EvidenceDirection | str = EvidenceDirection.INCONCLUSIVE,
        results_strength: float = 0.65,
        results_confidence: float = 0.7,
        literature_severity: float = 0.6,
        referee_severity: float = 0.75,
        decision_id: str = "",
        execution_status: ExecutionStatus | str | None = None,
        runtime_seconds: float | None = None,
        cost_estimate_usd: float | None = None,
        artifact_quality: float | None = None,
    ) -> dict[str, object]:
        snapshot = cls.load_project(project_dir)
        claim = cls._get_or_create_claim(
            ledger=ledger,
            snapshot=snapshot,
            claim_id=claim_id,
            novelty=novelty,
            falsifiability=falsifiability,
            tags=tags or [],
        )
        method_artifact = cls._upsert_method_artifact(
            ledger=ledger,
            claim=claim,
            snapshot=snapshot,
        )
        paper_artifacts = cls._upsert_paper_artifacts(
            ledger=ledger,
            claim=claim,
            snapshot=snapshot,
        )

        denario_meta = {
            "project_dir": snapshot.project_dir,
            "input_file_paths": snapshot.input_file_paths,
            "plot_paths": snapshot.plot_paths,
            "paper_paths": snapshot.paper_paths,
            "data_description": snapshot.data_description,
            "method": snapshot.method,
            "method_artifact_id": method_artifact.id if method_artifact else "",
            "paper_artifact_ids": [item.id for item in paper_artifacts],
        }
        claim.metadata["denario"] = denario_meta
        ledger.save()

        evidence = None
        if snapshot.results:
            evidence = ledger.add_evidence(
                claim_id=claim.id,
                summary=cls._build_results_summary(snapshot),
                direction=(
                    results_direction
                    if isinstance(results_direction, EvidenceDirection)
                    else EvidenceDirection(results_direction)
                ),
                strength=results_strength,
                confidence=results_confidence,
                source_type="denario",
                source_ref=snapshot.project_dir,
                artifact_paths=[
                    snapshot.input_file_paths["results"],
                    *snapshot.plot_paths,
                    *snapshot.paper_paths,
                ],
                metadata={
                    "adapter": "denario",
                    "stage": "results",
                    "project_dir": snapshot.project_dir,
                    "method": snapshot.method,
                    "method_artifact_id": method_artifact.id if method_artifact else "",
                    "paper_artifact_ids": [item.id for item in paper_artifacts],
                    "paper_paths": snapshot.paper_paths,
                },
            )

        attacks: list[Attack] = []
        if snapshot.literature:
            attacks.append(
                ledger.add_attack(
                    claim_id=claim.id,
                    description=cls._truncate(
                        f"Denario literature review: {cls._first_paragraph(snapshot.literature)}",
                        1200,
                    ),
                    target_kind="claim",
                    severity=literature_severity,
                    metadata={
                        "adapter": "denario",
                        "stage": "literature",
                        "project_dir": snapshot.project_dir,
                        "input_path": snapshot.input_file_paths["literature"],
                        "method_artifact_id": method_artifact.id if method_artifact else "",
                    },
                )
            )

        if snapshot.referee:
            attacks.append(
                ledger.add_attack(
                    claim_id=claim.id,
                    description=cls._truncate(
                        f"Denario referee report: {cls._first_paragraph(snapshot.referee)}",
                        1200,
                    ),
                    target_kind="paper",
                    severity=referee_severity,
                    metadata={
                        "adapter": "denario",
                        "stage": "referee",
                        "project_dir": snapshot.project_dir,
                        "input_path": snapshot.input_file_paths["referee"],
                        "paper_artifact_ids": [item.id for item in paper_artifacts],
                        "paper_paths": snapshot.paper_paths,
                    },
                )
            )

        execution = None
        if decision_id:
            resolved_artifact_quality = (
                artifact_quality
                if artifact_quality is not None
                else cls.infer_artifact_quality(snapshot)
            )
            execution = ledger.record_execution(
                decision_id=decision_id,
                claim_id=claim.id,
                claim_title=claim.title,
                status=execution_status or ExecutionStatus.SUCCEEDED,
                notes=f"Imported Denario project artifacts from {snapshot.project_dir}.",
                runtime_seconds=runtime_seconds,
                cost_estimate_usd=cost_estimate_usd,
                artifact_quality=resolved_artifact_quality,
                artifact_paths=[
                    *snapshot.input_file_paths.values(),
                    *snapshot.plot_paths,
                    *snapshot.paper_paths,
                ],
                metadata={
                    "adapter": "denario",
                    "project_dir": snapshot.project_dir,
                    "results_evidence_id": evidence.id if evidence else "",
                    "attack_ids": [item.id for item in attacks],
                    "artifact_ids": [
                        *( [method_artifact.id] if method_artifact else [] ),
                        *[item.id for item in paper_artifacts],
                    ],
                },
            )
            cls._attach_execution_link(
                ledger=ledger,
                execution_id=execution.id,
                decision_id=decision_id,
                evidence=evidence,
                attacks=attacks,
                artifacts=[item for item in [method_artifact, *paper_artifacts] if item is not None],
            )

        return {
            "claim": serialize_dataclass(claim),
            "method_artifact": serialize_dataclass(method_artifact) if method_artifact else None,
            "paper_artifacts": [serialize_dataclass(item) for item in paper_artifacts],
            "results_evidence": serialize_dataclass(evidence) if evidence else None,
            "attacks": [serialize_dataclass(item) for item in attacks],
            "execution": serialize_dataclass(execution) if execution else None,
            "snapshot": snapshot.as_metadata(),
        }

    @classmethod
    def _get_or_create_claim(
        cls,
        *,
        ledger: EpistemicLedger,
        snapshot: DenarioProjectSnapshot,
        claim_id: str | None,
        novelty: float,
        falsifiability: float,
        tags: list[str],
    ) -> Claim:
        if claim_id:
            return ledger.get_claim(claim_id)

        title = cls._extract_title(snapshot.idea) or Path(snapshot.project_dir).name
        statement_source = snapshot.idea or snapshot.data_description or title
        return ledger.add_claim(
            title=title,
            statement=cls._truncate(statement_source.strip(), 4000),
            novelty=novelty,
            falsifiability=falsifiability,
            tags=tags,
            metadata={"source_type": "denario"},
        )

    @classmethod
    def _build_results_summary(cls, snapshot: DenarioProjectSnapshot) -> str:
        lead = cls._first_paragraph(snapshot.results) or "Denario generated project results."
        title = cls._extract_title(snapshot.idea)
        prefix = f"Denario results for '{title}': " if title else "Denario results: "
        return cls._truncate(prefix + lead, 1200)

    @classmethod
    def _upsert_method_artifact(
        cls,
        *,
        ledger: EpistemicLedger,
        claim: Claim,
        snapshot: DenarioProjectSnapshot,
    ) -> Artifact | None:
        if not snapshot.method:
            return None
        method_path = snapshot.input_file_paths["method"]
        return ledger.upsert_artifact(
            claim_id=claim.id,
            kind=ArtifactKind.METHOD,
            title=f"Denario method for {claim.title}",
            content=cls._truncate(snapshot.method.strip(), 8000),
            source_type="denario",
            source_ref=snapshot.project_dir,
            source_path=method_path,
            metadata={
                "adapter": "denario",
                "stage": "method",
                "project_dir": snapshot.project_dir,
                "input_path": method_path,
            },
        )

    @classmethod
    def _upsert_paper_artifacts(
        cls,
        *,
        ledger: EpistemicLedger,
        claim: Claim,
        snapshot: DenarioProjectSnapshot,
    ) -> list[Artifact]:
        artifacts: list[Artifact] = []
        for paper_path in snapshot.paper_paths:
            path = Path(paper_path)
            artifacts.append(
                ledger.upsert_artifact(
                    claim_id=claim.id,
                    kind=ArtifactKind.PAPER,
                    title=path.name,
                    content=cls._truncate(cls._read_artifact_text(path), 12000),
                    source_type="denario",
                    source_ref=snapshot.project_dir,
                    source_path=str(path),
                    metadata={
                        "adapter": "denario",
                        "stage": "paper",
                        "project_dir": snapshot.project_dir,
                        "paper_path": str(path),
                    },
                )
            )
        return artifacts

    @staticmethod
    def _collect_files(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [
            str(item)
            for item in sorted(path.iterdir())
            if item.is_file() and item.name != ".DS_Store"
        ]

    @staticmethod
    def _read_text(path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _read_artifact_text(path: Path) -> str:
        if not path.exists():
            return ""
        if path.suffix.lower() not in {".md", ".txt", ".tex", ".rst"}:
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            return ""

    @staticmethod
    def _extract_title(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            return re.sub(r"^[*\-\d.\s]+", "", stripped).strip()
        return ""

    @staticmethod
    def _first_paragraph(text: str) -> str:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if not paragraphs:
            return ""
        return re.sub(r"\s+", " ", paragraphs[0]).strip()

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def infer_artifact_quality(snapshot: DenarioProjectSnapshot) -> float:
        quality = 0.0
        if snapshot.idea:
            quality += 0.20
        if snapshot.method:
            quality += 0.20
        if snapshot.results:
            quality += 0.20
        if snapshot.plot_paths:
            quality += 0.15
        if snapshot.paper_paths:
            quality += 0.15
        if snapshot.literature:
            quality += 0.05
        if snapshot.referee:
            quality += 0.05
        return min(1.0, quality)

    @staticmethod
    def _attach_execution_link(
        *,
        ledger: EpistemicLedger,
        execution_id: str,
        decision_id: str,
        evidence: Evidence | None,
        attacks: list[Attack],
        artifacts: list[Artifact],
    ) -> None:
        if evidence is not None:
            evidence.metadata["decision_id"] = decision_id
            evidence.metadata["execution_id"] = execution_id
        for attack in attacks:
            attack.metadata["decision_id"] = decision_id
            attack.metadata["execution_id"] = execution_id
        for artifact in artifacts:
            artifact.metadata["decision_id"] = decision_id
            artifact.metadata["execution_id"] = execution_id
        ledger.save()
