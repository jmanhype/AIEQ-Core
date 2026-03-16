from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..models import ActionProposal, Claim


@dataclass(frozen=True, slots=True)
class ModeDescriptor:
    name: str
    label: str
    description: str


class ModeAdapter(ABC):
    name: str
    label: str
    description: str

    def descriptor(self) -> ModeDescriptor:
        return ModeDescriptor(
            name=self.name,
            label=self.label,
            description=self.description,
        )

    def claim_mode(self, claim: Claim) -> str:
        raw_mode = claim.metadata.get("mode", "")
        normalized = str(raw_mode).strip()
        return normalized or self.name

    def supports_claim(self, claim: Claim) -> bool:
        return self.claim_mode(claim) == self.name

    @abstractmethod
    def bootstrap_proposal(self, *, ledger: Any) -> ActionProposal | None:
        raise NotImplementedError

    @abstractmethod
    def build_proposals(self, *, ledger: Any, claim: Claim) -> list[ActionProposal]:
        raise NotImplementedError

    @abstractmethod
    def doctor(self, *, config: Any, ledger_path: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class ModeRegistry:
    def __init__(self, adapters: list[ModeAdapter]) -> None:
        self._adapters = {adapter.name: adapter for adapter in adapters}

    def list_modes(self) -> list[ModeDescriptor]:
        return [self._adapters[key].descriptor() for key in sorted(self._adapters)]

    def get(self, name: str) -> ModeAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise KeyError(f"Unknown mode: {name}") from exc

    def default(self) -> ModeAdapter:
        return self.get("ml_research")

    def for_claim(self, claim: Claim) -> ModeAdapter:
        raw_mode = str(claim.metadata.get("mode", "")).strip()
        if raw_mode and raw_mode in self._adapters:
            return self._adapters[raw_mode]
        return self.default()


def default_mode_registry() -> ModeRegistry:
    from .ml_research import MLResearchMode
    from .skill_optimizer import SkillOptimizerMode

    return ModeRegistry([MLResearchMode(), SkillOptimizerMode()])
