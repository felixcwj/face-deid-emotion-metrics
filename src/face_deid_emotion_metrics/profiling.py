from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict


class StageProfiler:
    def __init__(self) -> None:
        self._totals: Dict[str, float] = defaultdict(float)

    def add(self, stage: str, duration: float) -> None:
        if duration <= 0.0:
            return
        self._totals[stage] += duration

    def total(self, stage: str) -> float:
        return self._totals.get(stage, 0.0)

    def totals(self) -> Dict[str, float]:
        return dict(self._totals)


@dataclass(frozen=True)
class ProfileResult:
    processed: int
    requested: int
    totals: Dict[str, float]
    elapsed: float
