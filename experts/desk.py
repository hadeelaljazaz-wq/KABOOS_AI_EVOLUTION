"""The gold expert desk: run three independent experts on one snapshot/cycle.

Each expert gets the SAME snapshot and cycle id and its OWN memory, produces
its first report independently (never sees the others), and its report + plan +
screen are persisted. No shared recommendation is decided here — that is the
next task. Re-running an existing cycle does not duplicate anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from gold_data.snapshot import MarketSnapshot

from .base import ExpertResult
from .momentum import MomentumTrendExpert
from .price_action import PriceStructureExpert
from .storage import ExpertDeskStore
from .supply_demand import SupplyDemandExpert

try:
    from collective_memory import CollectiveMemory
except Exception:  # pragma: no cover
    CollectiveMemory = None  # type: ignore


def default_experts():
    return [PriceStructureExpert(), MomentumTrendExpert(), SupplyDemandExpert()]


class ExpertDesk:
    def __init__(self, store_path: str | Path, memory_path: Optional[str | Path] = None,
                 experts=None):
        self.store = ExpertDeskStore(store_path)
        self.memory = CollectiveMemory(memory_path) if (memory_path and CollectiveMemory) else None
        self.experts = experts if experts is not None else default_experts()

    def _resolve_llm(self, run_mode: str):
        if run_mode != "llm":
            return None
        try:
            from agent_community.llm import get_llm
            return get_llm(allow_mock=False)   # real backend only; never Mock for experts
        except Exception:
            return None                        # no key -> narrator falls back, clearly labelled

    def run_cycle(self, snapshot: MarketSnapshot, cycle_id: str,
                  reference_time: Optional[str] = None, run_mode: str = "deterministic",
                  force: bool = False) -> tuple[list[ExpertResult], bool]:
        """Return (results, inserted). If the cycle already exists and not force,
        returns the stored results unchanged with inserted=False (no duplicates)."""
        reference_time = reference_time or snapshot.analysis_reference_time
        if self.store.cycle_exists(cycle_id) and not force:
            return self.load_cycle_results(cycle_id), False

        llm_backend = self._resolve_llm(run_mode)
        results: list[ExpertResult] = []
        for expert in self.experts:
            results.append(expert.analyze(
                snapshot, cycle_id, reference_time=reference_time,
                memory=self.memory, llm_backend=llm_backend, run_mode=run_mode))
        inserted = self.store.save_cycle(
            cycle_id, snapshot.snapshot_id, snapshot.symbol_canonical,
            reference_time, run_mode, results)
        return results, inserted

    def load_cycle_results(self, cycle_id: str) -> list[ExpertResult]:
        data = self.store.load_cycle(cycle_id)
        if data is None:
            return []
        out = []
        for e in data["experts"]:
            r = e["report"]
            out.append(ExpertResult(
                expert_id=e["expert_id"], specialty=e["specialty"],
                method_version=e["method_version"], cycle_id=cycle_id,
                snapshot_id=r.snapshot_id, timeframes_used=e["timeframes_used"],
                report=r, plan=e["plan"], analysis_ar=e["analysis_ar"],
                run_mode=e["run_mode"], success=e["success"], abstained=e["abstained"],
                abstain_reason=e["abstain_reason"], missing=e["screen"].get("missing", []),
                screen=e["screen"]))
        return out

    def load_cycle(self, cycle_id: str):
        return self.store.load_cycle(cycle_id)

    def list_cycles(self):
        return self.store.list_cycles()

    def close(self) -> None:
        self.store.close()
        if self.memory is not None:
            self.memory.close()
