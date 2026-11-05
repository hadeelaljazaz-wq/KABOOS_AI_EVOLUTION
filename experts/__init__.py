"""KABOOS gold expert desk — first working version (three independent experts).

Built on top of the existing gold_data contracts/snapshot and the
collective_memory store. Each expert has a stable id, its own method and
settings, computes its evidence/levels/plan with program code (LLM only words
the Arabic narrative, never the numbers), and its original report + plan are
saved and recoverable after a restart. No shared recommendation / discussion
is produced here — that is the next task.
"""

from .base import BaseExpert, ComputeOutput, ExpertResult, MethodSettings, PlanProposal
from .desk import ExpertDesk, default_experts
from .momentum import MomentumTrendExpert, MomentumSettings
from .price_action import PriceStructureExpert, PriceStructureSettings
from .storage import ExpertDeskStore
from .supply_demand import SupplyDemandExpert, SupplyDemandSettings

__all__ = [
    "ExpertDesk", "default_experts", "ExpertDeskStore",
    "BaseExpert", "ExpertResult", "ComputeOutput", "MethodSettings", "PlanProposal",
    "PriceStructureExpert", "PriceStructureSettings",
    "MomentumTrendExpert", "MomentumSettings",
    "SupplyDemandExpert", "SupplyDemandSettings",
]
