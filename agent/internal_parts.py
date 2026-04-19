"""Internal Parts — simplified competition system for agent perspectives.

Each "part" represents an internal motivation (executor, quality, curiosity, safety).
Parts compete via relevance × intensity scoring. The winner influences the agent's
behavior for that turn.

Based on issue #90 (Segmented Motivational Systems) — lightweight implementation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Part:
    """A single internal perspective/motivation."""
    name: str
    wants: str
    intensity: float
    keywords: List[str] = field(default_factory=list)
    wins: int = 0
    losses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


# Default parts configuration
DEFAULT_PARTS = [
    Part(
        name="executor",
        wants="complete the task efficiently",
        intensity=0.5,
        keywords=["task", "steps", "progress", "done", "finish", "complete"],
    ),
    Part(
        name="quality",
        wants="ensure high quality output",
        intensity=0.3,
        keywords=["quality", "review", "check", "error", "mistake", "improve"],
    ),
    Part(
        name="curiosity",
        wants="explore interesting possibilities",
        intensity=0.2,
        keywords=["interesting", "explore", "discover", "new", "try", "wonder"],
    ),
    Part(
        name="safety",
        wants="prevent destructive actions",
        intensity=0.4,
        keywords=["delete", "destroy", "remove", "drop", "destructive", "danger", "risk"],
    ),
]


class InternalParts:
    """Simplified internal parts competition system.

    Parts compete based on context relevance × intensity.
    The winner can influence agent behavior (e.g., safety wins → be more cautious).
    Weights evolve based on outcomes (success → winning part gains, failure → loses).
    """

    def __init__(self, parts: Optional[List[Part]] = None):
        self.parts = parts or [Part(**{k: v for k, v in p.__dict__.items()}) for p in DEFAULT_PARTS]

    def get_part(self, name: str) -> Optional[Part]:
        for p in self.parts:
            if p.name == name:
                return p
        return None

    def bid(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parts compete for relevance. Returns sorted bids (highest first)."""
        bids = []
        context_str = json.dumps(context, ensure_ascii=False).lower()

        for part in self.parts:
            relevance = self._compute_relevance(part, context_str, context)
            score = relevance * part.intensity
            if score > 0.05:  # minimum threshold
                bids.append({
                    "part": part.name,
                    "wants": part.wants,
                    "score": round(score, 3),
                    "relevance": round(relevance, 3),
                    "intensity": part.intensity,
                })

        bids.sort(key=lambda b: b["score"], reverse=True)
        return bids

    def get_top_bid(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the winning part for the current context."""
        bids = self.bid(context)
        return bids[0] if bids else None

    def evolve(self, outcome: Dict[str, Any]):
        """Adjust part intensities based on outcomes.

        Supported outcomes:
        - {"quality_win": True}  → quality intensity +0.05
        - {"quality_lose": True} → quality intensity -0.03
        - {"executor_win": True} → executor intensity +0.03
        - {"executor_lose": True} → executor intensity -0.02
        - {"safety_triggered": True} → safety intensity +0.05
        - {"curiosity_rewarded": True} → curiosity intensity +0.04
        """
        delta_map = {
            "win": 0.05,
            "lose": -0.03,
            "triggered": 0.05,
            "rewarded": 0.04,
        }
        for part in self.parts:
            win_key = f"{part.name}_win"
            lose_key = f"{part.name}_lose"
            triggered_key = f"{part.name}_triggered"
            rewarded_key = f"{part.name}_rewarded"

            if outcome.get(win_key):
                part.intensity = min(1.0, part.intensity + delta_map["win"])
                part.wins += 1
            elif outcome.get(lose_key):
                part.intensity = max(0.05, part.intensity + delta_map["lose"])
                part.losses += 1
            elif outcome.get(triggered_key):
                part.intensity = min(1.0, part.intensity + delta_map["triggered"])
                part.wins += 1
            elif outcome.get(rewarded_key):
                part.intensity = min(1.0, part.intensity + delta_map["rewarded"])
                part.wins += 1

    def reset(self):
        """Reset win/loss counters."""
        for p in self.parts:
            p.wins = 0
            p.losses = 0

    def to_system_hint(self, top_bid: Dict[str, Any]) -> str:
        """Convert top bid to a subtle hint for the agent."""
        if not top_bid:
            return ""
        return f"[internal: {top_bid['part']} wants to {top_bid['wants']}]"

    def _compute_relevance(
        self, part: Part, context_str: str, context: Dict[str, Any]
    ) -> float:
        """Compute how relevant a part is to the current context (0.0-1.0)."""
        score = 0.0

        # Keyword matching
        for kw in part.keywords:
            if kw in context_str:
                score += 0.2

        # Special triggers
        if part.name == "safety" and context.get("destructive_action"):
            score += 0.8
        if part.name == "quality" and context.get("user_complaint"):
            score += 0.7
        if part.name == "executor" and context.get("task_steps", 0) > 3:
            score += 0.5
        if part.name == "curiosity" and context.get("new_topic"):
            score += 0.6

        return min(1.0, score)
