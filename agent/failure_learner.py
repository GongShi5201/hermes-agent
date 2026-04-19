"""Failure Learner — classifies tool failures and logs capability gaps.

Tracks affordance gaps (missing capabilities) in a JSONL file.
When the same gap appears 3+ times, suggests creating a skill.
"""

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FailureCategory(str, Enum):
    """Failure classification categories."""
    TOOL_ERROR = "tool_error"              # Tool ran but returned error
    INCORRECT_USAGE = "incorrect_usage"    # Wrong arguments or usage
    MISSING_AFFORDANCE = "missing_affordance"  # No tool/capability for this task
    TOOL_LIMITATION = "tool_limitation"    # Tool exists but can't handle this case


class FailureLearner:
    """Classifies tool failures and logs capability gaps.

    Affordance gaps are persisted to a JSONL file for tracking.
    When the same gap appears >= suggest_threshold times, should_suggest_skill()
    returns True.
    """

    def __init__(
        self,
        gap_log_path: Optional[str] = None,
        suggest_threshold: int = 3,
    ):
        self.gap_log_path = gap_log_path or os.path.join(
            os.path.expanduser("~/.hermes"), "affordance_gaps.jsonl"
        )
        self.suggest_threshold = suggest_threshold
        self._gap_counts: Counter = Counter()

    @staticmethod
    def _is_error_result(result: str) -> bool:
        """Quick check if tool result indicates an error."""
        if not result:
            return False
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                if data.get("success") is False:
                    return True
                if "error" in data:
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
        lower = result.lower()
        return any(kw in lower for kw in ["error", "failed", "exception", "traceback"])

    def classify(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
    ) -> FailureCategory:
        """Classify a tool failure based on context."""
        if not self._is_error_result(tool_result):
            return FailureCategory.TOOL_ERROR  # default even for non-errors
        result_lower = (tool_result or "").lower()

        # Missing affordance: tool doesn't exist
        if "tool not found" in result_lower or "unknown tool" in result_lower:
            return FailureCategory.MISSING_AFFORDANCE

        # Incorrect usage: bad arguments, syntax errors
        if any(kw in result_lower for kw in [
            "invalid", "required", "missing", "unexpected",
            "syntax error", "command not found",
        ]):
            return FailureCategory.INCORRECT_USAGE

        # Tool limitation: tool exists but can't handle this case
        if any(kw in result_lower for kw in [
            "not supported", "limitation", "cannot handle",
            "too large", "exceeds",
        ]):
            return FailureCategory.TOOL_LIMITATION

        # Default: tool error
        return FailureCategory.TOOL_ERROR

    def log_gap(
        self,
        capability: str,
        evidence: str,
        category: FailureCategory = FailureCategory.MISSING_AFFORDANCE,
    ):
        """Log an affordance gap to the JSONL file."""
        self._gap_counts[capability] += 1
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capability": capability,
            "category": category.value,
            "evidence": evidence[:500],
            "count": self._gap_counts[capability],
        }
        try:
            os.makedirs(os.path.dirname(self.gap_log_path), exist_ok=True)
            with open(self.gap_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to log affordance gap: %s", e)

    def count_gap(self, capability: str) -> int:
        """Get the count of times a capability gap was encountered."""
        return self._gap_counts.get(capability, 0)

    def should_suggest_skill(self, capability: str) -> bool:
        """Check if a skill should be suggested for this capability gap."""
        return self._gap_counts.get(capability, 0) >= self.suggest_threshold

    def get_gaps_summary(self) -> Dict[str, int]:
        """Get a summary of all capability gaps and their counts."""
        return dict(self._gap_counts)

    def load_existing_gaps(self):
        """Load gap counts from the existing JSONL file."""
        if not os.path.exists(self.gap_log_path):
            return
        try:
            with open(self.gap_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    cap = entry.get("capability", "")
                    if cap:
                        self._gap_counts[cap] += 1
        except Exception as e:
            logger.debug("Failed to load existing gaps: %s", e)
