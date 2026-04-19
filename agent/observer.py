"""Observer Agent — periodic conversation observer using cheap LLM.

Observes the conversation every N turns or on critical events (tool errors,
memory pressure). Outputs structured JSON with confidence threshold.
Injected as tool message to preserve prefix cache.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Observation:
    """Structured output from the observer."""
    has_insight: bool
    confidence: float = 0.0
    category: str = ""
    insight: str = ""


class ObserverAgent:
    """Lightweight observer that checks conversation health periodically.

    Design:
    - Triggered every N turns OR on critical events
    - Uses cheap model (MiMo) for cost efficiency
    - Structured JSON output with confidence threshold (0.7)
    - Injected as tool message (prefix cache safe)
    - Default disabled — opt-in via config
    """

    def __init__(
        self,
        enabled: bool = False,
        check_interval: int = 3,
        confidence_threshold: float = 0.7,
    ):
        self.enabled = enabled
        self.check_interval = check_interval
        self.confidence_threshold = confidence_threshold
        self.turn_count = 0

    def increment_turn(self):
        self.turn_count += 1

    def reset(self):
        self.turn_count = 0

    def should_observe(self, tool_result: str = None) -> bool:
        """Decide if observer should run this turn."""
        # Critical events bypass enabled check
        if tool_result and self._is_critical(tool_result):
            return True
        if not self.enabled:
            return False
        # Periodic check
        if self.turn_count > 0 and self.turn_count % self.check_interval == 0:
            return True
        return False

    def should_observe_critical_only(self, tool_result: str = None) -> bool:
        """Check only critical events (for per-tool-call observation)."""
        if not self.enabled:
            return False
        if tool_result and self._is_critical(tool_result):
            return True
        return False

    def _is_critical(self, result: str) -> bool:
        """Check if result indicates a critical event worth observing."""
        if not result:
            return False
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                # Tool error
                if data.get("success") is False:
                    return True
                # Memory pressure warning
                if "warning" in data:
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    def _build_prompt(
        self,
        recent_messages: List[Dict[str, Any]],
        soul_rules: Dict[str, Any],
    ) -> str:
        """Build observer prompt."""
        # Format last few messages for context
        msg_text = ""
        for m in recent_messages[-6:]:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:300]
            msg_text += f"[{role}]: {content}\n"

        rules_text = json.dumps(soul_rules.get("rules", []), ensure_ascii=False)

        return f"""You are an internal observer for an AI agent. Review the recent conversation.

Recent messages:
{msg_text}

Active SOUL rules: {rules_text}

Check for:
1. Does the agent's tone/style match the SOUL rules?
2. Is the agent being too verbose or too terse?
3. Is the agent missing something it should have noticed?
4. Is the agent repeating mistakes?

Respond with JSON only:
{{
  "has_insight": true/false,
  "confidence": 0.0-1.0,
  "category": "style|missed_opportunity|repetition|quality",
  "insight": "one sentence suggestion for the agent"
}}

Only set has_insight=true if confidence > 0.7. If everything looks fine, set has_insight=false."""

    def observe(
        self,
        recent_messages: List[Dict[str, Any]],
        soul_rules: Dict[str, Any],
        llm_client: Any = None,
        model: str = "xiaomi/mimo-v2-pro",
    ) -> Optional[Observation]:
        """Run observation. Returns Observation or None."""
        if not self.enabled:
            return None
        if llm_client is None:
            logger.debug("No LLM client for observer, skipping")
            return None

        try:
            prompt = self._build_prompt(recent_messages, soul_rules)
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            obs = Observation(
                has_insight=data.get("has_insight", False),
                confidence=data.get("confidence", 0.0),
                category=data.get("category", ""),
                insight=data.get("insight", ""),
            )
            if obs.confidence < self.confidence_threshold:
                obs.has_insight = False
            return obs
        except Exception as e:
            logger.debug("Observer failed: %s", e)
            return None

    def to_tool_message(self, observation: Observation) -> Optional[Dict[str, Any]]:
        """Convert observation to tool message for injection."""
        if not observation.has_insight:
            return None
        return {
            "role": "tool",
            "name": "internal_observer",
            "content": json.dumps(
                {
                    "has_insight": observation.has_insight,
                    "confidence": observation.confidence,
                    "category": observation.category,
                    "insight": observation.insight,
                },
                ensure_ascii=False,
            ),
        }
