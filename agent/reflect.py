"""Reflection layer — triggers on tool errors, injects suggestions as tool messages."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReflectedSuggestion:
    """Structured output from the reflection layer."""
    has_issue: bool
    confidence: float = 0.0
    category: str = ""
    suggestion: str = ""


class ReflectionLayer:
    """Lightweight reflection triggered on tool failures.

    Design constraints:
    - Only triggers on tool errors (not every tool call)
    - Output injected as tool message (preserves prefix cache)
    - Max reflections per conversation enforced
    - Confidence threshold (0.7) to filter bad suggestions
    """

    def __init__(self, max_reflections: int = 3):
        self.max_reflections = max_reflections
        self._reflection_count = 0

    def can_reflect(self) -> bool:
        return self._reflection_count < self.max_reflections

    def record_reflection(self):
        self._reflection_count += 1

    def reset(self):
        self._reflection_count = 0

    def _is_error(self, result: str) -> bool:
        """Check if tool result indicates an error."""
        if not result:
            return False
        # JSON error
        try:
            data = json.loads(result)
            if isinstance(data, dict):
                if data.get("success") is False:
                    return True
                if "error" in data:
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
        # Text error patterns
        lower = result.lower()
        if any(kw in lower for kw in ["error", "failed", "exception", "traceback"]):
            return True
        return False

    def _build_prompt(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
        soul_rules: Dict[str, Any],
    ) -> str:
        """Build the reflection prompt for the cheap model."""
        rules_text = json.dumps(soul_rules.get("rules", []), ensure_ascii=False)
        return f"""You are an internal observer for an AI agent. A tool call just failed.

Tool: {tool_name}
Arguments: {json.dumps(tool_args, ensure_ascii=False)[:500]}
Result: {tool_result[:1000]}

Active rules: {rules_text}

Analyze the failure and respond with JSON only:
{{
  "has_issue": true/false,
  "confidence": 0.0-1.0,
  "category": "tool_error|incorrect_usage|soul_violation|missed_opportunity",
  "suggestion": "one sentence of what the agent should do differently"
}}

Only set has_issue=true if confidence > 0.7. Be brief. Respond with JSON only, no markdown."""

    def reflect(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: str,
        soul_rules: Dict[str, Any],
        llm_client: Any = None,
        model: str = "xiaomi/mimo-v2-pro",
    ) -> Optional[ReflectedSuggestion]:
        """Trigger reflection. Returns suggestion or None."""
        if not self.can_reflect():
            return None
        if not self._is_error(tool_result):
            return None
        if llm_client is None:
            logger.debug("No LLM client for reflection, skipping")
            return None

        self.record_reflection()

        try:
            prompt = self._build_prompt(tool_name, tool_args, tool_result, soul_rules)
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = response.choices[0].message.content.strip()
            # Extract JSON (handle markdown code blocks)
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            suggestion = ReflectedSuggestion(
                has_issue=data.get("has_issue", False),
                confidence=data.get("confidence", 0.0),
                category=data.get("category", ""),
                suggestion=data.get("suggestion", ""),
            )
            # Enforce confidence threshold
            if suggestion.confidence < 0.7:
                suggestion.has_issue = False
            return suggestion
        except Exception as e:
            logger.debug("Reflection failed: %s", e)
            return None

    def to_tool_message(self, suggestion: ReflectedSuggestion) -> Optional[Dict[str, Any]]:
        """Convert suggestion to a tool message for injection into messages list."""
        if not suggestion.has_issue:
            return None
        return {
            "role": "tool",
            "name": "internal_observer",
            "content": json.dumps(
                {
                    "has_issue": suggestion.has_issue,
                    "confidence": suggestion.confidence,
                    "category": suggestion.category,
                    "suggestion": suggestion.suggestion,
                },
                ensure_ascii=False,
            ),
        }
