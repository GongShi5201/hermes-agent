"""Soul Evolver — proposes SOUL.md updates at session end.

Does NOT modify SOUL.md directly. Generates a diff proposal that the user
confirms at next startup. Inspired by issue #11919 (SOUL.md should evolve).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvolutionProposal:
    """A proposed change to SOUL.md."""
    rationale: str
    lessons: List[str]
    diff: str
    new_content: str
    timestamp: str = ""


class SoulEvolver:
    """Generates SOUL.md evolution proposals from session lessons.

    Called at session end (or when _flush_memories_for_session runs).
    Analyzes tool errors, style feedback, and observer insights to
    propose SOUL.md rule additions or modifications.

    IMPORTANT: Never modifies SOUL.md directly. Only generates proposals
    that await user confirmation.
    """

    def __init__(self):
        self._current_soul: str = ""
        self._session_errors: List[Dict[str, Any]] = []
        self._session_insights: List[str] = []

    def record_error(self, tool_name: str, error: str, turn: int):
        """Record a tool error for later analysis."""
        self._session_errors.append({
            "tool": tool_name,
            "error": error[:200],
            "turn": turn,
        })

    def record_insight(self, insight: str):
        """Record an observer or reflection insight."""
        self._session_insights.append(insight)

    def extract_lessons(
        self,
        errors: List[Dict[str, Any]],
        style_issues: Optional[List[str]] = None,
    ) -> List[str]:
        """Extract actionable lessons from session data."""
        lessons = []

        # Group errors by tool
        tool_errors: Dict[str, List[str]] = {}
        for err in errors:
            tool = err.get("tool", "unknown")
            if tool not in tool_errors:
                tool_errors[tool] = []
            tool_errors[tool].append(err.get("error", ""))

        # Repeated errors → lesson
        for tool, errs in tool_errors.items():
            if len(errs) >= 2:
                lessons.append(
                    f"Repeated errors with {tool} ({len(errs)} times): "
                    f"should check limits/usage before calling"
                )

        # Style issues → lesson
        if style_issues:
            for issue in style_issues:
                lessons.append(f"Style issue: {issue}")

        # Insights → lesson
        for insight in self._session_insights:
            if insight and len(insight) > 10:
                lessons.append(f"Observer insight: {insight}")

        return lessons

    def generate_proposal(
        self,
        lessons: List[str],
        current_soul: str,
    ) -> Optional[EvolutionProposal]:
        """Generate a SOUL.md evolution proposal from lessons.

        Returns None if no changes are needed.
        """
        if not lessons:
            return None

        self._current_soul = current_soul

        # Build new rules from lessons
        new_rules = []
        for lesson in lessons:
            rule = self._lesson_to_rule(lesson)
            if rule:
                new_rules.append(rule)

        if not new_rules:
            return None

        # Generate the proposed new content
        new_content = self._apply_rules(current_soul, new_rules)
        diff = self._generate_diff(current_soul, new_content)

        return EvolutionProposal(
            rationale=f"Learned {len(lessons)} lesson(s) this session",
            lessons=lessons,
            diff=diff,
            new_content=new_content,
        )

    def _lesson_to_rule(self, lesson: str) -> Optional[str]:
        """Convert a lesson to a SOUL.md rule."""
        if "memory" in lesson.lower() and "limit" in lesson.lower():
            return "- Check memory usage before adding entries. Consolidate when above 70%."
        if "repeated" in lesson.lower():
            return f"- {lesson}"
        if "style" in lesson.lower():
            return f"- {lesson}"
        if "observer" in lesson.lower():
            return f"- {lesson}"
        return None

    def _apply_rules(self, current: str, rules: List[str]) -> str:
        """Apply new rules to SOUL.md content."""
        # Find or create a "Learned Rules" section
        if "## Learned Rules" in current:
            # Append to existing section
            parts = current.split("## Learned Rules")
            after = parts[1]
            # Find end of section (next ## or end of file)
            lines = after.split("\n")
            insert_idx = 1  # After the header
            for i, line in enumerate(lines[1:], 1):
                if line.startswith("## "):
                    insert_idx = i
                    break
            else:
                insert_idx = len(lines)

            rules_text = "\n".join(rules)
            new_lines = lines[:insert_idx] + [rules_text] + lines[insert_idx:]
            return parts[0] + "## Learned Rules" + "\n".join(new_lines)
        else:
            # Add new section at the end
            section = "\n\n## Learned Rules\n_Auto-generated from session experience._\n"
            section += "\n".join(rules) + "\n"
            return current.rstrip() + section

    def _generate_diff(self, old: str, new: str) -> str:
        """Generate a simple diff between old and new content."""
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        diff_lines = []

        # Simple approach: show what's added
        old_set = set(old_lines)
        for line in new_lines:
            if line not in old_set:
                diff_lines.append(f"+ {line}")

        if not diff_lines:
            return "(no changes)"

        return "\n".join(diff_lines)
