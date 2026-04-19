"""SOUL.md parser — extracts YAML Frontmatter rules + markdown body."""

import re
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SoulRules:
    """Structured representation of SOUL.md."""
    personality: Dict[str, Any] = field(default_factory=dict)
    rules: List[Dict[str, Any]] = field(default_factory=list)
    boundaries: List[Dict[str, Any]] = field(default_factory=list)
    raw_markdown: str = ""

    def match(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return rules whose conditions match the given context, sorted by priority."""
        matched = []
        for rule in self.rules:
            if self._eval_condition(rule.get("condition", ""), context):
                matched.append(rule)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        matched.sort(key=lambda r: priority_order.get(r.get("priority", "low"), 3))
        return matched

    def _eval_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate compound condition (supports ' and ')."""
        if not condition:
            return False
        parts = condition.split(" and ")
        for part in parts:
            part = part.strip()
            if not self._eval_single(part, context):
                return False
        return True

    def _eval_single(self, expr: str, context: Dict[str, Any]) -> bool:
        """Evaluate a single condition like 'tool_error' or 'retry_count < 3'."""
        # Boolean flag
        if expr in context:
            return bool(context[expr])
        # Comparison operators
        for op in ["<=", ">=", "<", ">", "==", "!="]:
            if op in expr:
                left, right = expr.split(op, 1)
                left = left.strip()
                right = right.strip()
                if left in context:
                    try:
                        return eval(f"{context[left]} {op} {right}")
                    except Exception:
                        return False
        return False


class SoulParser:
    """Parse SOUL.md with optional YAML Frontmatter."""

    _FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)

    def parse(self, content: str) -> SoulRules:
        """Parse SOUL.md content. Returns SoulRules even without frontmatter."""
        match = self._FRONTMATTER_RE.match(content)
        if not match:
            return SoulRules(raw_markdown=content)

        yaml_str, markdown = match.groups()
        try:
            data = yaml.safe_load(yaml_str) or {}
        except yaml.YAMLError:
            return SoulRules(raw_markdown=content)

        return SoulRules(
            personality=data.get("personality", {}),
            rules=data.get("rules", []),
            boundaries=data.get("boundaries", []),
            raw_markdown=markdown.strip(),
        )

    def load_from_file(self, path: str) -> SoulRules:
        """Load and parse SOUL.md from a file path."""
        with open(path, "r", encoding="utf-8") as f:
            return self.parse(f.read())
