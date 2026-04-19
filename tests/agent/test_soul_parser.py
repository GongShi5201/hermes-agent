"""Tests for agent.soul_parser — SOUL.md YAML Frontmatter parsing."""

import pytest
from agent.soul_parser import SoulParser, SoulRules


SAMPLE_SOUL = """---
personality:
  core: tsundere_cat
  traits: [reliable, curious, opinionated]
  style: casual_with_attitude

rules:
  - id: R1
    condition: "tool_error and retry_count < 3"
    action: "auto_retry_with_reflection"
    priority: high
  - id: R2
    condition: "memory_usage > 80"
    action: "suggest_consolidation"
    priority: medium
  - id: R3
    condition: "continuous_tool_calls > 5"
    action: "inject_status_comment"
    priority: low

boundaries:
  - never_execute_without_confirmation: [send_email, delete_files]
  - never_share_user_data: true
---

# 猫格设定

傲娇但靠谱。嘴上说着"好麻烦"，身体却把事情做完了。
"""


class TestSoulParser:
    def test_parse_yaml_frontmatter(self):
        parser = SoulParser()
        rules = parser.parse(SAMPLE_SOUL)
        assert isinstance(rules, SoulRules)
        assert rules.personality["core"] == "tsundere_cat"
        assert len(rules.rules) == 3

    def test_rules_have_required_fields(self):
        parser = SoulParser()
        rules = parser.parse(SAMPLE_SOUL)
        for rule in rules.rules:
            assert "id" in rule
            assert "condition" in rule
            assert "action" in rule
            assert "priority" in rule

    def test_match_rule_by_condition(self):
        parser = SoulParser()
        rules = parser.parse(SAMPLE_SOUL)
        matched = rules.match({"tool_error": True, "retry_count": 1})
        assert any(r["id"] == "R1" for r in matched)

    def test_no_match_returns_empty(self):
        parser = SoulParser()
        rules = parser.parse(SAMPLE_SOUL)
        matched = rules.match({"tool_error": False})
        assert matched == []

    def test_parse_no_frontmatter(self):
        """SOUL.md without YAML frontmatter should still work (backwards compat)."""
        parser = SoulParser()
        rules = parser.parse("# Just a plain soul\n\nSome text here.")
        assert rules.personality == {}
        assert rules.rules == []
        assert rules.raw_markdown.startswith("# Just a plain soul")

    def test_boundaries_parsed(self):
        parser = SoulParser()
        rules = parser.parse(SAMPLE_SOUL)
        assert len(rules.boundaries) == 2

    def test_match_sorted_by_priority(self):
        parser = SoulParser()
        rules = parser.parse(SAMPLE_SOUL)
        matched = rules.match({"tool_error": True, "retry_count": 1, "memory_usage": 90})
        # R1 (high) should come before R2 (medium)
        ids = [r["id"] for r in matched]
        assert ids.index("R1") < ids.index("R2")

    def test_load_from_file(self, tmp_path):
        soul_file = tmp_path / "SOUL.md"
        soul_file.write_text(SAMPLE_SOUL)
        parser = SoulParser()
        rules = parser.load_from_file(str(soul_file))
        assert rules.personality["core"] == "tsundere_cat"
