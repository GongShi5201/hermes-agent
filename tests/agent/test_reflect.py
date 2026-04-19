"""Tests for agent.reflect — tool error reflection layer."""

import json
import pytest
from agent.reflect import ReflectionLayer, ReflectedSuggestion


class TestReflectionLayer:
    def test_is_error_json_success_false(self):
        layer = ReflectionLayer()
        assert layer._is_error('{"success": false, "error": "too big"}')

    def test_is_error_text(self):
        layer = ReflectionLayer()
        assert layer._is_error('Error: file not found')
        assert layer._is_error('Failed to connect')
        assert layer._is_error('Exception occurred')

    def test_is_not_error_success(self):
        layer = ReflectionLayer()
        assert not layer._is_error('{"success": true, "data": "ok"}')

    def test_is_not_error_plain_text(self):
        layer = ReflectionLayer()
        assert not layer._is_error("just plain text result")

    def test_is_not_error_empty(self):
        layer = ReflectionLayer()
        assert not layer._is_error("")
        assert not layer._is_error(None)

    def test_build_reflection_prompt(self):
        layer = ReflectionLayer()
        prompt = layer._build_prompt(
            tool_name="memory",
            tool_args={"action": "replace"},
            tool_result='{"success": false, "error": "exceeds limit"}',
            soul_rules={"rules": [{"id": "R1", "action": "auto_retry"}]},
        )
        assert "memory" in prompt
        assert "replace" in prompt
        assert "exceeds limit" in prompt

    def test_to_tool_message(self):
        layer = ReflectionLayer()
        suggestion = ReflectedSuggestion(
            has_issue=True,
            confidence=0.9,
            category="tool_error",
            suggestion="The replace failed because content is too long.",
        )
        msg = layer.to_tool_message(suggestion)
        assert msg["role"] == "tool"
        assert msg["name"] == "internal_observer"
        parsed = json.loads(msg["content"])
        assert parsed["has_issue"] is True
        assert parsed["confidence"] == 0.9

    def test_to_tool_message_no_issue(self):
        layer = ReflectionLayer()
        suggestion = ReflectedSuggestion(has_issue=False, confidence=0.0)
        msg = layer.to_tool_message(suggestion)
        assert msg is None

    def test_max_reflections_enforced(self):
        layer = ReflectionLayer(max_reflections=2)
        assert layer.can_reflect()
        layer.record_reflection()
        assert layer.can_reflect()
        layer.record_reflection()
        assert not layer.can_reflect()

    def test_reset(self):
        layer = ReflectionLayer(max_reflections=1)
        layer.record_reflection()
        assert not layer.can_reflect()
        layer.reset()
        assert layer.can_reflect()

    def test_reflect_skips_non_error(self):
        layer = ReflectionLayer()
        result = layer.reflect(
            tool_name="memory",
            tool_args={},
            tool_result='{"success": true}',
            soul_rules={},
        )
        assert result is None

    def test_reflect_skips_no_client(self):
        layer = ReflectionLayer()
        result = layer.reflect(
            tool_name="memory",
            tool_args={},
            tool_result='{"success": false, "error": "fail"}',
            soul_rules={},
            llm_client=None,
        )
        assert result is None
