"""Tests for agent.observer — periodic conversation observer."""

import json
import pytest
from agent.observer import ObserverAgent, Observation


class TestObserverAgent:
    def test_should_observe_every_n_turns(self):
        obs = ObserverAgent(enabled=True, check_interval=3)
        assert not obs.should_observe()  # turn 1
        obs.turn_count = 2
        assert not obs.should_observe()  # turn 2
        obs.turn_count = 3
        assert obs.should_observe()  # turn 3

    def test_should_observe_on_critical_event(self):
        obs = ObserverAgent(check_interval=10)
        # Tool error triggers observation regardless of interval
        assert obs.should_observe(tool_result='{"success": false, "error": "fail"}')
        # Memory pressure warning triggers observation
        assert obs.should_observe(tool_result='{"warning": "Memory pressure: 85%"}')

    def test_should_not_observe_on_normal_result(self):
        obs = ObserverAgent(check_interval=10)
        assert not obs.should_observe(tool_result='{"success": true}')

    def test_build_prompt(self):
        obs = ObserverAgent()
        prompt = obs._build_prompt(
            recent_messages=[{"role": "user", "content": "test"}],
            soul_rules={"rules": [{"id": "R1", "action": "be_nice"}]},
        )
        assert "observer" in prompt.lower() or "Observe" in prompt
        assert "test" in prompt

    def test_to_tool_message(self):
        obs = ObserverAgent()
        observation = Observation(
            has_insight=True,
            confidence=0.9,
            category="style",
            insight="Your reply was too long, be more concise.",
        )
        msg = obs.to_tool_message(observation)
        assert msg["role"] == "tool"
        assert msg["name"] == "internal_observer"
        parsed = json.loads(msg["content"])
        assert parsed["has_insight"] is True
        assert parsed["confidence"] == 0.9

    def test_to_tool_message_no_insight(self):
        obs = ObserverAgent()
        observation = Observation(has_insight=False, confidence=0.0)
        msg = obs.to_tool_message(observation)
        assert msg is None

    def test_turn_counter_increments(self):
        obs = ObserverAgent(enabled=True, check_interval=3)
        assert obs.turn_count == 0
        obs.increment_turn()
        assert obs.turn_count == 1
        obs.increment_turn()
        assert obs.turn_count == 2

    def test_reset(self):
        obs = ObserverAgent()
        obs.turn_count = 10
        obs.reset()
        assert obs.turn_count == 0

    def test_disabled_observer(self):
        obs = ObserverAgent(enabled=False)
        assert not obs.should_observe()
        obs.turn_count = 100
        assert not obs.should_observe()
