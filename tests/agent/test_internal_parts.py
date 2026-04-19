"""Tests for agent.internal_parts — simplified internal parts competition system."""

import json
import pytest
from agent.internal_parts import InternalParts, Part


class TestPart:
    def test_part_creation(self):
        p = Part(name="executor", wants="complete tasks", intensity=0.5)
        assert p.name == "executor"
        assert p.intensity == 0.5
        assert p.wins == 0
        assert p.losses == 0

    def test_win_rate(self):
        p = Part(name="test", wants="test", intensity=0.5)
        assert p.win_rate == 0.0
        p.wins = 3
        p.losses = 1
        assert p.win_rate == 0.75


class TestInternalParts:
    def test_default_parts_exist(self):
        parts = InternalParts()
        names = [p.name for p in parts.parts]
        assert "executor" in names
        assert "quality" in names
        assert "curiosity" in names
        assert "safety" in names

    def test_bid_returns_relevant_parts(self):
        parts = InternalParts()
        bids = parts.bid({"task_steps": 5, "has_error": False})
        assert len(bids) > 0
        # All bids should have part, wants, score
        for b in bids:
            assert "part" in b
            assert "wants" in b
            assert "score" in b

    def test_safety_wins_on_dangerous_context(self):
        parts = InternalParts()
        bids = parts.bid({"destructive_action": True})
        # Safety should be in bids for dangerous context
        bid_names = [b["part"] for b in bids]
        assert "safety" in bid_names

    def test_quality_wins_on_complaint(self):
        parts = InternalParts()
        bids = parts.bid({"user_complaint": True})
        bid_names = [b["part"] for b in bids]
        assert "quality" in bid_names

    def test_evolve_adjusts_weights(self):
        parts = InternalParts()
        original_intensity = parts.get_part("quality").intensity
        parts.evolve({"quality_win": True})
        assert parts.get_part("quality").intensity > original_intensity

    def test_evolve_loss_decreases(self):
        parts = InternalParts()
        original = parts.get_part("executor").intensity
        parts.evolve({"executor_lose": True})
        assert parts.get_part("executor").intensity < original

    def test_get_top_bid(self):
        parts = InternalParts()
        top = parts.get_top_bid({"task_steps": 5})
        assert top is not None
        assert "part" in top

    def test_get_part(self):
        parts = InternalParts()
        p = parts.get_part("executor")
        assert p is not None
        assert p.name == "executor"
        assert parts.get_part("nonexistent") is None

    def test_reset(self):
        parts = InternalParts()
        parts.evolve({"quality_win": True})
        parts.evolve({"quality_win": True})
        parts.reset()
        for p in parts.parts:
            assert p.wins == 0
            assert p.losses == 0
