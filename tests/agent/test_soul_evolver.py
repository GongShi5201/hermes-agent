"""Tests for agent.soul_evolver — SOUL.md evolution at session end."""

import json
import os
import pytest
from agent.soul_evolver import SoulEvolver, EvolutionProposal


class TestSoulEvolver:
    def test_extract_lessons_from_errors(self):
        evolver = SoulEvolver()
        lessons = evolver.extract_lessons([
            {"tool": "memory", "error": "exceeds limit", "turn": 5},
            {"tool": "memory", "error": "exceeds limit", "turn": 12},
            {"tool": "terminal", "error": "command not found", "turn": 8},
        ])
        # memory error happened twice, should be extracted
        assert any("memory" in l.lower() for l in lessons)

    def test_extract_lessons_from_style_feedback(self):
        evolver = SoulEvolver()
        lessons = evolver.extract_lessons([], style_issues=[
            "回复太长了",
            "又用了清单格式",
        ])
        assert len(lessons) > 0

    def test_generate_proposal_no_lessons(self):
        evolver = SoulEvolver()
        proposal = evolver.generate_proposal([], current_soul="# Test Soul\n")
        assert proposal is None  # No changes needed

    def test_generate_proposal_with_lessons(self):
        evolver = SoulEvolver()
        proposal = evolver.generate_proposal(
            lessons=["Memory tool often hits limit — suggest consolidating before adding"],
            current_soul="# Test Soul\n\n## Rules\n\nNone yet.\n",
        )
        assert proposal is not None
        assert isinstance(proposal, EvolutionProposal)
        assert len(proposal.diff) > 0
        assert proposal.rationale

    def test_proposal_is_dry_run(self):
        """Proposal should not modify SOUL.md — just suggest changes."""
        evolver = SoulEvolver()
        proposal = evolver.generate_proposal(
            lessons=["Always test before committing"],
            current_soul="# Old Soul\n",
        )
        # Original should be unchanged
        assert "# Old Soul" in evolver._current_soul

    def test_proposal_diff_format(self):
        evolver = SoulEvolver()
        proposal = evolver.generate_proposal(
            lessons=["Check memory usage before adding entries"],
            current_soul="# Soul\n## Rules\n- Be nice\n",
        )
        if proposal:
            assert "---" in proposal.diff or "+++" in proposal.diff or "+" in proposal.diff
