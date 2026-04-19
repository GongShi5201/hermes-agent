"""Tests for agent.failure_learner — failure classification and affordance gap logging."""

import json
import os
import pytest
from agent.failure_learner import FailureLearner, FailureCategory


class TestFailureLearner:
    def test_classify_tool_error(self):
        learner = FailureLearner()
        cat = learner.classify(
            tool_name="memory",
            tool_args={"action": "replace"},
            tool_result='{"success": false, "error": "No entry matched"}',
        )
        assert cat == FailureCategory.TOOL_ERROR

    def test_classify_incorrect_usage(self):
        learner = FailureLearner()
        cat = learner.classify(
            tool_name="terminal",
            tool_args={"command": "invalid syntax here"},
            tool_result="Error: command not found",
        )
        assert cat == FailureCategory.INCORRECT_USAGE

    def test_classify_missing_affordance(self):
        learner = FailureLearner()
        cat = learner.classify(
            tool_name="unknown_tool",
            tool_args={},
            tool_result="Error: tool not found",
        )
        assert cat == FailureCategory.MISSING_AFFORDANCE

    def test_log_gap(self, tmp_path):
        log_file = str(tmp_path / "gaps.jsonl")
        learner = FailureLearner(gap_log_path=log_file)
        learner.log_gap(
            capability="pdf_editing",
            evidence="Tried to edit PDF but no tool available",
            category=FailureCategory.MISSING_AFFORDANCE,
        )
        assert os.path.exists(log_file)
        with open(log_file) as f:
            entry = json.loads(f.readline())
        assert entry["capability"] == "pdf_editing"
        assert entry["category"] == "missing_affordance"

    def test_count_gap_occurrences(self, tmp_path):
        log_file = str(tmp_path / "gaps.jsonl")
        learner = FailureLearner(gap_log_path=log_file)
        for _ in range(3):
            learner.log_gap(
                capability="pdf_editing",
                evidence="no tool",
                category=FailureCategory.MISSING_AFFORDANCE,
            )
        count = learner.count_gap("pdf_editing")
        assert count == 3

    def test_should_suggest_skill(self, tmp_path):
        log_file = str(tmp_path / "gaps.jsonl")
        learner = FailureLearner(gap_log_path=log_file, suggest_threshold=3)
        for _ in range(2):
            learner.log_gap("pdf_editing", "no tool", FailureCategory.MISSING_AFFORDANCE)
        assert not learner.should_suggest_skill("pdf_editing")
        learner.log_gap("pdf_editing", "no tool", FailureCategory.MISSING_AFFORDANCE)
        assert learner.should_suggest_skill("pdf_editing")

    def test_get_gaps_summary(self, tmp_path):
        log_file = str(tmp_path / "gaps.jsonl")
        learner = FailureLearner(gap_log_path=log_file)
        learner.log_gap("pdf_editing", "e1", FailureCategory.MISSING_AFFORDANCE)
        learner.log_gap("pdf_editing", "e2", FailureCategory.MISSING_AFFORDANCE)
        learner.log_gap("image_gen", "e3", FailureCategory.MISSING_AFFORDANCE)
        summary = learner.get_gaps_summary()
        assert summary["pdf_editing"] == 2
        assert summary["image_gen"] == 1
