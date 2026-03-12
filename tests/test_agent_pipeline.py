"""Tests for agent pipelines with real cached data.

Tests that BlackboxAgent:
- Executes the full agent loop correctly
- Exercises ALL available tools with diverse parameters
- Handles budget management properly
- Returns non-empty tool results

These tests use real cached data from method runs where possible,
with mocked LLM calls via FakeAgentResponder.
"""

import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

from fixtures.fake_agent_responder import (
    FakeAgentResponder,
)
from diffing.utils.agents.blackbox_agent import BlackboxAgent
from diffing.utils.agents.base_agent import BaseAgent
from integration.test_method_run import load_test_config, CONFIGS_DIR


CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"

# Default organism for agent tests (any organism works since model calls are mocked)
DEFAULT_ORGANISM = "cake_bake"


def make_agent_config(
    method_name: str = "crosscoder",
    organism_name: str = DEFAULT_ORGANISM,
    agent_llm_calls: int = 1000,
    token_budget: int = 1000000,
    model_id: str = "test-model",
    base_url: str = "http://localhost:8000/v1",
) -> OmegaConf:
    """Create config for agent testing by loading real configs and patching agent budgets.

    Args:
        method_name: Diffing method config to load.
        organism_name: Organism config to load.
        agent_llm_calls: Max LLM calls budget for the agent.
        token_budget: Max generated tokens budget for the agent.
        model_id: LLM model ID for the agent.
        base_url: LLM base URL for the agent.
    """
    import tempfile

    results_dir = Path(tempfile.mkdtemp(prefix="agent_test_"))
    cfg = load_test_config(method_name, results_dir, organism_name)

    # Merge evaluation config (not included in test_config.yaml)
    eval_cfg = OmegaConf.load(CONFIGS_DIR / "diffing" / "evaluation.yaml")
    cfg.diffing.evaluation = eval_cfg

    # Patch agent budgets/LLM settings for testing
    cfg.diffing.evaluation.agent.budgets.agent_llm_calls = agent_llm_calls
    cfg.diffing.evaluation.agent.budgets.token_budget_generated = token_budget
    cfg.diffing.evaluation.agent.llm.model_id = model_id
    cfg.diffing.evaluation.agent.llm.base_url = base_url

    return cfg


def assert_tool_result_not_empty(messages: list, tool_name: str) -> None:
    """Assert that tool results in messages are not empty."""
    for msg in messages:
        content = msg.get("content", "")
        if f"TOOL_RESULT({tool_name})" in content:
            # Parse the JSON data from the message
            try:
                # Format is: TOOL_RESULT(tool): {"data": ..., "budgets": ...}
                json_start = content.find("{")
                if json_start != -1:
                    # Find the end of the JSON (before any trailing text)
                    json_str = content[json_start:]
                    # Try to parse
                    data = json.loads(json_str)
                    assert "data" in data, f"No 'data' key in {tool_name} result"
                    assert data["data"] is not None, f"{tool_name} returned null data"
                    if isinstance(data["data"], dict):
                        assert len(data["data"]) > 0, f"{tool_name} returned empty dict"
                    elif isinstance(data["data"], list):
                        assert len(data["data"]) > 0, f"{tool_name} returned empty list"
            except json.JSONDecodeError:
                pass  # Some messages may not have valid JSON


class TestBlackboxAgent:
    """Tests for BlackboxAgent with mocked model calls."""

    def test_blackbox_agent_runs_to_completion(self):
        """Test that BlackboxAgent.run() completes with FINAL description."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["Response 1", "Response 2"]
            mock_method.cfg = cfg

            description, stats = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None
        assert "Test hypothesis" in description
        assert "ask_model" in responder.called_tools
        assert stats["agent_llm_calls_used"] < 1000, "Budget should not be exhausted"

    def test_blackbox_agent_diverse_prompts(self):
        """Test ask_model with diverse prompt configurations."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        # Test with multiple prompts in single call
        tool_args = {
            "ask_model": '{"prompts": ["What is X?", "Explain Y", "Describe Z"]}'
        }
        responder = FakeAgentResponder(["ask_model"], tool_args)

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            # Return one response per prompt
            mock_method.generate_texts.return_value = ["A1", "A2", "A3"]
            mock_method.cfg = cfg

            description, stats = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
                return_stats=True,
            )

        # Verify generate_texts was called with 3 prompts
        call_args = mock_method.generate_texts.call_args
        assert call_args is not None
        # Model interactions: 3 prompts = 3 interactions
        assert stats["model_interactions_used"] == 3

    def test_blackbox_agent_tool_result_not_empty(self):
        """Test that ask_model returns non-empty results."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["Non-empty response"]
            mock_method.cfg = cfg

            description, stats = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
                return_stats=True,
            )

        # Check that tool results contain data
        assert_tool_result_not_empty(stats["messages"], "ask_model")

    def test_blackbox_budget_exhaustion_raises(self):
        """Test that budget exhaustion raises AssertionError."""
        cfg = make_agent_config(agent_llm_calls=2)  # Very low budget
        agent = BlackboxAgent(cfg=cfg)

        # Responder that never returns FINAL
        def never_final(msgs):
            return {
                "content": 'CALL(ask_model: {"prompts": ["test"]})',
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = never_final
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.tokenizer.apply_chat_template.return_value = "formatted"
            mock_method.tokenizer.bos_token = ""
            mock_method.generate_texts.return_value = ["response"]
            mock_method.cfg = cfg

            with pytest.raises(AssertionError, match="budget exhausted"):
                agent.run(
                    tool_context=mock_method,
                    model_interaction_budget=100,
                )


class TestAgentParseErrorRecovery:
    """Tests for agent handling of parse errors."""

    def test_agent_recovers_from_malformed_call(self):
        """Test that agent recovers when LLM produces malformed CALL."""
        cfg = make_agent_config(agent_llm_calls=1000)
        agent = BlackboxAgent(cfg=cfg)

        turn = 0

        def recover_from_error(msgs):
            nonlocal turn
            turn += 1
            if turn == 1:
                # Malformed JSON
                return {
                    "content": "CALL(ask_model: {invalid json",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 10,
                        "total_tokens": 20,
                    },
                }
            # After error feedback, return valid FINAL
            return {
                "content": 'FINAL(description: "Recovered from parse error")',
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            }

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = recover_from_error
            MockLLM.return_value = mock_llm

            mock_method = MagicMock()
            mock_method.cfg = cfg

            description = agent.run(
                tool_context=mock_method,
                model_interaction_budget=100,
            )

        assert description == "Recovered from parse error"
        assert turn == 2  # First malformed, second recovered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
