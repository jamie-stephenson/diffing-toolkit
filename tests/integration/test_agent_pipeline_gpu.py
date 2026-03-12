"""GPU integration tests for agent pipelines with real method instances.

Tests that BlackboxAgent runs with real method instances (real models, real GPU
computations). Only the agent LLM is faked via FakeAgentResponder.

Requires CUDA. Run on GPU node:
    lrun -J test_agent_gpu --qos=debug uv run pytest tests/integration/test_agent_pipeline_gpu.py -v
"""

import contextlib
import json
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

# Register custom resolvers (project_root, get_all_models)
import diffing.utils.configs  # noqa: F401

from integration.test_method_run import load_test_config, CONFIGS_DIR
from fixtures.fake_agent_responder import FakeAgentResponder

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)


def assert_no_tool_parameter_errors(stats: dict) -> None:
    """Assert that no TOOL_PARAMETER_ERROR appeared in agent messages.

    If the FakeAgentResponder sends known-good args and a TOOL_PARAMETER_ERROR
    still appears, it means the tool's own signature is wrong.
    """
    for msg in stats["messages"]:
        content = msg.get("content", "")
        assert "TOOL_PARAMETER_ERROR" not in content, f"Tool signature bug: {content}"


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Patch streamlit.spinner for non-Streamlit test environment.

    generate_texts() uses st.spinner() which requires a Streamlit runtime.
    """
    with patch("streamlit.spinner", return_value=contextlib.nullcontext()):
        yield


@pytest.fixture(scope="module")
def crosscoder_method_with_cache(mock_openai_server, tmp_path_factory):
    """Run crosscoder preprocessing to create activation caches for agent testing.

    Returns the method instance with models loaded.
    """
    from diffing.pipeline.preprocessing import PreprocessingPipeline
    from diffing.methods.crosscoder.method import CrosscoderDiffingMethod

    tmp_dir = tmp_path_factory.mktemp("crosscoder_gpu_agent")

    api_key_file = tmp_dir / "test_api_key.txt"
    api_key_file.write_text("test-api-key")

    cfg = load_test_config("crosscoder", tmp_dir, "swedish_fineweb")

    # Merge evaluation config for agent
    cfg.diffing.evaluation = OmegaConf.load(CONFIGS_DIR / "diffing" / "evaluation.yaml")
    cfg.diffing.evaluation.agent.llm.model_id = "test-model"
    cfg.diffing.evaluation.agent.llm.base_url = mock_openai_server.base_url
    cfg.diffing.evaluation.agent.llm.api_key_path = str(api_key_file)

    method = CrosscoderDiffingMethod(cfg)

    yield method

    method.clear_base_model()
    method.clear_finetuned_model()


class TestBlackboxAgentGPU:
    """Tests for BlackboxAgent with real model generation."""

    def test_blackbox_real_generation(self, crosscoder_method_with_cache):
        """Test that ask_model calls real generate_texts on GPU."""
        from diffing.utils.agents.blackbox_agent import BlackboxAgent

        method = crosscoder_method_with_cache
        agent = BlackboxAgent(cfg=method.cfg)

        responder = FakeAgentResponder(["ask_model"])

        with patch("diffing.utils.agents.base_agent.AgentLLM") as MockLLM:
            mock_llm = MagicMock()
            mock_llm.chat.side_effect = responder.get_response
            MockLLM.return_value = mock_llm

            description, stats = agent.run(
                tool_context=method,
                model_interaction_budget=100,
                return_stats=True,
            )

        assert description is not None
        assert "ask_model" in responder.called_tools

        # Verify tool results contain real generated text (not hardcoded "Response 1")
        found_result = False
        for msg in stats["messages"]:
            content = msg.get("content", "")
            if "TOOL_RESULT(ask_model)" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                assert json_start != -1
                data = json.loads(content[json_start:json_end])
                result = data["data"]
                assert "base" in result and "finetuned" in result
                for text in result["base"] + result["finetuned"]:
                    assert len(text) > 0
                found_result = True
                break
        assert found_result, "No ask_model TOOL_RESULT found in messages"
        assert_no_tool_parameter_errors(stats)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
