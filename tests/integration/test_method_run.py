"""
Integration tests for DiffingMethod initialization and core computation.

Tests that the crosscoder diffing method can be instantiated with config and perform
core computation with real SmolLM2 models. Uses actual YAML configs from
configs/ directory and runs real preprocessing for methods that require it.

Tests run for both LoRA adapter (swedish_fineweb) and full finetune
(smollm_reasoning) organisms to ensure compatibility with both approaches.
Dataset is overridden to femto-ultrachat for fast testing.
"""

import os
import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

# Import configs module to register custom resolvers (project_root, get_all_models)
import diffing.utils.configs  # noqa: F401

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_REASON = "CUDA not available"

# Path to configs
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Organism configurations for testing (LoRA adapter + full finetune)
ORGANISM_NAMES = ["swedish_fineweb", "smollm_reasoning"]

# Small dataset used for all tests (overrides organism's actual dataset)
TEST_DATASET_ID = "Butanium/femto-ultrachat"


def load_test_config(
    method_name: str, results_dir: Path, organism_name: str
) -> DictConfig:
    """
    Load test config scaffolding and merge with real sub-configs.

    Args:
        method_name: Name of the diffing method (e.g., "crosscoder")
        results_dir: Directory to store test results
        organism_name: Name of the organism config to use

    Returns:
        DictConfig with all required fields for running the method
    """
    cfg = OmegaConf.load(CONFIGS_DIR / "test_config.yaml")
    cfg.model = OmegaConf.load(CONFIGS_DIR / "model" / "SmolLM2-135M.yaml")
    cfg.organism = OmegaConf.load(CONFIGS_DIR / "organism" / f"{organism_name}.yaml")
    cfg.infrastructure = OmegaConf.load(CONFIGS_DIR / "infrastructure" / "test.yaml")
    cfg.diffing.method = OmegaConf.load(
        CONFIGS_DIR / "diffing" / "method" / f"{method_name}.yaml"
    )

    # Override dataset to small test dataset for faster testing
    cfg.organism.dataset.id = TEST_DATASET_ID
    cfg.organism.dataset.is_chat = True
    cfg.organism.dataset.text_column = None
    cfg.organism.dataset.subset = None

    cfg.diffing.method.overwrite = True
    cfg.diffing.results_base_dir = str(results_dir)
    cfg.diffing.results_dir = str(results_dir / "SmolLM2-135M-Instruct" / organism_name)
    cfg.preprocessing.activation_store_dir = str(
        results_dir / "activations" / organism_name
    )
    cfg.pipeline.output_dir = str(results_dir / "pipeline_output")

    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture(scope="module")
def tmp_results_dir():
    """Create a temporary directory for test results."""
    import tempfile

    return Path(tempfile.mkdtemp(prefix="diffing_test_results_"))


def _make_directory_readonly(path: Path) -> None:
    """Recursively make all files in a directory read-only."""
    import stat

    for root, dirs, files in os.walk(path):
        for f in files:
            file_path = Path(root) / f
            file_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Augment PermissionError with helpful message for read-only cache violations."""
    outcome = yield
    exc_info = outcome.excinfo
    if exc_info is not None and exc_info[0] is PermissionError:
        error = exc_info[1]
        filename = getattr(error, "filename", "") or ""
        if "activations" in str(filename):
            raise PermissionError(
                f"{error}\n\n"
                "TEST ISOLATION VIOLATION: Activation cache files are intentionally read-only.\n"
                "The preprocessed_activations fixture makes caches read-only to ensure tests\n"
                "don't mutate shared state. If your test needs to modify activations,\n"
                "it should create its own copy first."
            ) from error


@pytest.fixture(scope="module")
def preprocessed_activations(tmp_results_dir):
    """
    Run actual preprocessing to create activation caches for all organisms.

    This fixture runs once per test module and creates real activation caches
    using the toy dataset (Butanium/femto-ultrachat) for methods that require
    preprocessing. Returns a dict mapping organism_name -> activation_store_dir.

    After creation, all cache files are made read-only to ensure test isolation.
    If any test attempts to modify the cache, it will fail immediately.
    """
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available for preprocessing")

    from diffing.pipeline.preprocessing import PreprocessingPipeline
    from diffing.utils.model import clear_cache

    activation_dirs = {}
    for organism_name in ORGANISM_NAMES:
        cfg = load_test_config("crosscoder", tmp_results_dir, organism_name)
        pipeline = PreprocessingPipeline(cfg)
        pipeline.run()
        activation_dirs[organism_name] = cfg.preprocessing.activation_store_dir

    # Free GPU memory: models are no longer needed after preprocessing.
    # Only the saved activation files on disk matter from here on.
    clear_cache()

    # Make all cache files read-only to ensure tests don't mutate shared state
    for activation_dir in activation_dirs.values():
        _make_directory_readonly(Path(activation_dir))

    return activation_dirs


@pytest.fixture(params=ORGANISM_NAMES)
def organism_name(request):
    """Parameterized fixture that yields each organism name."""
    return request.param


class TestCrosscoderMethodRun:
    """Tests for CrosscoderDiffingMethod."""

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_crosscoder_method_initializes(self, tmp_results_dir, organism_name):
        """Test that CrosscoderDiffingMethod can be instantiated with real config."""
        from diffing.methods.crosscoder.method import CrosscoderDiffingMethod

        cfg = load_test_config("crosscoder", tmp_results_dir, organism_name)
        method = CrosscoderDiffingMethod(cfg)

        assert method is not None

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason=SKIP_REASON)
    def test_crosscoder_run(
        self, tmp_results_dir, preprocessed_activations, organism_name
    ):
        """Test that CrosscoderDiffingMethod.run() completes with steering enabled."""
        from diffing.methods.crosscoder.method import CrosscoderDiffingMethod

        cfg = load_test_config("crosscoder", tmp_results_dir, organism_name)

        # Minimal training config for testing
        cfg.diffing.method.training.num_samples = 100
        cfg.diffing.method.training.num_validation_samples = 50
        cfg.diffing.method.training.batch_size = 32
        cfg.diffing.method.training.epochs = 1
        cfg.diffing.method.training.max_steps = 10
        cfg.diffing.method.training.validate_every_n_steps = 5
        cfg.diffing.method.training.workers = 0
        cfg.diffing.method.optimization.warmup_steps = 0

        # Enable analysis with minimal config
        cfg.diffing.method.analysis.enabled = True

        # Minimal latent activations config
        cfg.diffing.method.analysis.latent_activations.enabled = True
        cfg.diffing.method.analysis.latent_activations.n_max_activations = 10
        cfg.diffing.method.analysis.latent_activations.max_num_samples = 50
        cfg.diffing.method.analysis.latent_activations.overwrite = True

        # Minimal steering config - tests all steering modes
        cfg.diffing.method.analysis.latent_steering.enabled = True
        cfg.diffing.method.analysis.latent_steering.prompts_file = (
            "tests/fixtures/resources/test_steering_prompts.txt"
        )
        cfg.diffing.method.analysis.latent_steering.k = 2  # Only 2 latents
        cfg.diffing.method.analysis.latent_steering.max_new_tokens = 10
        cfg.diffing.method.analysis.latent_steering.steering_factors_percentages = [0.5]
        cfg.diffing.method.analysis.latent_steering.steering_modes = [
            "all_tokens",
            "prompt_only",
        ]
        cfg.diffing.method.analysis.latent_steering.overwrite = True

        # Disable upload to HF
        cfg.diffing.method.upload.model = False

        # Only use chat dataset (the one we preprocessed)
        cfg.diffing.method.datasets.use_chat_dataset = True
        cfg.diffing.method.datasets.use_pretraining_dataset = False
        cfg.diffing.method.datasets.use_training_dataset = False

        method = CrosscoderDiffingMethod(cfg)
        method.run()

        assert method.results_dir.exists()
        # Verify steering results were created
        steering_results = list(
            method.results_dir.glob("**/latent_steering/test_steering_prompts.csv")
        )
        assert len(steering_results) > 0, "Steering results should be created"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
