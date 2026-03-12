"""
Diffing pipeline for orchestrating model comparison methods.
"""

from typing import Dict, Any
from omegaconf import DictConfig

from .pipeline import Pipeline
from diffing.methods.crosscoder import CrosscoderDiffingMethod
from diffing.methods.diffing_method import DiffingMethod


def get_method_class(method_name: str) -> DiffingMethod:
    """Get the appropriate method class for a given method name."""
    if method_name == "crosscoder":
        return CrosscoderDiffingMethod
    else:
        raise ValueError(f"Unknown method: {method_name}")


class DiffingPipeline(Pipeline):
    """
    Pipeline for running diffing methods to analyze differences between models.

    This pipeline can orchestrate multiple diffing methods and chain them together.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg, name="DiffingPipeline")

        # Store diffing configuration
        self.diffing_cfg = cfg.diffing

        # Initialize diffing method
        self.diffing_method = get_method_class(self.diffing_cfg.method.name)(self.cfg)

    def validate_config(self) -> None:
        """Validate the diffing pipeline configuration."""
        super().validate_config()

        # Check diffing-specific configuration
        if not hasattr(self.cfg, "diffing"):
            raise ValueError("Configuration must contain 'diffing' section")

        if not hasattr(self.cfg.diffing, "method"):
            raise ValueError("Diffing configuration must contain 'method' section")

        if not hasattr(self.cfg.diffing.method, "name"):
            raise ValueError("Diffing method configuration must contain 'name'")

        # Check model configuration
        if not hasattr(self.cfg, "model"):
            raise ValueError("Configuration must contain 'model' section")

        if not hasattr(self.cfg, "organism"):
            raise ValueError("Configuration must contain 'organism' section")

        self.logger.info("Diffing pipeline configuration validation passed")

    def run(self) -> Dict[str, Any]:
        """
        Run the diffing pipeline.

        Returns:
            Dictionary containing pipeline metadata and status
        """
        self.logger.info(f"Running diffing method: {self.diffing_cfg.method.name}")

        # Run the diffing method (results are saved to disk internally)
        self.diffing_method.run()

        self.logger.info("Diffing pipeline completed successfully")
