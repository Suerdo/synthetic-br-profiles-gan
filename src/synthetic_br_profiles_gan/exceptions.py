"""Domain-specific exceptions for the synthetic profile pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for expected pipeline failures."""


class ConfigurationError(PipelineError):
    """Raised when a configuration file is missing required settings."""


class ModelBackendUnavailable(PipelineError):
    """Raised when an optional model backend is not installed."""


class SyntheticModelError(PipelineError):
    """Raised when a synthesizer cannot be trained, loaded, or sampled."""


class ModelSerializationError(SyntheticModelError):
    """Raised when a saved synthesizer artifact is missing or corrupted."""


class StructuralValidationError(PipelineError):
    """Raised when generated data fails a required structural validation."""


class QualityGateError(PipelineError):
    """Raised when an approval-required command fails quality gates."""
