"""Exceções específicas de domínio para o pipeline de perfis sintéticos."""

from __future__ import annotations


class PipelineError(Exception):
    """Classe-base para falhas esperadas do pipeline."""


class ConfigurationError(PipelineError):
    """Gerada quando um arquivo de configuração não contém definições obrigatórias."""


class ModelBackendUnavailable(PipelineError):
    """Gerada quando um backend opcional de modelo não está instalado."""


class SyntheticModelError(PipelineError):
    """Gerada quando um sintetizador não pode ser treinado, carregado ou amostrado."""


class ModelSerializationError(SyntheticModelError):
    """Gerada quando um artefato salvo do sintetizador está ausente ou corrompido."""


class StructuralValidationError(PipelineError):
    """Gerada quando os dados gerados falham em uma validação estrutural obrigatória."""


class QualityGateError(PipelineError):
    """Gerada quando um comando que exige aprovação falha nos quality gates."""
