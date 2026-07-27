"""Validadores estruturais para os perfis sinteticos."""

from synthetic_br_profiles_gan.validators.brazilian import (
    avaliar_regras_bruto,
    avaliar_regras_final,
    checar_unicidade,
    extrair_ddd,
    validar_cnh,
    validar_cpf,
    validar_formato_cpf,
    validar_formato_rg,
    validar_telefone,
    validar_titulo_eleitor,
)
from synthetic_br_profiles_gan.validators.structural import (
    ValidationResult,
    validate_core_dataframe,
    validate_profile_dataframe,
)

__all__ = [
    "ValidationResult",
    "avaliar_regras_bruto",
    "avaliar_regras_final",
    "checar_unicidade",
    "extrair_ddd",
    "validar_cnh",
    "validar_cpf",
    "validar_formato_cpf",
    "validar_formato_rg",
    "validar_telefone",
    "validar_titulo_eleitor",
    "validate_core_dataframe",
    "validate_profile_dataframe",
]

