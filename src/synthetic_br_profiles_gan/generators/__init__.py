"""Geradores de atributos sinteticos brasileiros."""

from synthetic_br_profiles_gan.generators.demographics import (
    calcular_idade,
    criar_faker,
    finalizar_perfis_sinteticos,
    gerar_data_nascimento_por_idade,
    gerar_dataset_calibracao,
    gerar_nome_por_genero,
    gerar_pessoa_base,
    gerar_renda,
)
from synthetic_br_profiles_gan.generators.context import SyntheticProfileContext
from synthetic_br_profiles_gan.generators.identifiers import (
    gerar_cnh,
    gerar_cpf,
    gerar_rg,
    gerar_telefone,
    gerar_titulo_eleitor,
)

__all__ = [
    "SyntheticProfileContext",
    "calcular_idade",
    "criar_faker",
    "finalizar_perfis_sinteticos",
    "gerar_cnh",
    "gerar_cpf",
    "gerar_data_nascimento_por_idade",
    "gerar_dataset_calibracao",
    "gerar_nome_por_genero",
    "gerar_pessoa_base",
    "gerar_renda",
    "gerar_rg",
    "gerar_telefone",
    "gerar_titulo_eleitor",
]

