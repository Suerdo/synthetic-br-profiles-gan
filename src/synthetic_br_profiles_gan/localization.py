"""Normalização linguística dos valores textuais em português brasileiro."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable
from typing import Any

import pandas as pd


DATA_LOCALE = "pt-BR"
UNICODE_NORMALIZATION = "NFC"
CATEGORICAL_VOCABULARY_VERSION = 2


LEGACY_CATEGORY_ALIASES: dict[str, str] = {
    "Ensino Medio": "Ensino Médio",
    "Pos-graduacao": "Pós-graduação",
    "Uniao Estavel": "União Estável",
    "Viuvo": "Viúvo",
    "Servicos Gerais": "Serviços Gerais",
    "Tecnico": "Técnico",
    "Autonomo": "Autônomo",
    "Maceio": "Maceió",
    "Palmeira dos Indios": "Palmeira dos Índios",
    "Macapa": "Macapá",
    "Vitoria da Conquista": "Vitória da Conquista",
    "Brasilia": "Brasília",
    "Ceilandia": "Ceilândia",
    "Vitoria": "Vitória",
    "Goiania": "Goiânia",
    "Anapolis": "Anápolis",
    "Aparecida de Goiania": "Aparecida de Goiânia",
    "Sao Luis": "São Luís",
    "Cuiaba": "Cuiabá",
    "Varzea Grande": "Várzea Grande",
    "Rondonopolis": "Rondonópolis",
    "Tres Lagoas": "Três Lagoas",
    "Uberlandia": "Uberlândia",
    "Belem": "Belém",
    "Santarem": "Santarém",
    "Joao Pessoa": "João Pessoa",
    "Maringa": "Maringá",
    "Parnaiba": "Parnaíba",
    "Niteroi": "Niterói",
    "Petropolis": "Petrópolis",
    "Mossoro": "Mossoró",
    "Ji-Parana": "Ji-Paraná",
    "Rorainopolis": "Rorainópolis",
    "Caracarai": "Caracaraí",
    "Florianopolis": "Florianópolis",
    "Sao Paulo": "São Paulo",
    "Araguaina": "Araguaína",
}


def normalize_text_value(value: Any) -> Any:
    """Normaliza um valor textual para NFC e aplica aliases legados conhecidos."""
    if value is None:
        return value
    if isinstance(value, float) and math.isnan(value):
        return value
    if not isinstance(value, str):
        return value
    normalized = unicodedata.normalize(UNICODE_NORMALIZATION, value)
    aliased = LEGACY_CATEGORY_ALIASES.get(normalized, normalized)
    return unicodedata.normalize(UNICODE_NORMALIZATION, aliased)


def normalize_text_frame(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Retorna uma cópia do DataFrame com colunas textuais normalizadas em NFC."""
    normalized = df.copy()
    target_columns = list(columns) if columns is not None else list(normalized.columns)
    for column in target_columns:
        if column not in normalized.columns:
            continue
        series = normalized[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            normalized[column] = series.map(normalize_text_value)
    return normalized
