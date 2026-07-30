"""Catálogo geográfico composto para representações neurais internas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from synthetic_br_profiles_gan.domain.brazil import (
    REGIONS,
    REGION_STATES,
    STATE_DDDS,
    STATE_MUNICIPALITIES,
    region_for_state,
)
from synthetic_br_profiles_gan.localization import normalize_text_value
from synthetic_br_profiles_gan.metadata import ColumnMetadata, DatasetMetadata, MODEL_COLUMNS


GEO_KEY_COLUMN = "Geo_Key"
GEOGRAPHY_MODEL_VERSION = 2
LEGACY_GEOGRAPHY_MODEL_VERSION = 1
GEOGRAPHY_CATALOG_VERSION = 1
UNKNOWN_GEOGRAPHY_KEY = "unknown_geography_key"
GEOGRAPHY_COLUMNS = ("Regiao", "Estado", "Municipio", "DDD")
GEOGRAPHY_V2_MODEL_COLUMNS = [
    GEO_KEY_COLUMN,
    "Idade",
    "Genero",
    "Escolaridade",
    "Estado_Civil",
    "Ocupacao",
    "Renda",
    "Dependentes",
]


@dataclass(frozen=True)
class GeographyCatalogEntry:
    """Combinação geográfica permitida pela tabela local do projeto."""

    geo_key: str
    regiao: str
    estado: str
    municipio: str
    ddd: int


def build_geography_catalog() -> tuple[GeographyCatalogEntry, ...]:
    """Constrói o catálogo geográfico determinístico a partir das fontes canônicas."""
    entries: list[GeographyCatalogEntry] = []
    counter = 1
    for region in REGIONS:
        for state in REGION_STATES.get(region, ()):
            for municipality in STATE_MUNICIPALITIES.get(state, ()):
                for ddd in STATE_DDDS.get(state, ()):
                    entries.append(
                        GeographyCatalogEntry(
                            geo_key=f"GEO_{counter:06d}",
                            regiao=str(region),
                            estado=str(state),
                            municipio=str(municipality),
                            ddd=int(ddd),
                        )
                    )
                    counter += 1
    return tuple(entries)


def geography_catalog_records() -> list[dict[str, Any]]:
    """Retorna registros JSON do catálogo geográfico."""
    return [asdict(entry) for entry in build_geography_catalog()]


def geography_catalog_checksum() -> str:
    """Calcula checksum SHA-256 estável do catálogo geográfico."""
    payload = json.dumps(geography_catalog_records(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def geography_key_categories() -> tuple[str, ...]:
    """Retorna as categorias válidas de `Geo_Key`."""
    return tuple(entry.geo_key for entry in build_geography_catalog())


def is_valid_geography_key(value: Any) -> bool:
    """Indica se um valor pertence ao catálogo de chaves geográficas."""
    return str(value) in _catalog_by_key()


def encode_geography_tuple(regiao: Any, estado: Any, municipio: Any, ddd: Any) -> str | None:
    """Codifica uma combinação geográfica válida como `Geo_Key`."""
    try:
        normalized = (
            str(normalize_text_value(regiao)),
            str(estado),
            str(normalize_text_value(municipio)),
            int(float(ddd)),
        )
    except (TypeError, ValueError):
        return None
    return _catalog_by_tuple().get(normalized)


def decode_geography_key(value: Any) -> GeographyCatalogEntry | None:
    """Decodifica `Geo_Key` em uma entrada do catálogo."""
    if value is None or pd.isna(value):
        return None
    return _catalog_by_key().get(str(value))


def validate_geography_mapping() -> dict[str, Any]:
    """Valida consistência interna do catálogo derivado das tabelas locais."""
    catalog = build_geography_catalog()
    keys = [entry.geo_key for entry in catalog]
    tuple_keys = [(entry.regiao, entry.estado, entry.municipio, entry.ddd) for entry in catalog]
    invalid_entries = [
        asdict(entry)
        for entry in catalog
        if region_for_state(entry.estado) != entry.regiao
        or entry.municipio not in STATE_MUNICIPALITIES.get(entry.estado, ())
        or entry.ddd not in STATE_DDDS.get(entry.estado, ())
    ]
    return {
        "geography_model_version": GEOGRAPHY_MODEL_VERSION,
        "geography_catalog_version": GEOGRAPHY_CATALOG_VERSION,
        "entries": int(len(catalog)),
        "unique_keys": int(len(set(keys))),
        "unique_tuples": int(len(set(tuple_keys))),
        "checksum": geography_catalog_checksum(),
        "is_valid": bool(len(keys) == len(set(keys)) == len(tuple_keys) == len(set(tuple_keys)) and not invalid_entries),
        "invalid_entries": invalid_entries,
        "ddd_limitation": (
            "A fonte local associa DDDs ao estado, não a cada município. Portanto, `Geo_Key` representa "
            "Região + Estado + Município + DDD permitido para o Estado no modelo sintético atual."
        ),
    }


def encode_geography_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui as quatro colunas geográficas por `Geo_Key` para treino neural."""
    encoded = df.copy()
    encoded[GEO_KEY_COLUMN] = [
        encode_geography_tuple(row["Regiao"], row["Estado"], row["Municipio"], row["DDD"])
        for _, row in encoded.iterrows()
    ]
    return encoded[GEOGRAPHY_V2_MODEL_COLUMNS].copy()


def decode_geography_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Restaura as quatro colunas geográficas públicas a partir de `Geo_Key`."""
    decoded = df.copy()
    regions: list[Any] = []
    states: list[Any] = []
    municipalities: list[Any] = []
    ddds: list[Any] = []
    for value in decoded.get(GEO_KEY_COLUMN, pd.Series([None] * len(decoded))):
        entry = decode_geography_key(value)
        if entry is None:
            regions.append(UNKNOWN_GEOGRAPHY_KEY)
            states.append(UNKNOWN_GEOGRAPHY_KEY)
            municipalities.append(UNKNOWN_GEOGRAPHY_KEY)
            ddds.append(0)
            continue
        regions.append(entry.regiao)
        states.append(entry.estado)
        municipalities.append(entry.municipio)
        ddds.append(entry.ddd)
    decoded["Regiao"] = regions
    decoded["Estado"] = states
    decoded["Municipio"] = municipalities
    decoded["DDD"] = ddds
    if GEO_KEY_COLUMN in decoded.columns:
        decoded = decoded.drop(columns=[GEO_KEY_COLUMN])
    return decoded[[column for column in MODEL_COLUMNS if column in decoded.columns]].copy()


def geography_v2_metadata(external_metadata: DatasetMetadata) -> DatasetMetadata:
    """Cria metadados internos com `Geo_Key` para a CTGAN geography v2."""
    columns = {
        name: column
        for name, column in external_metadata.columns.items()
        if name not in set(GEOGRAPHY_COLUMNS)
    }
    columns[GEO_KEY_COLUMN] = ColumnMetadata(
        name=GEO_KEY_COLUMN,
        kind="categorical",
        categories=list(geography_key_categories()),
        dependencies=list(GEOGRAPHY_COLUMNS),
        description="Chave composta interna para Região, Estado, Município e DDD.",
    )
    return DatasetMetadata(
        columns=columns,
        model_columns=list(GEOGRAPHY_V2_MODEL_COLUMNS),
        final_columns=list(external_metadata.final_columns),
        identifier_columns=list(external_metadata.identifier_columns),
        proximity_excluded_columns=list(external_metadata.proximity_excluded_columns),
        structural_dependencies={
            key: value
            for key, value in external_metadata.structural_dependencies.items()
            if key not in set(GEOGRAPHY_COLUMNS)
        },
    )


def _catalog_by_key() -> dict[str, GeographyCatalogEntry]:
    return {entry.geo_key: entry for entry in build_geography_catalog()}


def _catalog_by_tuple() -> dict[tuple[str, str, str, int], str]:
    return {
        (entry.regiao, entry.estado, entry.municipio, int(entry.ddd)): entry.geo_key
        for entry in build_geography_catalog()
    }
