"""Catálogo estruturado das colunas finais exportáveis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from synthetic_br_profiles_gan.exceptions import ConfigurationError
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS


@dataclass(frozen=True)
class ColumnCatalogEntry:
    """Descrição reutilizável de uma coluna final do schema."""

    name: str
    label: str
    description: str
    group: str
    kind: str
    generated_by: str
    dependencies: tuple[str, ...] = ()
    sensitive_like: bool = False
    default_selected: bool = True


@dataclass(frozen=True)
class ColumnSelection:
    """Seleção resolvida de colunas para exportação."""

    requested_columns: tuple[str, ...] | None
    exported_columns: tuple[str, ...]
    mode: str
    preset: str | None
    internal_dependencies: dict[str, tuple[str, ...]]


IDENTIFICATION_GROUP = "Identificação sintética"
DEMOGRAPHICS_GROUP = "Demografia"
LOCATION_CONTACT_GROUP = "Localização e contato"
SOCIOECONOMIC_GROUP = "Perfil socioeconômico"


COLUMN_CATALOG: tuple[ColumnCatalogEntry, ...] = (
    ColumnCatalogEntry(
        name="Nome",
        label="Nome",
        description="Nome sintético gerado por estratégia compatível com o gênero do perfil.",
        group=IDENTIFICATION_GROUP,
        kind="string",
        generated_by="postprocessing",
        dependencies=("Genero",),
        sensitive_like=True,
    ),
    ColumnCatalogEntry(
        name="Genero",
        label="Gênero",
        description="Categoria sintética de gênero usada na base de calibração e no pós-processamento.",
        group=DEMOGRAPHICS_GROUP,
        kind="categorical",
        generated_by="model",
    ),
    ColumnCatalogEntry(
        name="Data_Nascimento",
        label="Data de nascimento",
        description="Data sintética calculada para ser coerente com a idade na data de referência.",
        group=IDENTIFICATION_GROUP,
        kind="date",
        generated_by="postprocessing",
        dependencies=("Idade",),
        sensitive_like=True,
    ),
    ColumnCatalogEntry(
        name="Idade",
        label="Idade",
        description="Idade sintética inteira dentro do domínio configurado.",
        group=DEMOGRAPHICS_GROUP,
        kind="integer",
        generated_by="model",
    ),
    ColumnCatalogEntry(
        name="Regiao",
        label="Região",
        description="Região brasileira sintética coerente com o estado gerado.",
        group=LOCATION_CONTACT_GROUP,
        kind="categorical",
        generated_by="model",
    ),
    ColumnCatalogEntry(
        name="Estado",
        label="Estado",
        description="Unidade federativa sintética pertencente à região gerada.",
        group=LOCATION_CONTACT_GROUP,
        kind="categorical",
        generated_by="model",
        dependencies=("Regiao",),
    ),
    ColumnCatalogEntry(
        name="Municipio",
        label="Município",
        description="Município sintético coerente com o estado gerado.",
        group=LOCATION_CONTACT_GROUP,
        kind="categorical",
        generated_by="model",
        dependencies=("Estado",),
    ),
    ColumnCatalogEntry(
        name="DDD",
        label="DDD",
        description="Código DDD sintético compatível com o estado gerado.",
        group=LOCATION_CONTACT_GROUP,
        kind="integer",
        generated_by="model",
        dependencies=("Estado",),
    ),
    ColumnCatalogEntry(
        name="Telefone",
        label="Telefone",
        description="Número sintético com DDD coerente com o estado gerado.",
        group=LOCATION_CONTACT_GROUP,
        kind="identifier",
        generated_by="postprocessing",
        dependencies=("Estado", "DDD"),
        sensitive_like=True,
    ),
    ColumnCatalogEntry(
        name="Escolaridade",
        label="Escolaridade",
        description="Categoria sintética de escolaridade relacionada probabilisticamente à idade.",
        group=SOCIOECONOMIC_GROUP,
        kind="categorical",
        generated_by="model",
        dependencies=("Idade",),
    ),
    ColumnCatalogEntry(
        name="Estado_Civil",
        label="Estado civil",
        description="Categoria sintética de estado civil relacionada probabilisticamente à idade.",
        group=DEMOGRAPHICS_GROUP,
        kind="categorical",
        generated_by="model",
        dependencies=("Idade",),
    ),
    ColumnCatalogEntry(
        name="Ocupacao",
        label="Ocupação",
        description="Ocupação sintética relacionada probabilisticamente à idade e à escolaridade.",
        group=SOCIOECONOMIC_GROUP,
        kind="categorical",
        generated_by="model",
        dependencies=("Escolaridade", "Idade"),
    ),
    ColumnCatalogEntry(
        name="Renda",
        label="Renda",
        description="Renda mensal sintética expressa em reais.",
        group=SOCIOECONOMIC_GROUP,
        kind="numeric",
        generated_by="model",
        dependencies=("Idade", "Escolaridade", "Ocupacao", "Regiao"),
    ),
    ColumnCatalogEntry(
        name="Dependentes",
        label="Dependentes",
        description="Quantidade sintética de dependentes relacionada à idade e ao estado civil.",
        group=DEMOGRAPHICS_GROUP,
        kind="integer",
        generated_by="model",
        dependencies=("Idade", "Estado_Civil"),
    ),
    ColumnCatalogEntry(
        name="CPF",
        label="CPF",
        description="Número sintético com dígitos verificadores estruturalmente válidos.",
        group=IDENTIFICATION_GROUP,
        kind="identifier",
        generated_by="postprocessing",
        sensitive_like=True,
    ),
    ColumnCatalogEntry(
        name="CNH",
        label="CNH",
        description="Número sintético semelhante à CNH, com verificadores conforme a regra local do projeto.",
        group=IDENTIFICATION_GROUP,
        kind="identifier",
        generated_by="postprocessing",
        sensitive_like=True,
    ),
    ColumnCatalogEntry(
        name="RG",
        label="RG",
        description="Identificador sintético no formato de RG definido pelo projeto.",
        group=IDENTIFICATION_GROUP,
        kind="identifier",
        generated_by="postprocessing",
        sensitive_like=True,
    ),
    ColumnCatalogEntry(
        name="Titulo_Eleitor",
        label="Título de eleitor",
        description="Número sintético de título de eleitor com dígitos verificadores locais.",
        group=IDENTIFICATION_GROUP,
        kind="identifier",
        generated_by="postprocessing",
        sensitive_like=True,
    ),
)


COLUMN_PRESETS: dict[str, tuple[str, ...]] = {
    "completo": tuple(FINAL_COLUMNS),
    "demografico": (
        "Genero",
        "Data_Nascimento",
        "Idade",
        "Regiao",
        "Estado",
        "Municipio",
        "Escolaridade",
        "Estado_Civil",
        "Ocupacao",
        "Renda",
        "Dependentes",
    ),
    "contato": ("Nome", "Regiao", "Estado", "Municipio", "DDD", "Telefone"),
    "documentos": ("Nome", "Data_Nascimento", "CPF", "CNH", "RG", "Titulo_Eleitor"),
    "minimo": ("Nome", "Idade", "Estado", "CPF"),
}


def catalog_by_name() -> dict[str, ColumnCatalogEntry]:
    """Retorna o catálogo indexado pelo nome canônico da coluna."""
    return {entry.name: entry for entry in COLUMN_CATALOG}


def available_column_names() -> tuple[str, ...]:
    """Retorna os nomes canônicos das colunas exportáveis."""
    return tuple(entry.name for entry in COLUMN_CATALOG)


def available_presets() -> tuple[str, ...]:
    """Retorna os nomes dos presets de colunas disponíveis."""
    return tuple(COLUMN_PRESETS)


def resolve_column_selection(
    selected_columns: Sequence[str] | None,
    preset: str | None = None,
    available_columns: Iterable[str] | None = None,
) -> ColumnSelection:
    """Valida e resolve uma seleção de colunas para exportação."""
    available = tuple(available_columns or available_column_names())
    if preset is not None and selected_columns is not None:
        raise ConfigurationError("Use --columns ou --preset, não ambos.")
    if preset is not None:
        if not isinstance(preset, str):
            raise ConfigurationError("O preset de colunas deve ser uma string.")
        preset_name = preset.strip()
        if preset_name not in COLUMN_PRESETS:
            raise ConfigurationError(
                f"Preset de colunas desconhecido: '{preset}'. "
                f"Presets disponíveis: {', '.join(available_presets())}."
            )
        columns = _validate_column_names(COLUMN_PRESETS[preset_name], available)
        return ColumnSelection(
            requested_columns=columns,
            exported_columns=columns,
            mode="preset",
            preset=preset_name,
            internal_dependencies=_dependencies_for(columns),
        )

    if selected_columns is None:
        columns = tuple(available)
        return ColumnSelection(
            requested_columns=None,
            exported_columns=columns,
            mode="all",
            preset=None,
            internal_dependencies=_dependencies_for(columns),
        )

    if isinstance(selected_columns, str):
        raise ConfigurationError("selected_columns deve ser uma sequência de strings, não uma string única.")
    columns = _validate_column_names(selected_columns, available)
    return ColumnSelection(
        requested_columns=columns,
        exported_columns=columns,
        mode="explicit",
        preset=None,
        internal_dependencies=_dependencies_for(columns),
    )


def _validate_column_names(values: Sequence[str], available: tuple[str, ...]) -> tuple[str, ...]:
    valid = set(available)
    columns: list[str] = []
    seen: set[str] = set()
    if len(values) == 0:
        raise ConfigurationError("A seleção de colunas não pode ficar vazia.")
    for value in values:
        if not isinstance(value, str):
            raise ConfigurationError("Todas as colunas selecionadas devem ser strings.")
        column = value.strip()
        if not column:
            raise ConfigurationError("Nomes de colunas não podem ser vazios.")
        if column in seen:
            raise ConfigurationError(f"A coluna '{column}' foi informada mais de uma vez.")
        if column not in valid:
            raise ConfigurationError(
                f"Coluna desconhecida: '{column}'. Colunas disponíveis: {', '.join(available)}."
            )
        seen.add(column)
        columns.append(column)
    return tuple(columns)


def _dependencies_for(columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    catalog = catalog_by_name()
    dependencies: dict[str, tuple[str, ...]] = {}
    for column in columns:
        entry = catalog[column]
        if entry.dependencies:
            dependencies[column] = entry.dependencies
    return dependencies
