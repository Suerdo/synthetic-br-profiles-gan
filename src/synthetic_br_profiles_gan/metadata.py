"""Dataset metadata and canonical schema definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json

from synthetic_br_profiles_gan.domain.brazil import (
    REGIONS,
    STATE_MUNICIPALITIES,
    STATE_REGION,
    all_ddds,
)


MODEL_COLUMNS = [
    "Idade",
    "Genero",
    "Regiao",
    "Estado",
    "Municipio",
    "Escolaridade",
    "Estado_Civil",
    "Ocupacao",
    "Renda",
    "Dependentes",
    "DDD",
]

DERIVED_COLUMNS = ["Nome", "Data_Nascimento", "CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"]

FINAL_COLUMNS = [
    "Nome",
    "Genero",
    "Data_Nascimento",
    "Idade",
    "Regiao",
    "Estado",
    "Municipio",
    "DDD",
    "Telefone",
    "Escolaridade",
    "Estado_Civil",
    "Ocupacao",
    "Renda",
    "Dependentes",
    "CPF",
    "CNH",
    "RG",
    "Titulo_Eleitor",
]

IDENTIFIER_COLUMNS = ["CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"]

PROXIMITY_EXCLUDED_COLUMNS = ["Nome", "Data_Nascimento", *IDENTIFIER_COLUMNS]

GENDER_CATEGORIES = ["Feminino", "Masculino", "Outro"]
EDUCATION_CATEGORIES = [
    "Fundamental",
    "Ensino Medio",
    "Superior Incompleto",
    "Superior Completo",
    "Pos-graduacao",
]
MARITAL_STATUS_CATEGORIES = ["Solteiro", "Casado", "Uniao Estavel", "Divorciado", "Viuvo"]
OCCUPATION_CATEGORIES = [
    "Estudante",
    "Servicos Gerais",
    "Tecnico",
    "Analista",
    "Coordenador",
    "Gerente",
    "Autonomo",
    "Aposentado",
]


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata for a single dataset column."""

    name: str
    kind: str
    required: bool = True
    categories: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    discrete: bool = False
    nullable: bool = False
    dependencies: list[str] = field(default_factory=list)
    description: str = ""


@dataclass(frozen=True)
class DatasetMetadata:
    """Schema, domains, and semantic dependencies for generated profiles."""

    columns: dict[str, ColumnMetadata]
    model_columns: list[str] = field(default_factory=lambda: list(MODEL_COLUMNS))
    final_columns: list[str] = field(default_factory=lambda: list(FINAL_COLUMNS))
    identifier_columns: list[str] = field(default_factory=lambda: list(IDENTIFIER_COLUMNS))
    proximity_excluded_columns: list[str] = field(default_factory=lambda: list(PROXIMITY_EXCLUDED_COLUMNS))
    structural_dependencies: dict[str, list[str]] = field(default_factory=dict)

    def required_columns(self, final: bool = True) -> list[str]:
        """Return required model or final columns."""
        columns = self.final_columns if final else self.model_columns
        return [name for name in columns if self.columns.get(name, ColumnMetadata(name, "unknown")).required]

    def categorical_columns(self, include_discrete_numeric: bool = True) -> list[str]:
        """Return columns that must be treated as categorical/discrete."""
        result = []
        for name in self.model_columns:
            column = self.columns[name]
            if column.kind == "categorical" or (include_discrete_numeric and column.discrete):
                result.append(name)
        return result

    def numeric_columns(self, include_discrete_numeric: bool = False) -> list[str]:
        """Return numeric columns used for statistical comparison."""
        result = []
        for name in self.model_columns:
            column = self.columns[name]
            if column.kind in {"integer", "numeric"} and (include_discrete_numeric or not column.discrete):
                result.append(name)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata into JSON-compatible primitives."""
        return {
            "columns": {name: asdict(column) for name, column in self.columns.items()},
            "model_columns": self.model_columns,
            "final_columns": self.final_columns,
            "identifier_columns": self.identifier_columns,
            "proximity_excluded_columns": self.proximity_excluded_columns,
            "structural_dependencies": self.structural_dependencies,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetMetadata":
        """Build metadata from a serialized dictionary."""
        columns = {
            name: ColumnMetadata(**column_payload)
            for name, column_payload in payload.get("columns", {}).items()
        }
        return cls(
            columns=columns,
            model_columns=list(payload.get("model_columns", MODEL_COLUMNS)),
            final_columns=list(payload.get("final_columns", FINAL_COLUMNS)),
            identifier_columns=list(payload.get("identifier_columns", IDENTIFIER_COLUMNS)),
            proximity_excluded_columns=list(
                payload.get("proximity_excluded_columns", PROXIMITY_EXCLUDED_COLUMNS)
            ),
            structural_dependencies=dict(payload.get("structural_dependencies", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Write metadata as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "DatasetMetadata":
        """Read metadata from JSON."""
        with Path(path).open(encoding="utf-8") as file:
            payload = json.load(file)
        return cls.from_dict(payload)


def default_metadata() -> DatasetMetadata:
    """Return the canonical metadata for this project."""
    municipalities = sorted({city for cities in STATE_MUNICIPALITIES.values() for city in cities})
    states = sorted(STATE_REGION)
    ddds = list(all_ddds())
    columns = {
        "Idade": ColumnMetadata("Idade", "integer", min_value=18, max_value=85),
        "Genero": ColumnMetadata("Genero", "categorical", categories=GENDER_CATEGORIES),
        "Regiao": ColumnMetadata("Regiao", "categorical", categories=list(REGIONS)),
        "Estado": ColumnMetadata(
            "Estado",
            "categorical",
            categories=states,
            dependencies=["Regiao"],
            description="State must belong to Regiao.",
        ),
        "Municipio": ColumnMetadata(
            "Municipio",
            "categorical",
            categories=municipalities,
            dependencies=["Estado"],
            description="Municipio must belong to Estado.",
        ),
        "Escolaridade": ColumnMetadata("Escolaridade", "categorical", categories=EDUCATION_CATEGORIES),
        "Estado_Civil": ColumnMetadata("Estado_Civil", "categorical", categories=MARITAL_STATUS_CATEGORIES),
        "Ocupacao": ColumnMetadata(
            "Ocupacao",
            "categorical",
            categories=OCCUPATION_CATEGORIES,
            dependencies=["Escolaridade", "Idade"],
        ),
        "Renda": ColumnMetadata("Renda", "numeric", min_value=800.0, max_value=50000.0),
        "Dependentes": ColumnMetadata("Dependentes", "integer", min_value=0, max_value=6),
        "DDD": ColumnMetadata(
            "DDD",
            "integer",
            categories=ddds,
            min_value=min(ddds),
            max_value=max(ddds),
            discrete=True,
            dependencies=["Estado"],
        ),
        "Nome": ColumnMetadata("Nome", "string"),
        "Data_Nascimento": ColumnMetadata(
            "Data_Nascimento",
            "date",
            dependencies=["Idade"],
            description="Birth date must produce Idade on the configured reference date.",
        ),
        "CPF": ColumnMetadata("CPF", "identifier"),
        "CNH": ColumnMetadata("CNH", "identifier"),
        "RG": ColumnMetadata("RG", "identifier"),
        "Titulo_Eleitor": ColumnMetadata("Titulo_Eleitor", "identifier"),
        "Telefone": ColumnMetadata("Telefone", "identifier", dependencies=["DDD", "Estado"]),
    }
    dependencies = {
        "Estado": ["Regiao"],
        "Municipio": ["Estado"],
        "DDD": ["Estado"],
        "Escolaridade": ["Idade"],
        "Ocupacao": ["Escolaridade", "Idade"],
        "Renda": ["Idade", "Escolaridade", "Ocupacao", "Regiao"],
        "Estado_Civil": ["Idade"],
        "Dependentes": ["Idade", "Estado_Civil"],
        "Data_Nascimento": ["Idade"],
        "Telefone": ["DDD", "Estado"],
    }
    return DatasetMetadata(columns=columns, structural_dependencies=dependencies)
