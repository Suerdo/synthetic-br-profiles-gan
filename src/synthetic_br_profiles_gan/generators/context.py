"""Context object used before deriving Brazilian profile attributes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SyntheticProfileContext:
    """Canonical profile context before identifiers and derived fields are generated."""

    idade: int
    genero: str
    regiao: str
    estado: str
    municipio: str
    escolaridade: str
    ocupacao: str
    estado_civil: str
    renda: float
    dependentes: int
    ddd: int

    @classmethod
    def from_mapping(cls, row: dict[str, Any] | pd.Series) -> "SyntheticProfileContext":
        """Create a context from a pandas row or mapping with canonical column names."""
        return cls(
            idade=int(row["Idade"]),
            genero=str(row["Genero"]),
            regiao=str(row["Regiao"]),
            estado=str(row["Estado"]),
            municipio=str(row["Municipio"]),
            escolaridade=str(row["Escolaridade"]),
            ocupacao=str(row["Ocupacao"]),
            estado_civil=str(row["Estado_Civil"]),
            renda=float(row["Renda"]),
            dependentes=int(row["Dependentes"]),
            ddd=int(row["DDD"]),
        )

    def to_model_row(self) -> dict[str, Any]:
        """Return the context using canonical model column names."""
        return {
            "Idade": self.idade,
            "Genero": self.genero,
            "Regiao": self.regiao,
            "Estado": self.estado,
            "Municipio": self.municipio,
            "Escolaridade": self.escolaridade,
            "Estado_Civil": self.estado_civil,
            "Ocupacao": self.ocupacao,
            "Renda": self.renda,
            "Dependentes": self.dependentes,
            "DDD": self.ddd,
        }
