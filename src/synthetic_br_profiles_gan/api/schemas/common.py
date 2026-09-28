"""Schemas comuns da API."""

from __future__ import annotations

from pydantic import BaseModel


class StrictBaseModel(BaseModel):
    """Modelo-base que rejeita campos desconhecidos."""

    class Config:
        extra = "forbid"
