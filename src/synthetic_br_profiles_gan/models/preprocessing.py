"""Pré-processamento tabular para colunas numéricas e categóricas do modelo."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthetic_br_profiles_gan.metadata import DatasetMetadata


@dataclass
class NumericTransform:
    """Estado de mínimo e máximo de uma coluna numérica."""

    minimum: float
    maximum: float
    integer: bool = False


@dataclass
class CategoricalTransform:
    """Estado one-hot de uma coluna categórica."""

    categories: list[Any]


class DataPreprocessor:
    """Codifica dados tabulares em vetores numéricos densos e reverte a codificação."""

    def __init__(self, metadata: DatasetMetadata | None = None) -> None:
        self.metadata = metadata
        self.columns: list[str] | None = None
        self.numeric_transforms: dict[str, NumericTransform] = {}
        self.categorical_transforms: dict[str, CategoricalTransform] = {}
        self.feature_slices: dict[str, slice] = {}

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Ajusta o estado de pré-processamento a um DataFrame."""
        self.columns = list(self.metadata.model_columns if self.metadata else df.columns)
        offset = 0
        for column in self.columns:
            series = df[column]
            column_meta = self.metadata.columns[column] if self.metadata and column in self.metadata.columns else None
            is_categorical = bool(column_meta and (column_meta.kind == "categorical" or column_meta.discrete))
            if is_categorical or not pd.api.types.is_numeric_dtype(series):
                categories = list(column_meta.categories or []) if column_meta else sorted(series.dropna().unique().tolist())
                if not categories:
                    categories = sorted(series.dropna().unique().tolist())
                self.categorical_transforms[column] = CategoricalTransform(categories=categories)
                self.feature_slices[column] = slice(offset, offset + len(categories))
                offset += len(categories)
                continue

            numeric = pd.to_numeric(series, errors="coerce")
            minimum = float(column_meta.min_value) if column_meta and column_meta.min_value is not None else float(numeric.min())
            maximum = float(column_meta.max_value) if column_meta and column_meta.max_value is not None else float(numeric.max())
            if minimum == maximum:
                maximum = minimum + 1.0
            integer = bool(column_meta and column_meta.kind == "integer")
            self.numeric_transforms[column] = NumericTransform(minimum=minimum, maximum=maximum, integer=integer)
            self.feature_slices[column] = slice(offset, offset + 1)
            offset += 1
        return self

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Ajusta e transforma um DataFrame."""
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforma um DataFrame em uma matriz numérica."""
        if self.columns is None:
            raise RuntimeError("DataPreprocessor must be fitted before transform.")
        arrays: list[np.ndarray] = []
        for column in self.columns:
            if column in self.numeric_transforms:
                transform = self.numeric_transforms[column]
                values = pd.to_numeric(df[column], errors="coerce").fillna(transform.minimum).astype(float)
                scaled = (values.to_numpy() - transform.minimum) / (transform.maximum - transform.minimum)
                arrays.append(np.clip(scaled, 0.0, 1.0).reshape(-1, 1))
            else:
                transform = self.categorical_transforms[column]
                values = df[column].astype(str).tolist()
                category_index = {str(category): index for index, category in enumerate(transform.categories)}
                encoded = np.zeros((len(df), len(transform.categories)), dtype=np.float32)
                for row_index, value in enumerate(values):
                    index = category_index.get(value)
                    if index is not None:
                        encoded[row_index, index] = 1.0
                arrays.append(encoded)
        return np.hstack(arrays).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        """Inverte uma matriz numérica de volta para o domínio tabular."""
        if self.columns is None:
            raise RuntimeError("DataPreprocessor must be fitted before inverse_transform.")
        matrix = np.asarray(data)
        result: dict[str, Any] = {}
        for column in self.columns:
            column_slice = self.feature_slices[column]
            values = matrix[:, column_slice]
            if column in self.numeric_transforms:
                transform = self.numeric_transforms[column]
                restored = values.reshape(-1) * (transform.maximum - transform.minimum) + transform.minimum
                restored = np.clip(restored, transform.minimum, transform.maximum)
                if transform.integer:
                    restored = np.rint(restored).astype(int)
                result[column] = restored
            else:
                transform = self.categorical_transforms[column]
                indices = np.argmax(values, axis=1)
                result[column] = [transform.categories[int(index)] for index in indices]
        return pd.DataFrame(result, columns=self.columns)

    @property
    def output_dim(self) -> int:
        """Retorna a dimensionalidade codificada."""
        if self.columns is None:
            raise RuntimeError("DataPreprocessor must be fitted before output_dim is available.")
        return max(column_slice.stop for column_slice in self.feature_slices.values())

    def save(self, path: str | Path) -> Path:
        """Persiste o estado de pré-processamento com pickle."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as file:
            pickle.dump(self, file)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "DataPreprocessor":
        """Carrega o estado de pré-processamento a partir de pickle."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError("Serialized object is not a DataPreprocessor.")
        return loaded
