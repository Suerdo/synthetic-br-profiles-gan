"""Preprocessamento numerico da base de calibracao."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """Aplica MinMaxScaler coluna a coluna e reverte para o espaco original."""

    def __init__(self) -> None:
        self.scalers: dict[str, MinMaxScaler] = {}
        self.columns: list[str] | None = None

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.columns = list(df.columns)
        processed = pd.DataFrame(index=df.index)

        for col in self.columns:
            scaler = MinMaxScaler()
            processed[col] = scaler.fit_transform(df[[col]]).ravel()
            self.scalers[col] = scaler

        return processed.to_numpy(dtype=np.float32)

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        if self.columns is None:
            raise RuntimeError("DataPreprocessor precisa ser ajustado antes do inverse_transform.")

        df = pd.DataFrame(data, columns=self.columns)
        for col in self.columns:
            df[col] = self.scalers[col].inverse_transform(df[[col]]).ravel()

        return df

