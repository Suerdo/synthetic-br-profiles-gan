"""Persistencia dos artefatos gerados pelo pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def exportar_resultados(
    dataset: pd.DataFrame,
    relatorio: dict,
    output_dir: str | Path,
    dataset_filename: str = "dados_sinteticos_realistas.xlsx",
    report_filename: str = "relatorio_execucao.json",
) -> dict[str, Path]:
    """Exporta dataset e relatorio de execucao em um diretorio controlado."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_path = output_path / dataset_filename
    report_path = output_path / report_filename

    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        dataset.to_parquet(dataset_path, index=False)
    elif suffix == ".xlsx":
        dataset.to_excel(dataset_path, index=False)
    elif suffix == ".csv":
        dataset.to_csv(dataset_path, index=False)
    else:
        raise ValueError("Formato de dataset nao suportado. Use .parquet, .xlsx ou .csv.")

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(relatorio, file, indent=2, ensure_ascii=False, default=str)

    return {"dataset": dataset_path, "relatorio": report_path}

