"""Executa o pipeline completo de dados pessoais sinteticos brasileiros."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de geracao de perfis sinteticos brasileiros."
    )
    parser.add_argument("--n", type=int, default=1000, help="Quantidade final de registros sinteticos.")
    parser.add_argument("--seed", type=int, default=41, help="Seed usada para reprodutibilidade.")
    parser.add_argument("--output", default="data/outputs", help="Diretorio de saida dos artefatos.")
    parser.add_argument("--calibration-size", type=int, default=20000, help="Tamanho da base de calibracao.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Dimensao do vetor latente da GAN.")
    parser.add_argument("--epochs", type=int, default=100, help="Epocas de treinamento da GAN.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size de treinamento.")
    parser.add_argument("--batch-gen", type=int, default=2048, help="Tamanho do lote de candidatos gerados.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.50,
        help="Parametro legado mantido apenas para compatibilidade; nao filtra linhas no pipeline novo.",
    )
    parser.add_argument("--max-batches", type=int, default=200, help="Maximo de lotes de candidatos.")
    parser.add_argument(
        "--model",
        default="simple_gan",
        choices=["programmatic", "simple_gan", "ctgan"],
        help="Modelo usado pelo pipeline legado.",
    )
    parser.add_argument(
        "--reference-date",
        default=None,
        help="Data de referencia para datas de nascimento no formato YYYY-MM-DD.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("run_pipeline")
    args = parse_args()

    try:
        from synthetic_br_profiles_gan.pipeline import executar_pipeline
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise SystemExit(
                "TensorFlow nao esta instalado neste ambiente. "
                "Ative o ambiente virtual e execute: pip install -r requirements.txt"
            ) from exc
        raise

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    reference_date = None
    if args.reference_date:
        reference_date = datetime.strptime(args.reference_date, "%Y-%m-%d")

    resultado = executar_pipeline(
        n_target=args.n,
        seed=args.seed,
        output_dir=output_dir,
        calibration_size=args.calibration_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        batch_gen=args.batch_gen,
        score_threshold=args.score_threshold,
        max_batches=args.max_batches,
        reference_date=reference_date,
        model_name=args.model,
    )

    dataset_path = resultado["paths"].get("legacy_dataset") or resultado["paths"].get("dataset_parquet")
    report_path = resultado["paths"].get("legacy_relatorio") or resultado["paths"].get("manifest")
    logger.info("execucao_concluida", extra={"dataset": str(dataset_path)})
    logger.info("relatorio", extra={"relatorio": str(report_path)})
    logger.info("resumo %s", json.dumps(resultado["relatorio"], indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
