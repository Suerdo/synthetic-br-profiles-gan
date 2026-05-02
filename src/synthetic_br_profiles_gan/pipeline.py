"""Pipeline principal de geracao de perfis pessoais sinteticos brasileiros."""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam

from synthetic_br_profiles_gan.generators.demographics import (
    criar_faker,
    finalizar_perfis_sinteticos,
    gerar_dataset_calibracao,
)
from synthetic_br_profiles_gan.models.gan import (
    build_discriminator,
    build_gan,
    build_generator,
    train_gan,
)
from synthetic_br_profiles_gan.models.preprocessing import DataPreprocessor
from synthetic_br_profiles_gan.reports.execution import exportar_resultados
from synthetic_br_profiles_gan.utils.reproducibility import set_global_seed
from synthetic_br_profiles_gan.validators.brazilian import avaliar_regras_final, checar_unicidade


def gerar_sinteticos_com_metricas(
    generator,
    discriminator,
    preprocessor: DataPreprocessor,
    latent_dim: int,
    n_target: int = 1000,
    batch_gen: int = 2048,
    score_threshold: float = 0.50,
    max_batches: int = 200,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Gera amostras aceitas pela GAN e retorna metricas operacionais."""
    t0 = time.perf_counter()
    total_candidatos = 0
    total_aceitos = 0
    rejeicoes = Counter()
    aceitos_scaled = []
    aceitos_orig = []

    for _ in range(max_batches):
        noise = np.random.normal(0, 1, (batch_gen, latent_dim))
        gen_scaled = generator.predict(noise, verbose=0)
        total_candidatos += batch_gen

        scores = discriminator.predict(gen_scaled, verbose=0).reshape(-1)
        df_orig = preprocessor.inverse_transform(gen_scaled)

        idade = df_orig["Idade"]
        sexo = df_orig["Sexo"]
        renda = df_orig["Renda"]

        mask_dom = idade.between(18, 65) & renda.between(1200, 25000) & sexo.between(0, 1)
        mask_disc = scores >= score_threshold
        mask_ok = mask_dom & mask_disc
        n_ok = int(mask_ok.sum())

        rejeicoes["rejeitado_total"] += int((~mask_ok).sum())
        rejeicoes["rejeitado_disc"] += int((mask_dom & ~mask_disc).sum())
        rejeicoes["rejeitado_dom"] += int((~mask_dom).sum())

        if n_ok > 0:
            aceitos_scaled.append(gen_scaled[mask_ok])
            aceitos_orig.append(df_orig.loc[mask_ok])

            total_aceitos += n_ok
            if total_aceitos >= n_target:
                break

    if total_aceitos == 0:
        raise RuntimeError("Nenhuma amostra foi aceita. Reduza score_threshold ou aumente max_batches.")
    if total_aceitos < n_target:
        raise RuntimeError(
            f"Apenas {total_aceitos} amostras foram aceitas para n_target={n_target}. "
            "Reduza score_threshold ou aumente max_batches."
        )

    x_scaled = np.vstack(aceitos_scaled)[:n_target]
    df_final = pd.concat(aceitos_orig, ignore_index=True).iloc[:n_target].copy()

    tempo = time.perf_counter() - t0
    relatorio = {
        "n_target": int(n_target),
        "score_threshold": float(score_threshold),
        "batch_gen": int(batch_gen),
        "max_batches": int(max_batches),
        "total_candidatos": int(total_candidatos),
        "total_aceitos": int(n_target),
        "total_rejeitados": int(total_candidatos - n_target),
        "taxa_aceitacao": float(n_target / total_candidatos),
        "tempo_geracao_seg": float(tempo),
        "throughput_aceitos_por_seg": float(n_target / tempo),
        "rejeicoes": dict(rejeicoes),
        "resumo_univariado": {
            "idade_media": float(df_final["Idade"].mean()),
            "idade_dp": float(df_final["Idade"].std()),
            "renda_media": float(df_final["Renda"].mean()),
            "renda_mediana": float(df_final["Renda"].median()),
            "renda_dp": float(df_final["Renda"].std()),
            "sexo_prop_1": float((df_final["Sexo"].round().clip(0, 1) == 1).mean()),
        },
    }

    return df_final, x_scaled, relatorio


def executar_pipeline(
    n_target: int = 1000,
    seed: int = 41,
    output_dir: str | Path = "data/outputs",
    calibration_size: int = 20000,
    latent_dim: int = 16,
    epochs: int = 100,
    batch_size: int = 64,
    batch_gen: int = 2048,
    score_threshold: float = 0.50,
    max_batches: int = 200,
    reference_date: datetime | None = None,
) -> dict:
    """Executa o fluxo completo: calibracao, treino, geracao, validacao e exportacao."""
    set_global_seed(seed)
    fake = criar_faker(seed)
    reference_date = reference_date or datetime.now()

    print("Gerando base de calibracao...")
    real_data = gerar_dataset_calibracao(calibration_size)

    print("Pre-processando...")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.fit_transform(real_data)
    output_dim = processed_data.shape[1]

    print("Construindo e treinando a GAN...")
    generator = build_generator(latent_dim, output_dim)
    discriminator = build_discriminator(output_dim)
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
        metrics=["accuracy"],
    )
    gan = build_gan(generator, discriminator, latent_dim)
    train_gan(
        generator=generator,
        discriminator=discriminator,
        gan=gan,
        data=processed_data,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
    )

    print("Gerando dados sinteticos e metricas...")
    synthetic_raw, synthetic_scaled, relatorio = gerar_sinteticos_com_metricas(
        generator=generator,
        discriminator=discriminator,
        preprocessor=preprocessor,
        latent_dim=latent_dim,
        n_target=n_target,
        batch_gen=batch_gen,
        score_threshold=score_threshold,
        max_batches=max_batches,
    )

    synthetic_final = finalizar_perfis_sinteticos(
        synthetic_raw,
        fake=fake,
        referencia=reference_date,
    )

    identificadores = ["CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"]
    colisoes = checar_unicidade(synthetic_final, identificadores)
    relatorio.update(
        {
            "seed": int(seed),
            "calibration_size": int(calibration_size),
            "latent_dim": int(latent_dim),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "data_referencia": reference_date.strftime("%Y-%m-%d"),
            "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
            "colisoes_cpf": int(colisoes.get("CPF", 0)),
            "colisoes_identificadores": colisoes,
            "validacoes_finais": avaliar_regras_final(synthetic_final),
        }
    )

    paths = exportar_resultados(synthetic_final, relatorio, output_dir)
    return {
        "dataset": synthetic_final,
        "scaled": synthetic_scaled,
        "relatorio": relatorio,
        "paths": paths,
    }
