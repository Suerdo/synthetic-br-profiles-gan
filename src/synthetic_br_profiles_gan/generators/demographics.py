"""Geradores demográficos e pós-processamento final de perfis."""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from faker import Faker

from synthetic_br_profiles_gan.calibration import generate_calibration_dataset
from synthetic_br_profiles_gan.domain.brazil import (
    STATE_DDDS,
    STATE_MUNICIPALITIES,
    region_for_state,
)
from synthetic_br_profiles_gan.generators.context import SyntheticProfileContext
from synthetic_br_profiles_gan.generators.identifiers import (
    gerar_cnh,
    gerar_cpf,
    gerar_rg,
    gerar_telefone,
    gerar_titulo_eleitor,
)
from synthetic_br_profiles_gan.localization import normalize_text_frame
from synthetic_br_profiles_gan.metadata import FINAL_COLUMNS, MODEL_COLUMNS


def criar_faker(seed: int | None = None) -> Faker:
    """Cria uma instância pt_BR do Faker com seed determinística opcional."""
    if seed is not None:
        Faker.seed(seed)
    fake = Faker("pt_BR")
    if seed is not None:
        fake.seed_instance(seed)
    return fake


def _as_date(value: datetime | date | None) -> date:
    if value is None:
        return datetime.now().date()
    if isinstance(value, datetime):
        return value.date()
    return value


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year - years)


def calcular_idade(data_nascimento: str | date | datetime, referencia: datetime | date | None = None) -> int:
    """Calcula a idade exata na data de referência."""
    reference = _as_date(referencia)
    birth = parse_data_nascimento(data_nascimento)
    age = reference.year - birth.year
    if (reference.month, reference.day) < (birth.month, birth.day):
        age -= 1
    return int(age)


def parse_data_nascimento(value: str | date | datetime) -> date:
    """Interpreta os formatos de data de nascimento aceitos pelo projeto."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported birth date format: {value}")


def gerar_data_nascimento_por_idade(
    idade: int,
    referencia: datetime | date | None = None,
    rng: random.Random | None = None,
    output_format: str = "%d/%m/%Y",
) -> str:
    """Gera uma data de nascimento que resulta exatamente em ``idade`` na ``referencia``."""
    random_source = rng if rng is not None else random
    reference = _as_date(referencia)
    start = _subtract_years(reference, int(idade) + 1) + timedelta(days=1)
    end = _subtract_years(reference, int(idade))
    ordinal = random_source.randint(start.toordinal(), end.toordinal())
    return date.fromordinal(ordinal).strftime(output_format)


def gerar_renda(rng: random.Random | None = None) -> float:
    """Gera uma renda mensal sintética assimétrica para chamadas legadas."""
    random_source = rng if rng is not None else random
    value = random_source.lognormvariate(8.0, 0.55)
    return round(max(800.0, min(value, 50000.0)), 2)


def gerar_pessoa_base(rng: random.Random | None = None) -> dict[str, Any]:
    """Gera uma linha canônica de calibração para chamadas legadas."""
    random_source = rng if rng is not None else random
    seed = random_source.randint(0, 2_147_483_647)
    return generate_calibration_dataset(1, seed=seed).iloc[0].to_dict()


def gerar_dataset_calibracao(n: int, rng: random.Random | None = None) -> pd.DataFrame:
    """Gera o dataset sintético de calibração usado no treinamento dos modelos."""
    seed = rng.randint(0, 2_147_483_647) if rng is not None else None
    return generate_calibration_dataset(num_rows=n, seed=seed)


def gerar_nome_por_genero(genero_label: str, fake: Faker) -> str:
    """Gera um nome fictício usando uma estratégia configurada por gênero."""
    if genero_label == "Feminino":
        return f"{fake.first_name_female()} {fake.last_name()}"
    if genero_label == "Masculino":
        return f"{fake.first_name_male()} {fake.last_name()}"
    return f"{fake.first_name()} {fake.last_name()}"


def _normalize_model_frame(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    normalized = normalize_text_frame(df)
    if "Sexo" in normalized.columns and "Genero" not in normalized.columns:
        normalized["Genero"] = normalized["Sexo"].round().astype(int).clip(0, 1).map({0: "Feminino", 1: "Masculino"})
        normalized = normalized.drop(columns=["Sexo"])

    missing = [column for column in MODEL_COLUMNS if column not in normalized.columns]
    if missing:
        filler = gerar_dataset_calibracao(len(normalized), rng=rng).reset_index(drop=True)
        for column in missing:
            normalized[column] = filler[column]

    normalized["Idade"] = pd.to_numeric(normalized["Idade"], errors="coerce").round().fillna(18).astype(int).clip(18, 85)
    normalized["Renda"] = pd.to_numeric(normalized["Renda"], errors="coerce").fillna(800.0).astype(float).clip(800, 50000)
    normalized["Dependentes"] = (
        pd.to_numeric(normalized["Dependentes"], errors="coerce").round().fillna(0).astype(int).clip(0, 6)
    )
    normalized["DDD"] = pd.to_numeric(normalized["DDD"], errors="coerce").round().astype("Int64")

    for index, row in normalized.iterrows():
        state = str(row["Estado"])
        if row["Regiao"] != region_for_state(state):
            normalized.at[index, "Regiao"] = region_for_state(state) or row["Regiao"]
        if row["Municipio"] not in STATE_MUNICIPALITIES.get(state, ()):
            normalized.at[index, "Municipio"] = rng.choice(STATE_MUNICIPALITIES.get(state, (str(row["Municipio"]),)))
        if int(row["DDD"]) not in STATE_DDDS.get(state, ()):
            normalized.at[index, "DDD"] = int(rng.choice(STATE_DDDS.get(state, (int(row["DDD"]),))))
    return normalize_text_frame(normalized[MODEL_COLUMNS])


def _unique_value(generator, used: set[str], max_attempts: int = 1000) -> str:
    for _ in range(max_attempts):
        value = str(generator())
        if value not in used:
            used.add(value)
            return value
    raise RuntimeError("Could not generate a unique synthetic value within max_attempts.")


def finalizar_perfis_sinteticos(
    df: pd.DataFrame,
    fake: Faker,
    referencia: datetime | date | None = None,
    rng: random.Random | None = None,
    date_format: str = "%Y-%m-%d",
) -> pd.DataFrame:
    """Adiciona campos derivados e identificadores contextuais às linhas do modelo."""
    random_source = rng if rng is not None else random
    normalized = _normalize_model_frame(df, random_source).reset_index(drop=True)

    used: dict[str, set[str]] = {
        "CPF": set(),
        "CNH": set(),
        "RG": set(),
        "Titulo_Eleitor": set(),
        "Telefone": set(),
    }
    rows: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        context = SyntheticProfileContext.from_mapping(row)
        profile = context.to_model_row()
        profile["Data_Nascimento"] = gerar_data_nascimento_por_idade(
            context.idade,
            referencia=referencia,
            rng=random_source,
            output_format=date_format,
        )
        profile["Nome"] = gerar_nome_por_genero(context.genero, fake)
        profile["CPF"] = _unique_value(lambda: gerar_cpf(random_source), used["CPF"])
        profile["CNH"] = _unique_value(lambda: gerar_cnh(random_source), used["CNH"])
        profile["RG"] = _unique_value(lambda: gerar_rg(random_source), used["RG"])
        profile["Titulo_Eleitor"] = _unique_value(
            lambda: gerar_titulo_eleitor(random_source),
            used["Titulo_Eleitor"],
        )
        profile["Telefone"] = _unique_value(
            lambda: gerar_telefone(random_source, estado=context.estado, ddd=context.ddd),
            used["Telefone"],
        )
        rows.append(profile)

    synthetic = pd.DataFrame(rows)
    synthetic["Renda"] = synthetic["Renda"].astype(float).round(2)
    return normalize_text_frame(synthetic[FINAL_COLUMNS])
