"""Geracao de atributos demograficos e finalizacao dos perfis sinteticos."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

from synthetic_br_profiles_gan.generators.identifiers import (
    gerar_cnh,
    gerar_cpf,
    gerar_rg,
    gerar_telefone,
    gerar_titulo_eleitor,
)


def criar_faker(seed: int | None = None) -> Faker:
    """Cria uma instancia pt_BR do Faker com seed opcional."""
    if seed is not None:
        Faker.seed(seed)
    fake = Faker("pt_BR")
    if seed is not None:
        fake.seed_instance(seed)
    return fake


def gerar_data_nascimento_por_idade(
    idade: int,
    referencia: datetime | None = None,
    rng: random.Random | None = None,
) -> str:
    """Gera data de nascimento coerente com uma idade inteira aproximada."""
    random_source = rng if rng is not None else random
    data_referencia = referencia or datetime.now()
    dias = int(idade) * 365 + random_source.randint(0, 364)
    data = data_referencia - timedelta(days=dias)
    return data.strftime("%d/%m/%Y")


def gerar_renda(rng: random.Random | None = None) -> float:
    """Gera renda mensal sintetica entre R$ 1.200 e R$ 25.000."""
    random_source = rng if rng is not None else random
    return round(random_source.uniform(1200, 25000), 2)


def gerar_pessoa_base(rng: random.Random | None = None) -> dict[str, int | float]:
    """Gera uma linha numerica para calibracao da GAN tabular."""
    random_source = rng if rng is not None else random
    idade = random_source.randint(18, 65)
    sexo = random_source.choice([0, 1])
    renda = gerar_renda(random_source)
    return {"Idade": idade, "Sexo": sexo, "Renda": renda}


def gerar_dataset_calibracao(n: int, rng: random.Random | None = None) -> pd.DataFrame:
    """Gera a base sintetica de calibracao usada no treinamento da GAN."""
    return pd.DataFrame([gerar_pessoa_base(rng) for _ in range(n)])


def gerar_nome_por_genero(genero_label: str, fake: Faker) -> str:
    """Gera nome ficticio coerente com o rotulo de genero sintetico."""
    if genero_label == "Feminino":
        return f"{fake.first_name_female()} {fake.last_name()}"
    return f"{fake.first_name_male()} {fake.last_name()}"


def finalizar_perfis_sinteticos(
    df: pd.DataFrame,
    fake: Faker,
    referencia: datetime | None = None,
) -> pd.DataFrame:
    """Transforma a saida numerica da GAN no dataset final do projeto."""
    synthetic = df.copy()

    synthetic["Idade"] = synthetic["Idade"].round().astype(int).clip(18, 65)
    synthetic["Gênero"] = synthetic["Sexo"].round().astype(int).clip(0, 1)
    synthetic["Gênero"] = synthetic["Gênero"].map({0: "Feminino", 1: "Masculino"})
    synthetic = synthetic.drop(columns=["Sexo"])

    synthetic["Data_Nascimento"] = synthetic["Idade"].apply(
        lambda idade: gerar_data_nascimento_por_idade(idade, referencia=referencia)
    )
    synthetic["Nome"] = synthetic["Gênero"].apply(lambda genero: gerar_nome_por_genero(genero, fake))

    synthetic["CPF"] = [gerar_cpf() for _ in range(len(synthetic))]
    synthetic["CNH"] = [gerar_cnh() for _ in range(len(synthetic))]
    synthetic["RG"] = [gerar_rg() for _ in range(len(synthetic))]
    synthetic["Titulo_Eleitor"] = [gerar_titulo_eleitor() for _ in range(len(synthetic))]
    synthetic["Telefone"] = [gerar_telefone() for _ in range(len(synthetic))]
    synthetic["Renda"] = synthetic["Renda"].astype(float).round(2)

    synthetic = synthetic.drop(columns=["Idade"])
    return synthetic[
        [
            "Nome",
            "Gênero",
            "Data_Nascimento",
            "CPF",
            "CNH",
            "RG",
            "Titulo_Eleitor",
            "Telefone",
            "Renda",
        ]
    ]

