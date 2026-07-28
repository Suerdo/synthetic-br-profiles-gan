"""Catálogo descritivo dos modelos exibidos na interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCatalogEntry:
    """Descrição de um sintetizador para apresentação ao usuário."""

    name: str
    label: str
    short_description: str
    detailed_description: str
    status_label: str
    recommended: bool
    experimental: bool
    requires_saved_artifact: bool
    best_for: tuple[str, ...]
    limitations: tuple[str, ...]


MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        name="programmatic",
        label="Programático — Recomendado",
        short_description=(
            "Gera dados por regras probabilísticas explícitas e dependências controladas."
        ),
        detailed_description=(
            "Gera dados por regras probabilísticas explícitas e dependências controladas. "
            "Não exige treinamento e oferece geração rápida, previsível e estruturalmente consistente."
        ),
        status_label="Recomendado",
        recommended=True,
        experimental=False,
        requires_saved_artifact=False,
        best_for=(
            "grandes volumes locais",
            "demonstrações",
            "testes funcionais",
            "cenários que exigem previsibilidade operacional",
        ),
        limitations=(
            "Distribuições determinadas pelas regras configuradas.",
            "Não aprende uma distribuição externa.",
            "A calibração programática não representa perfeitamente a população brasileira.",
        ),
    ),
    ModelCatalogEntry(
        name="ctgan",
        label="CTGAN — Modelo tabular avançado",
        short_description=(
            "Modelo generativo especializado em dados tabulares, com tratamento conjunto de atributos numéricos, categóricos e discretos."
        ),
        detailed_description=(
            "Modelo generativo especializado em dados tabulares, capaz de aprender conjuntamente "
            "atributos numéricos, categóricos e discretos."
        ),
        status_label="Avançado",
        recommended=False,
        experimental=False,
        requires_saved_artifact=True,
        best_for=(
            "experimentos tabulares",
            "comparações com baseline programático",
            "avaliações em que há artefato treinado e aprovado pela equipe",
        ),
        limitations=(
            "Exige artefato previamente treinado.",
            "Possui maior custo computacional.",
            "Apresentou desempenho melhor que a GAN simples nos experimentos atuais, sem implicar superioridade universal.",
            "O resultado depende do treinamento e dos hiperparâmetros.",
            "O modelo salvo deve ter sido produzido ou aprovado pela aplicação.",
        ),
    ),
    ModelCatalogEntry(
        name="simple_gan",
        label="GAN simples — Experimental",
        short_description=(
            "Baseline neural acadêmico baseado em uma GAN densa, mantido para comparação metodológica."
        ),
        detailed_description=(
            "Baseline neural acadêmico baseado em uma GAN densa. Foi mantido para comparação "
            "metodológica, mas apresentou limitações de qualidade e capacidade nos experimentos realizados."
        ),
        status_label="Experimental",
        recommended=False,
        experimental=True,
        requires_saved_artifact=True,
        best_for=(
            "comparação acadêmica",
            "demonstração de baseline neural",
            "reprodução dos experimentos do projeto",
        ),
        limitations=(
            "Modelo experimental — não recomendado como opção principal.",
            "Exige artefato previamente treinado.",
            "Apresentou limitações de qualidade e capacidade nos experimentos realizados.",
        ),
    ),
)


def model_catalog() -> tuple[ModelCatalogEntry, ...]:
    """Retorna o catálogo de modelos em ordem de apresentação."""
    return MODEL_CATALOG


def model_catalog_by_name() -> dict[str, ModelCatalogEntry]:
    """Retorna o catálogo de modelos indexado pelo identificador técnico."""
    return {entry.name: entry for entry in MODEL_CATALOG}
