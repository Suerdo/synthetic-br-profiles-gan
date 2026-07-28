"""Página descritiva dos modelos disponíveis."""

from __future__ import annotations

import html
from typing import Any

from synthetic_br_profiles_gan.ui.components.cards import render_equal_summary_cards
from synthetic_br_profiles_gan.ui.model_catalog import ModelCatalogEntry, model_catalog
from synthetic_br_profiles_gan.ui.services.governance_service import GovernanceSnapshot
from synthetic_br_profiles_gan.ui.theme import badge_html
from synthetic_br_profiles_gan.ui.ui_config import UIConfig


def render_models_page(st: Any, config: UIConfig, snapshot: GovernanceSnapshot) -> None:
    """Renderiza explicações, status e artefatos dos modelos."""
    st.header("Modelos")
    st.write(
        "Esta área descreve as estratégias disponíveis sem transformar os resultados experimentais em garantias universais. "
        "Artefatos neurais tecnicamente válidos podem ser selecionados na geração com identificação clara de finalidade e riscos."
    )
    st.markdown(_comparison_table_html(), unsafe_allow_html=True)

    for entry in model_catalog():
        _render_model_section(st, entry)


def _render_model_section(st: Any, entry: ModelCatalogEntry) -> None:
    st.markdown(f"### {entry.label}")
    badges = [
        badge_html(entry.status, "success" if entry.recommended else "warning" if entry.experimental else "info"),
        badge_html(entry.category, "muted"),
        badge_html(f"Complexidade {entry.complexity_level}", "muted"),
    ]
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    st.write(entry.detailed_description)

    render_equal_summary_cards(st, entry.simple_summary, entry.technical_summary)
    st.markdown("**Usos Recomendados**")
    st.write(_indicado_para(entry.name))

    with st.expander("Ver Benefícios"):
        for item in entry.benefits:
            st.write(f"- {item}")
    with st.expander("Ver Limitações"):
        for item in entry.limitations:
            st.write(f"- {item}")
    st.markdown("**Realismo**")
    st.write(entry.realism_profile)
    with st.expander("Ver Privacidade e Governança"):
        for item in entry.privacy_considerations:
            st.write(f"- {item}")
        for item in entry.compliance_considerations:
            st.write(f"- {item}")

    with st.expander("Ver Evidências Experimentais"):
        st.write(entry.benchmark_notes)
        st.write(entry.capacity_notes)
        st.write("Essas observações dependem do ambiente, das versões das bibliotecas e das configurações usadas.")


def _comparison_table_html() -> str:
    rows = [
        ("Programático", "Não exige", "Baixo", "Recomendado"),
        ("CTGAN", "Exige", "Alto", "Avançado"),
        ("GAN Simples", "Exige", "Médio", "Experimental"),
    ]
    body = "".join(
        "<tr>"
        f"<td>{html.escape(model)}</td>"
        f"<td>{html.escape(training)}</td>"
        f"<td>{html.escape(cost)}</td>"
        f"<td>{html.escape(status)}</td>"
        "</tr>"
        for model, training, cost, status in rows
    )
    return (
        "<table class='sbp-comparison-table'>"
        "<thead><tr><th>Modelo</th><th>Treinamento</th><th>Custo</th><th>Situação</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _indicado_para(model: str) -> str:
    values = {
        "programmatic": (
            "Indicado para: Desenvolvimento, testes funcionais, demonstrações técnicas, "
            "validação de pipelines, ensino e grandes volumes locais."
        ),
        "ctgan": (
            "Indicado para: Experimentos tabulares, comparação com baseline programático, "
            "pesquisa metodológica e uso de artefatos treinados e avaliados pela equipe responsável."
        ),
        "simple_gan": (
            "Indicado para: Estudos acadêmicos sobre GANs, comparação de arquiteturas, "
            "análise de colapso categórico e experimentos metodológicos controlados."
        ),
    }
    return values[model]
