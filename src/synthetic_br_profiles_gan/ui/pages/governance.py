"""Página de governança e histórico operacional."""

from __future__ import annotations

import html
from typing import Any

from synthetic_br_profiles_gan.ui.services.execution_history import filter_history, history_as_rows
from synthetic_br_profiles_gan.ui.services.governance_service import GovernanceSnapshot
from synthetic_br_profiles_gan.ui.theme import status_label
from synthetic_br_profiles_gan.ui.ui_config import UIConfig


def render_governance_page(st: Any, config: UIConfig, snapshot: GovernanceSnapshot) -> None:
    """Renderiza indicadores de governança, risco e auditoria."""
    st.header("Governança")
    st.write(
        "Esta página consolida evidências locais produzidas por manifestos, quality gates, validações e eventos sanitizados. "
        "Quando a evidência não existe, a interface informa explicitamente que o item não foi avaliado."
    )
    _render_glossary(st)

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Resumo Operacional",
            "Indicadores gerais extraídos dos manifestos e quality gates identificados localmente.",
        )
        _render_operational_metrics(st, snapshot)

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Modelo Neural Recomendado",
            "Fonte: training_manifest.json, approval_manifest.json, run_summary.csv e results.csv.",
        )
        st.dataframe(snapshot.recommended_neural_model, use_container_width=True)
        st.caption(
            "A aprovação representa uma decisão interna baseada nos critérios técnicos do projeto. "
            "Ela não constitui certificação externa, garantia de anonimização ou validação populacional oficial."
        )

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Qualidade dos Dados",
            "Fonte: manifesto de execução, validation.json e quality_gates.json quando disponíveis.",
        )
        st.dataframe(snapshot.quality_indicators, use_container_width=True)

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Privacidade",
            "Fonte: evaluation.json, métricas de privacidade e manifesto de execução quando disponíveis.",
        )
        st.dataframe(snapshot.privacy_indicators, use_container_width=True)

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Diversidade e Memorização",
            "Fonte: evaluation.json → privacy. Calculado sobre as 11 colunas-base, sem identificadores derivados.",
        )
        st.dataframe(snapshot.diversity_memorization_indicators, use_container_width=True)

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Realismo Condicional",
            "Fonte: evaluation.json → conditional_income e manifest.json → income_model_version.",
        )
        st.dataframe(snapshot.conditional_realism_indicators, use_container_width=True)

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Execuções Recentes",
            "Fonte: manifestos de execução, benchmark, treinamento e geração da interface.",
        )
        kinds = ["Todos"] + sorted({record.kind for record in snapshot.history})
        models = ["Todos"] + sorted({record.model for record in snapshot.history if record.model})
        statuses = ["Todos"] + sorted({record.status for record in snapshot.history if record.status})
        col_kind, col_model, col_status = st.columns(3)
        with col_kind:
            selected_kind = st.selectbox("Tipo", kinds)
        with col_model:
            selected_model = st.selectbox("Modelo", models)
        with col_status:
            selected_status = st.selectbox("Status", statuses)

        filtered = filter_history(
            snapshot.history,
            kind=None if selected_kind == "Todos" else selected_kind,
            model=None if selected_model == "Todos" else selected_model,
            status=None if selected_status == "Todos" else selected_status,
        )
        if filtered:
            st.dataframe(history_as_rows(filtered[:200]), use_container_width=True)
        else:
            st.info("Nenhum registro atende aos filtros selecionados.")

    with st.container(border=True):
        _render_governance_section_header(
            st,
            "Auditoria",
            f"Eventos sanitizados em `{config.audit_events_path}`. A auditoria não registra linhas geradas, CPF, nomes, telefones, IP, user agent ou traceback completo.",
        )
        if snapshot.audit_events:
            st.dataframe(snapshot.audit_events, use_container_width=True)
        else:
            st.info("Nenhum evento de auditoria foi registrado.")


def _render_governance_section_header(st: Any, title: str, description: str) -> None:
    st.markdown(
        (
            "<div class='sbp-governance-card-title'>"
            f"<h3>{html.escape(title)}</h3>"
            f"<p>{html.escape(description)}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_operational_metrics(st: Any, snapshot: GovernanceSnapshot) -> None:
    overview = snapshot.overview
    counts = _status_counts(snapshot)
    metrics = [
        ("Execuções registradas", str(overview["total_runs"]), "Quantidade de manifestos de execução identificados pela aplicação.", "manifesto de execução"),
        ("Aprovadas", str(counts["approved"]), "Execuções cujos quality gates obrigatórios foram atendidos.", "quality_gates.json"),
        ("Em quarentena", str(counts["quarantined"]), "Execuções tecnicamente concluídas, mas com alertas ou métricas informativas não atendidas.", "quality_gates.json"),
        ("Rejeitadas", str(counts["rejected"]), "Execuções que falharam em pelo menos um gate obrigatório.", "quality_gates.json"),
        ("Status operacional", status_label(str(overview["latest_status"])), "Situação técnica da execução mais recente identificada.", "manifesto de execução"),
        ("Modelo aprovado recente", str(overview["latest_approved_model_version"]), "Artefato neural aprovado mais recente quando há evidência.", "training_manifest.json"),
    ]
    columns = st.columns(3)
    for index, (label, value, help_text, source) in enumerate(metrics):
        with columns[index % 3]:
            st.metric(label, value, help=help_text)
            st.caption(f"Fonte: {source}")


def _status_counts(snapshot: GovernanceSnapshot) -> dict[str, int]:
    counts = {"approved": 0, "quarantined": 0, "rejected": 0}
    for record in snapshot.history:
        status = str(record.status or "").lower()
        if status in {"approved", "completed"}:
            counts["approved"] += 1
        elif "quarantine" in status or "quarantined" in status:
            counts["quarantined"] += 1
        elif "rejected" in status:
            counts["rejected"] += 1
    return counts


def _render_glossary(st: Any) -> None:
    with st.expander("Como interpretar os indicadores"):
        st.markdown(
            """
**Execuções registradas:** quantidade de manifestos de execução identificados pela aplicação.

**Aprovadas:** execuções sem falha nos quality gates obrigatórios.

**Em quarentena:** execuções tecnicamente concluídas, mas com alertas ou falhas em métricas informativas.

**Rejeitadas:** execuções que falharam em pelo menos um gate obrigatório.

**Linhas válidas:** linhas que passaram por todas as validações estruturais do schema final.

**Linhas inválidas:** linhas com problemas de domínio, consistência, nulidade, documento ou relacionamento estrutural.

**Identificadores duplicados:** repetições encontradas em CPF, CNH, RG, título de eleitor ou telefone dentro da mesma geração.

**Duplicidade de combinações-base:** repetição exata das 11 colunas produzidas pelo modelo. Identificadores derivados não participam.

**Correspondência exata com treino:** percentual de registros sintéticos cujas 11 colunas-base coincidem com pelo menos um registro de treinamento. A métrica isolada não comprova vazamento de dados.

**Correspondência exata com holdout:** métrica de controle contra registros não usados no treino. Ela ajuda a distinguir memorização de coincidências inerentes à distribuição.

**Realismo condicional:** capacidade de preservar distribuições dentro de contextos específicos, como renda de uma ocupação considerando escolaridade e idade.

**Cauda superior:** região dos valores mais altos da distribuição, avaliada por percentis como p95 e p99.

**Cobertura de ocupações:** proporção das ocupações canônicas reproduzidas pelo modelo na amostra avaliada.

**Distância de variação total:** diferença entre distribuições categóricas do conjunto sintético e da referência. Quanto menor, mais próximas estão as distribuições comparadas.

**Diferença de correlação:** maior diferença observada nas relações entre variáveis numéricas.

**Exact train match rate:** proporção de registros sintéticos cujas colunas-base coincidem exatamente com registros do treinamento.

**Risco de privacidade:** classificação derivada de métricas explícitas disponíveis. Não constitui garantia de anonimização.

**Status operacional:** situação técnica da execução mais recente identificada.

Quando uma métrica aparece como `Não avaliado`, esta execução não produziu essa métrica. Zero é usado somente quando o valor real registrado é zero.
"""
        )
