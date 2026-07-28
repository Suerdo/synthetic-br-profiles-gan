"""Componentes visuais simples para cartões e blocos de informação."""

from __future__ import annotations

import html
from typing import Any, Iterable

from synthetic_br_profiles_gan.ui.theme import badge_html


def render_card(st: Any, title: str, body: str, *, badge: str | None = None, tone: str = "muted") -> None:
    """Renderiza um cartão textual sem lógica de negócio."""
    badge_markup = "" if badge is None else badge_html(badge, tone)
    st.markdown(
        f"""
<div class="sbp-card">
  {badge_markup}
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(body)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_equal_summary_cards(st: Any, simple_summary: str, technical_summary: str) -> None:
    """Renderiza resumos simples e técnicos com a mesma altura visual."""
    st.markdown(
        f"""
<div class="sbp-equal-card-grid">
  <div class="sbp-card">
    <h3>Resumo Simples</h3>
    <p>{html.escape(simple_summary)}</p>
  </div>
  <div class="sbp-card">
    <h3>Resumo Técnico</h3>
    <p>{html.escape(technical_summary)}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_cards(st: Any, metrics: Iterable[tuple[str, str, str]]) -> None:
    """Renderiza indicadores em colunas responsivas."""
    materialized = list(metrics)
    if not materialized:
        st.info("Não há indicadores disponíveis.")
        return
    columns = st.columns(min(len(materialized), 4))
    for index, (label, value, help_text) in enumerate(materialized):
        with columns[index % len(columns)]:
            st.metric(label, value, help=help_text or None)


def render_flow(st: Any, steps: Iterable[str]) -> None:
    """Renderiza um fluxo acessível em cartões sequenciais."""
    content = "".join(f"<div>{step}</div>" for step in steps)
    st.markdown(f"<div class='sbp-flow'>{content}</div>", unsafe_allow_html=True)
