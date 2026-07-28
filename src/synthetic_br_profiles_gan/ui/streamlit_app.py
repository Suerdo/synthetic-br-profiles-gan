"""Aplicação Streamlit da plataforma de dados sintéticos brasileiros."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from synthetic_br_profiles_gan.ui.pages.generate import render_generation_page
from synthetic_br_profiles_gan.ui.pages.governance import render_governance_page
from synthetic_br_profiles_gan.ui.pages.models import render_models_page
from synthetic_br_profiles_gan.ui.services.audit_service import write_audit_event
from synthetic_br_profiles_gan.ui.services.governance_service import build_governance_snapshot
from synthetic_br_profiles_gan.ui.theme import ANONYMIZATION_WARNING, ui_css
from synthetic_br_profiles_gan.ui.ui_config import UIConfig, load_ui_config


LOGGER = logging.getLogger(__name__)

NAV_ITEMS = (
    ("Gerar dados", "▣"),
    ("Modelos", "◇"),
    ("Governança", "◉"),
)


def main() -> None:
    """Renderiza a aplicação Streamlit."""
    import streamlit as st

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = _load_cached_config()
    st.set_page_config(page_title=config.title, page_icon=None, layout="wide", initial_sidebar_state="expanded")
    st.markdown(ui_css(), unsafe_allow_html=True)
    _ensure_session_id(st)
    _audit_session_started(st, config)
    snapshot = build_governance_snapshot(config)
    current_page = _render_sidebar_menu(st)
    _audit_page_view(st, config, current_page)

    if current_page == "Gerar dados":
        render_generation_page(st, config, st.session_state["ui_session_id"])
    elif current_page == "Modelos":
        render_models_page(st, config, snapshot)
    elif current_page == "Governança":
        render_governance_page(st, config, snapshot)
    else:
        st.error("Página desconhecida.")

    st.markdown("---")
    st.markdown(
        f"<div class='sbp-final-warning'>{ANONYMIZATION_WARNING}</div>",
        unsafe_allow_html=True,
    )


def _ensure_session_id(st: Any) -> None:
    if "ui_session_id" not in st.session_state:
        st.session_state["ui_session_id"] = uuid.uuid4().hex


def _audit_session_started(st: Any, config: UIConfig) -> None:
    if st.session_state.get("ui_session_started_logged"):
        return
    write_audit_event(config.audit_events_path, "session_started", session_id=st.session_state["ui_session_id"])
    st.session_state["ui_session_started_logged"] = True


def _audit_page_view(st: Any, config: UIConfig, page: str) -> None:
    key = f"page_viewed_{page}"
    if st.session_state.get(key):
        return
    write_audit_event(config.audit_events_path, "page_viewed", {"page": page}, session_id=st.session_state["ui_session_id"])
    st.session_state[key] = True


def _render_sidebar_menu(st: Any) -> str:
    """Renderiza menu lateral com aparência de plataforma."""
    labels = tuple(label for label, _ in NAV_ITEMS)
    if st.session_state.get("current_page") not in labels:
        st.session_state["current_page"] = "Gerar dados"

    st.sidebar.markdown("<div class='sbp-sidebar-title'>Dados Sintéticos Brasileiro</div>", unsafe_allow_html=True)
    for label, icon in NAV_ITEMS:
        if st.session_state["current_page"] == label:
            st.sidebar.markdown(
                f"<div class='sbp-sidebar-item sbp-sidebar-active'>{icon} {label}</div>",
                unsafe_allow_html=True,
            )
        elif st.sidebar.button(f"{icon} {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state["current_page"] = label
            st.rerun()
    return str(st.session_state["current_page"])


def _load_cached_config() -> UIConfig:
    import streamlit as st

    @st.cache_data
    def _load() -> UIConfig:
        return load_ui_config("configs/ui.yaml")

    return _load()
