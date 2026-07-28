"""Aplicação Streamlit para geração assistida de perfis sintéticos."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from synthetic_br_profiles_gan.column_catalog import COLUMN_CATALOG, COLUMN_PRESETS
from synthetic_br_profiles_gan.exceptions import ConfigurationError, ModelSerializationError, PipelineError
from synthetic_br_profiles_gan.services.generation_service import GOVERNANCE_NOTICE
from synthetic_br_profiles_gan.ui.generation_adapter import (
    UIGenerationRequest,
    artifact_label,
    list_available_artifacts,
    run_ui_generation,
    validation_summary,
)
from synthetic_br_profiles_gan.ui.model_catalog import ModelCatalogEntry, model_catalog, model_catalog_by_name
from synthetic_br_profiles_gan.ui.ui_config import UIConfig, load_ui_config


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Renderiza a interface Streamlit."""
    import streamlit as st

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = _load_cached_config()
    st.set_page_config(page_title=config.title, layout="wide")
    _ensure_session_id(st)

    st.title(config.title)
    st.info(
        "Os dados gerados são sintéticos e não foram consultados ou validados em bases oficiais. "
        "A validade estrutural de documentos não comprova existência, regularidade ou associação a uma pessoa real. "
        "Os dados não devem ser utilizados para fraude, autenticação, identificação real ou acesso a serviços."
    )
    st.caption(
        "A ferramenta auxilia testes, ensino e pesquisa, mas não oferece garantia absoluta de anonimização "
        "ou ausência de coincidências com informações reais."
    )

    generate_tab, models_tab, governance_tab = st.tabs(["Gerar dados", "Conheça os modelos", "Sobre e governança"])
    with generate_tab:
        _render_generation_tab(st, config)
    with models_tab:
        _render_models_tab(st, config)
    with governance_tab:
        _render_governance_tab(st, config)


def _render_generation_tab(st: Any, config: UIConfig) -> None:
    model_entries = model_catalog_by_name()
    artifacts = _load_cached_artifacts(str(config.models_root))
    model_names = [entry.name for entry in model_catalog()]
    default_index = model_names.index(config.default_model) if config.default_model in model_names else 0

    st.subheader("Configuração da geração")
    model = st.radio(
        "Modelo",
        model_names,
        index=default_index,
        format_func=lambda value: model_entries[value].label,
        horizontal=True,
    )
    entry = model_entries[model]
    _render_selected_model(st, entry)

    selected_artifact_id: str | None = None
    generation_available = True
    if entry.requires_saved_artifact:
        available = artifacts.get(model, [])
        if not available:
            generation_available = False
            label = "CTGAN" if model == "ctgan" else "GAN simples"
            st.warning(
                f"Nenhum modelo {label} aprovado está disponível. "
                "O treinamento deve ser realizado previamente pela equipe responsável por meio da CLI."
            )
        else:
            selected = st.selectbox(
                "Artefato treinado",
                available,
                format_func=artifact_label,
            )
            selected_artifact_id = selected.artifact_id
            _render_artifact_details(st, selected)

    limit = int(config.limits[model])
    rows = st.number_input(
        "Quantidade de registros",
        min_value=int(config.min_rows),
        max_value=limit,
        value=min(int(config.default_rows), limit),
        step=1,
    )
    if int(rows) >= int(limit * 0.8):
        st.warning(
            "Gerações maiores podem consumir mais memória e levar mais tempo. "
            "O limite apresentado é operacional para esta interface e não representa a capacidade máxima absoluta do modelo."
        )

    output_format = st.selectbox(
        "Formato de saída",
        ["csv", "json", "parquet"],
        index=["csv", "json", "parquet"].index(config.default_format),
        format_func=lambda value: {
            "csv": "CSV — compatível com planilhas e ferramentas de análise",
            "json": "JSON — adequado para integrações e desenvolvimento",
            "parquet": "Parquet — indicado para análise de dados com preservação de tipos",
        }[value],
    )
    seed = st.number_input("Seed", min_value=0, value=int(config.default_seed), step=1)
    st.caption(
        "A seed ajuda a reproduzir a geração. Modelos neurais podem apresentar pequenas variações "
        "conforme backend, hardware e versões das bibliotecas."
    )

    selection_mode = st.radio("Seleção de colunas", ["Preset", "Personalizado"], horizontal=True)
    selected_columns: list[str] | None = None
    preset: str | None = None
    if selection_mode == "Preset":
        presets = list(COLUMN_PRESETS)
        default_preset_index = presets.index(config.default_preset) if config.default_preset in presets else 0
        preset = st.selectbox("Preset", presets, index=default_preset_index)
        st.caption(f"Colunas incluídas: {', '.join(COLUMN_PRESETS[preset])}")
    else:
        selected_columns = _render_custom_columns(st)
        if not selected_columns:
            generation_available = False
            st.error("Selecione pelo menos uma coluna para exportar.")
        st.caption(
            "As dependências necessárias são geradas internamente para preservar a coerência dos perfis, "
            "mas somente as colunas selecionadas serão exportadas."
        )

    if st.button("Gerar dados sintéticos", type="primary", disabled=not generation_available):
        try:
            request = UIGenerationRequest(
                model=model,
                rows=int(rows),
                output_format=output_format,
                seed=int(seed),
                config=config,
                artifact_id=selected_artifact_id,
                selected_columns=selected_columns,
                column_preset=preset,
                session_id=st.session_state["ui_session_id"],
            )
            with st.spinner("Gerando dados sintéticos..."):
                result = run_ui_generation(request)
            st.session_state["last_generation"] = result
            st.success("Geração concluída com sucesso.")
        except (ConfigurationError, ModelSerializationError, PipelineError, MemoryError, OSError) as exc:
            LOGGER.exception("streamlit_generation_error", extra={"error_type": type(exc).__name__})
            st.error(_friendly_error_message(exc))
        except Exception as exc:
            LOGGER.exception("streamlit_unexpected_generation_error", extra={"error_type": type(exc).__name__})
            st.error("Ocorreu uma falha inesperada durante a geração. Consulte os logs da aplicação.")

    if "last_generation" in st.session_state:
        _render_generation_result(st, st.session_state["last_generation"], config)


def _render_selected_model(st: Any, entry: ModelCatalogEntry) -> None:
    st.markdown(f"**{entry.label}**")
    st.write(entry.detailed_description)
    st.write(f"Situação: `{entry.status_label}`")
    st.write("Exige artefato treinado: " + ("sim" if entry.requires_saved_artifact else "não"))
    st.write("Indicado para: " + ", ".join(entry.best_for))
    if entry.limitations:
        st.write("Limitações:")
        for limitation in entry.limitations:
            st.write(f"- {limitation}")


def _render_artifact_details(st: Any, artifact: Any) -> None:
    st.caption(
        " · ".join(
            [
                f"Artefato: `{artifact.artifact_id}`",
                f"schema {artifact.schema_version}",
                f"seed {artifact.seed}" if artifact.seed is not None else "seed não registrada",
                f"treino {artifact.train_rows}" if artifact.train_rows is not None else "treino não registrado",
            ]
        )
    )


def _render_custom_columns(st: Any) -> list[str]:
    selected: list[str] = []
    groups: dict[str, list[Any]] = {}
    for entry in COLUMN_CATALOG:
        groups.setdefault(entry.group, []).append(entry)
    for group, entries in groups.items():
        with st.expander(group, expanded=True):
            for entry in entries:
                checked = st.checkbox(
                    f"{entry.label} (`{entry.name}`)",
                    value=bool(entry.default_selected),
                    key=f"column_{entry.name}",
                    help=entry.description,
                )
                suffix = "Coluna semelhante a dado pessoal." if entry.sensitive_like else "Coluna sintética de contexto."
                st.caption(f"{entry.description} {suffix}")
                if checked:
                    selected.append(entry.name)
    return selected


def _render_generation_result(st: Any, result: Any, config: UIConfig) -> None:
    manifest = result.manifest
    st.subheader("Resultado da geração")
    summary = {
        "Modelo": result.service_result.model,
        "Artefato": manifest.get("model_artifact") or "geração programática direta",
        "Registros": manifest.get("rows"),
        "Colunas exportadas": len(manifest.get("exported_columns", [])),
        "Formato": manifest.get("format"),
        "Seed": manifest.get("seed"),
        "Duração total (s)": round(float(manifest.get("timings", {}).get("total_seconds", 0.0)), 3),
        "Tamanho do arquivo (bytes)": manifest.get("output_size_bytes"),
        "Validação": "válida" if manifest.get("validation", {}).get("is_valid") else "inválida",
    }
    st.json(summary)

    st.markdown("**Amostra**")
    st.dataframe(result.dataset.head(int(config.preview_rows)), width="stretch")

    st.markdown("**Colunas exportadas**")
    st.write(", ".join(manifest.get("exported_columns", [])))

    st.markdown("**Validação estrutural**")
    for label, passed in validation_summary(manifest.get("validation", {})):
        st.write(f"{'✓' if passed else '✗'} {label}")

    dataset_bytes = result.output_path.read_bytes()
    manifest_bytes = result.manifest_path.read_bytes()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Baixar dataset",
            data=dataset_bytes,
            file_name=result.download_filename,
            mime=_mime_for_format(str(manifest.get("format"))),
        )
    with col2:
        st.download_button(
            "Baixar manifesto",
            data=manifest_bytes,
            file_name=result.manifest_download_filename,
            mime="application/json",
        )


def _render_models_tab(st: Any, config: UIConfig) -> None:
    st.subheader("Conheça os modelos")
    st.table(
        [
            {"Modelo": "Programático", "Treinamento": "Não exige", "Custo": "Baixo", "Situação": "Recomendado"},
            {"Modelo": "CTGAN", "Treinamento": "Exige", "Custo": "Alto", "Situação": "Avançado"},
            {"Modelo": "GAN simples", "Treinamento": "Exige", "Custo": "Médio", "Situação": "Experimental"},
        ]
    )
    artifacts = _load_cached_artifacts(str(config.models_root))
    for entry in model_catalog():
        st.markdown(f"### {entry.label}")
        st.write(entry.detailed_description)
        st.write("Indicado para: " + ", ".join(entry.best_for))
        st.write("Limitações:")
        for limitation in entry.limitations:
            st.write(f"- {limitation}")
        if entry.requires_saved_artifact:
            count = len(artifacts.get(entry.name, []))
            st.caption(f"Artefatos válidos disponíveis: {count}")
        else:
            st.caption("Disponível sem treinamento prévio.")


def _render_governance_tab(st: Any, config: UIConfig) -> None:
    st.subheader("Sobre e governança")
    st.write(
        "Esta interface é uma camada de apresentação sobre os serviços de geração do projeto. "
        "Ela não implementa regras próprias de CPF, documentos, pós-processamento, validação ou serialização."
    )
    st.write(
        "O fluxo gera internamente as 18 colunas finais, valida o schema completo e aplica a seleção de colunas "
        "somente na exportação."
    )
    st.write(GOVERNANCE_NOTICE)
    st.write(
        "A validade estrutural de identificadores sintéticos não comprova existência, regularidade ou associação "
        "a uma pessoa real. Os dados devem permanecer identificados como sintéticos."
    )
    st.write(
        "Os limites de linhas da interface são operacionais e configuráveis. Eles não representam a capacidade "
        "máxima absoluta dos modelos e não substituem benchmarks de capacidade."
    )
    st.write(
        "Os arquivos gerados pela interface são gravados em diretórios exclusivos sob "
        f"`{config.sessions_root}`. Esta primeira versão não implementa histórico persistente nem limpeza automática; "
        "a equipe responsável pode remover esses diretórios periodicamente."
    )
    st.write(
        "Modelos serializados devem ser carregados apenas de artefatos produzidos ou previamente aprovados pela aplicação. "
        "Não há upload de modelos nesta fase."
    )


def _friendly_error_message(exc: Exception) -> str:
    if isinstance(exc, ConfigurationError):
        return f"Configuração inválida: {exc}"
    if isinstance(exc, ModelSerializationError):
        return f"Artefato de modelo inválido ou incompatível: {exc}"
    if isinstance(exc, MemoryError):
        return "Memória insuficiente para concluir a geração solicitada."
    if isinstance(exc, OSError):
        return f"Falha operacional ao gerar ou exportar os dados: {exc}"
    return str(exc)


def _mime_for_format(output_format: str) -> str:
    if output_format == "csv":
        return "text/csv"
    if output_format == "json":
        return "application/json"
    if output_format == "parquet":
        return "application/octet-stream"
    return "application/octet-stream"


def _ensure_session_id(st: Any) -> None:
    if "ui_session_id" not in st.session_state:
        st.session_state["ui_session_id"] = uuid.uuid4().hex


def _load_cached_config() -> UIConfig:
    import streamlit as st

    @st.cache_data
    def _load() -> UIConfig:
        return load_ui_config("configs/ui.yaml")

    return _load()


def _load_cached_artifacts(models_root: str) -> dict[str, list[Any]]:
    import streamlit as st

    @st.cache_data(ttl=15)
    def _load(root: str) -> dict[str, list[Any]]:
        return list_available_artifacts(Path(root))

    return _load(models_root)
