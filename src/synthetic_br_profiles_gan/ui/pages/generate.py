"""Página de geração de dados sintéticos."""

from __future__ import annotations

import logging
from typing import Any

from synthetic_br_profiles_gan.column_catalog import COLUMN_CATALOG, COLUMN_PRESETS
from synthetic_br_profiles_gan.exceptions import ConfigurationError, ModelSerializationError, PipelineError
from synthetic_br_profiles_gan.ui.generation_adapter import (
    UIGenerationRequest,
    artifact_label,
    artifact_status_label,
    artifact_status_warning,
    list_generation_artifacts,
    run_ui_generation,
    validation_summary,
)
from synthetic_br_profiles_gan.ui.model_catalog import ModelCatalogEntry, model_catalog, model_catalog_by_name
from synthetic_br_profiles_gan.ui.services.audit_service import write_audit_event
from synthetic_br_profiles_gan.ui.theme import GOVERNANCE_WARNING, INSTITUTIONAL_DESCRIPTION, badge_html
from synthetic_br_profiles_gan.ui.ui_config import UIConfig


LOGGER = logging.getLogger(__name__)


def render_generation_page(st: Any, config: UIConfig, session_id: str) -> None:
    """Renderiza o formulário de geração assistida."""
    st.header("Gerar Dados Sintéticos")
    st.markdown(
        f"""
<div class="sbp-governance-alert">
  <strong>Atenção:</strong> {GOVERNANCE_WARNING}
</div>
""",
        unsafe_allow_html=True,
    )
    st.write(INSTITUTIONAL_DESCRIPTION)

    model_entries = model_catalog_by_name()
    artifacts = list_generation_artifacts(config)
    model_names = [entry.name for entry in model_catalog()]
    default_model = config.default_model if config.default_model in model_names else "programmatic"
    if default_model != "programmatic" and not artifacts.get(default_model):
        default_model = "programmatic"

    _render_step(st, 1, "Modelo")
    model = st.radio(
        "Modelo de geração",
        model_names,
        index=model_names.index(default_model),
        format_func=lambda value: model_entries[value].label,
        horizontal=True,
    )
    if st.session_state.get("last_model_selected") != model:
        write_audit_event(config.audit_events_path, "model_selected", {"model": model}, session_id=session_id)
        st.session_state["last_model_selected"] = model
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
                f"Nenhum artefato {label} válido foi encontrado. "
                "Treine ou disponibilize um modelo por meio da CLI para utilizá-lo nesta interface."
            )
        else:
            selected = st.selectbox(
                "Artefato do modelo",
                available,
                index=0,
                format_func=artifact_label,
                key=f"artifact_selector_{model}",
            )
            selected_artifact_id = selected.artifact_id
            _render_artifact_details(st, selected)

    _render_step(st, 2, "Volume e Reprodutibilidade")
    limit = int(config.limits[model])
    col_rows, col_seed = st.columns(2)
    with col_rows:
        rows = st.number_input(
            "Quantidade de registros",
            min_value=int(config.min_rows),
            max_value=limit,
            value=min(int(config.default_rows), limit),
            step=1,
        )
    with col_seed:
        seed = st.number_input("Seed", min_value=0, value=int(config.default_seed), step=1)
    if int(rows) >= int(limit * 0.8):
        st.warning(
            "Gerações maiores podem consumir mais memória e levar mais tempo. "
            "O limite apresentado é operacional para esta interface e não representa a capacidade máxima absoluta do modelo."
        )
    st.caption(
        "A seed ajuda a reproduzir a geração. Modelos neurais podem apresentar pequenas variações "
        "conforme backend, hardware e versões das bibliotecas."
    )

    _render_step(st, 3, "Colunas")
    selection_mode = st.radio("Modo de seleção", ["Preset", "Personalizado"], horizontal=True)
    selected_columns: list[str] | None = None
    preset: str | None = None
    if selection_mode == "Preset":
        presets = list(COLUMN_PRESETS)
        default_preset_index = presets.index(config.default_preset) if config.default_preset in presets else 0
        preset = st.selectbox("Preset", presets, index=default_preset_index, format_func=_preset_label)
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

    _render_step(st, 4, "Formato")
    output_format = st.selectbox(
        "Formato",
        ["csv", "json", "parquet"],
        index=["csv", "json", "parquet"].index(config.default_format),
        format_func=lambda value: {
            "csv": "CSV — compatível com planilhas e ferramentas de análise",
            "json": "JSON — adequado para integrações e desenvolvimento",
            "parquet": "Parquet — indicado para análise de dados com preservação de tipos",
        }[value],
    )

    exported_count = len(selected_columns) if selected_columns is not None else len(COLUMN_PRESETS[preset or "completo"])
    _render_step(st, 5, "Revisão")
    st.json(
        {
            "model": model,
            "artifact": selected_artifact_id or "geração programática direta",
            "rows": int(rows),
            "format": output_format,
            "seed": int(seed),
            "column_selection": "explicit" if selected_columns is not None else "preset",
            "preset": preset,
            "exported_columns": selected_columns if selected_columns is not None else list(COLUMN_PRESETS[preset or "completo"]),
            "exported_column_count": exported_count,
        }
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
                session_id=session_id,
            )
            with st.spinner("Gerando dados sintéticos. O tempo depende do modelo e da quantidade solicitada..."):
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
        _render_generation_result(st, st.session_state["last_generation"], config, session_id)


def _render_selected_model(st: Any, entry: ModelCatalogEntry) -> None:
    st.markdown(f"**{entry.label}**")
    st.write(entry.detailed_description)
    st.markdown(
        " ".join(
            [
                badge_html(entry.status, "success" if entry.recommended else "warning" if entry.experimental else "info"),
                badge_html("Exige artefato" if entry.requires_saved_artifact else "Sem treinamento", "muted"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption("Indicado para: " + _join_pt(entry.recommended_use_cases))


def _render_artifact_details(st: Any, artifact: Any) -> None:
    status = artifact_status_label(artifact)
    tone = {
        "Aprovado": "success",
        "Candidato recomendado": "candidate",
        "Candidato": "candidate",
        "Experimental": "warning",
        "Smoke": "smoke",
        "Legado": "legacy",
    }.get(status, "muted")
    st.markdown(badge_html(status, tone), unsafe_allow_html=True)
    manifest = artifact.manifest if isinstance(getattr(artifact, "manifest", None), dict) else {}
    library_versions = manifest.get("library_versions") or manifest.get("environment", {}).get("library_versions", {})
    ctgan_version = library_versions.get("ctgan") or manifest.get("ctgan_version")
    details = [
        f"Artefato: `{artifact.artifact_id}`",
        f"schema {artifact.schema_version}",
        f"vocabulário {artifact.categorical_vocabulary_version}",
        f"renda {artifact.income_model_version}",
        f"geografia {artifact.geography_model_version}",
        f"{artifact.data_locale or 'localidade não registrada'}",
        f"{artifact.unicode_normalization or 'normalização não registrada'}",
        f"seed {artifact.seed}" if artifact.seed is not None else "seed não registrada",
        f"treino {artifact.train_rows}" if artifact.train_rows is not None else "treino não registrado",
    ]
    if artifact.model == "ctgan" and ctgan_version:
        details.append(f"CTGAN {ctgan_version}")
    st.caption(" · ".join(details))
    if artifact.model == "ctgan" and artifact.approval_status == "approved" and artifact.recommended_for_neural_generation:
        st.info(
            "Este é o artefato neural recomendado do projeto. "
            "O modelo programático continua sendo a opção inicial geral para geração rápida e controlada."
        )
    warning = artifact_status_warning(artifact)
    if warning:
        st.warning(warning)


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


def _render_generation_result(st: Any, result: Any, config: UIConfig, session_id: str) -> None:
    manifest = result.manifest
    st.subheader("Resultado da Geração")
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
    st.dataframe(result.dataset.head(int(config.preview_rows)), use_container_width=True)

    st.markdown("**Colunas Exportadas**")
    st.write(", ".join(manifest.get("exported_columns", [])))

    st.markdown("**Validação Estrutural**")
    for label, passed in validation_summary(manifest.get("validation", {})):
        st.write(f"{'✓' if passed else '✗'} {label}")

    dataset_bytes = result.output_path.read_bytes()
    manifest_bytes = result.manifest_path.read_bytes()
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        clicked = st.download_button(
            "Baixar dataset",
            data=dataset_bytes,
            file_name=result.download_filename,
            mime=_mime_for_format(str(manifest.get("format"))),
        )
        if clicked:
            write_audit_event(
                config.audit_events_path,
                "dataset_download_requested",
                {"model": result.service_result.model, "rows": result.service_result.num_rows, "format": manifest.get("format")},
                session_id=session_id,
            )
    with col2:
        clicked = st.download_button(
            "Baixar manifesto",
            data=manifest_bytes,
            file_name=result.manifest_download_filename,
            mime="application/json",
        )
        if clicked:
            write_audit_event(
                config.audit_events_path,
                "manifest_download_requested",
                {"model": result.service_result.model, "rows": result.service_result.num_rows, "format": "json"},
                session_id=session_id,
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


def _preset_label(preset: str) -> str:
    labels = {
        "completo": "completo — todas as 18 colunas",
        "demografico": "demográfico — perfil populacional e socioeconômico",
        "contato": "contato — localização e telefone sintético",
        "documentos": "documentos — identificadores sintéticos",
        "minimo": "mínimo — Nome, Idade, Estado e CPF",
    }
    return labels.get(preset, preset)


def _render_step(st: Any, number: int, title: str) -> None:
    st.markdown(
        f"<div class='sbp-step'><span class='sbp-step-number'>{number}</span>{title}</div>",
        unsafe_allow_html=True,
    )


def _join_pt(values: tuple[str, ...]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" e {values[-1]}"
