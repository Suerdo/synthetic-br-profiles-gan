"""Tema visual e utilitários de acessibilidade da interface Streamlit."""

from __future__ import annotations


PALETTE: dict[str, str] = {
    "primary": "#2563EB",
    "primary_dark": "#1E3A8A",
    "background": "#FFFFFF",
    "app_background": "#F1F5F9",
    "surface": "#F8FAFC",
    "surface_strong": "#E2E8F0",
    "section_blue": "#EFF6FF",
    "section_teal": "#F0FDFA",
    "text": "#0F172A",
    "body": "#334155",
    "muted": "#64748B",
    "success": "#047857",
    "warning": "#B45309",
    "danger": "#B91C1C",
    "info": "#0369A1",
    "border": "#CBD5E1",
    "border_strong": "#94A3B8",
    "sidebar": "#0F172A",
    "sidebar_active": "#1E3A8A",
    "sidebar_hover": "#1E293B",
    "sidebar_text": "#F8FAFC",
    "sidebar_muted": "#CBD5E1",
    "sidebar_indicator": "#60A5FA",
    "alert_warning_bg": "#FFF7ED",
    "alert_warning_border": "#FDBA74",
    "alert_warning_text": "#9A3412",
}


STATUS_TONES: dict[str, str] = {
    "approved": "success",
    "completed": "success",
    "quality_quarantined": "warning",
    "quarantined": "warning",
    "quality_rejected": "danger",
    "rejected": "danger",
    "failed": "danger",
    "resource_limited": "danger",
    "backend_unavailable": "warning",
    "skipped_after_failure": "warning",
    "not_available": "muted",
    "unknown": "muted",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Converte uma cor hexadecimal em tupla RGB."""
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Cor hexadecimal inválida: {value}")
    return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)


def relative_luminance(value: str) -> float:
    """Calcula a luminância relativa conforme WCAG."""
    channels = []
    for channel in hex_to_rgb(value):
        component = channel / 255
        channels.append(component / 12.92 if component <= 0.03928 else ((component + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """Retorna a razão de contraste entre duas cores."""
    fg_luminance = relative_luminance(foreground)
    bg_luminance = relative_luminance(background)
    lighter = max(fg_luminance, bg_luminance)
    darker = min(fg_luminance, bg_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def assert_palette_contrast(minimum_ratio: float = 4.5) -> None:
    """Valida contrastes principais do tema para texto normal."""
    pairs = [
        ("text", "background"),
        ("text", "surface"),
        ("primary_dark", "background"),
        ("sidebar_text", "sidebar"),
        ("sidebar_muted", "sidebar"),
        ("alert_warning_text", "alert_warning_bg"),
        ("success", "background"),
        ("warning", "background"),
        ("danger", "background"),
        ("info", "background"),
    ]
    for foreground, background in pairs:
        ratio = contrast_ratio(PALETTE[foreground], PALETTE[background])
        if ratio < minimum_ratio:
            raise AssertionError(f"Contraste insuficiente para {foreground}/{background}: {ratio:.2f}")


def status_tone(status: str | None) -> str:
    """Retorna o tom visual associado a um status técnico ou de qualidade."""
    if not status:
        return "muted"
    return STATUS_TONES.get(str(status).lower(), "muted")


def status_label(status: str | None) -> str:
    """Traduz status internos para rótulos amigáveis sem alterar seus identificadores."""
    labels = {
        "approved": "Aprovado",
        "completed": "Concluído",
        "quality_quarantined": "Concluído com quarentena",
        "quarantined": "Em quarentena",
        "quality_rejected": "Concluído com rejeição de qualidade",
        "rejected": "Rejeitado",
        "failed": "Falha técnica",
        "resource_limited": "Limitação de recursos",
        "backend_unavailable": "Backend indisponível",
        "skipped_after_failure": "Pulado após falha",
        "not_available": "Não disponível",
    }
    if not status:
        return "Não disponível"
    normalized = str(status).lower()
    return labels.get(normalized, str(status))


def badge_html(label: str, tone: str = "muted") -> str:
    """Cria um badge acessível em HTML para uso controlado no Streamlit."""
    colors = {
        "success": ("#DCFCE7", "#166534", "#86EFAC"),
        "warning": ("#FEF3C7", "#92400E", "#FCD34D"),
        "danger": ("#FEE2E2", "#991B1B", "#FCA5A5"),
        "info": ("#DBEAFE", "#1E3A8A", "#93C5FD"),
        "muted": ("#F1F5F9", "#334155", "#CBD5E1"),
        "candidate": ("#E0F2FE", "#075985", "#7DD3FC"),
        "smoke": ("#F5F3FF", "#5B21B6", "#C4B5FD"),
        "legacy": ("#F3F4F6", "#374151", "#D1D5DB"),
    }
    background, color, border = colors.get(tone, colors["muted"])
    return (
        "<span class='sbp-badge' "
        f"style='border-color:{border}; color:{color}; background:{background};'>"
        f"{label}</span>"
    )


def ui_css() -> str:
    """Retorna CSS leve e compatível com modo claro ou escuro."""
    return f"""
<style>
:root {{
  --sbp-primary: {PALETTE["primary"]};
  --sbp-primary-dark: {PALETTE["primary_dark"]};
  --sbp-app-background: {PALETTE["app_background"]};
  --sbp-surface: {PALETTE["surface"]};
  --sbp-surface-strong: {PALETTE["surface_strong"]};
  --sbp-text: {PALETTE["text"]};
  --sbp-body: {PALETTE["body"]};
  --sbp-muted: {PALETTE["muted"]};
  --sbp-border: {PALETTE["border"]};
  --sbp-border-strong: {PALETTE["border_strong"]};
}}
.stApp {{
  background: var(--sbp-app-background);
  color: var(--sbp-body);
}}
.main .block-container {{
  padding-top: 1.65rem;
  padding-bottom: 2.25rem;
  max-width: 1280px;
  margin-left: auto;
  margin-right: auto;
}}
section[data-testid="stSidebar"] {{
  background: {PALETTE["sidebar"]};
}}
section[data-testid="stSidebar"] * {{
  color: {PALETTE["sidebar_text"]};
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] .stCaptionContainer {{
  color: {PALETTE["sidebar_muted"]};
}}
.sbp-sidebar-title {{
  color: {PALETTE["sidebar_text"]};
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: .01em;
  margin: .25rem 0 1.1rem 0;
}}
.sbp-sidebar-item {{
  display: flex;
  align-items: center;
  gap: .55rem;
  border-radius: 10px;
  padding: .72rem .78rem;
  margin: .2rem 0;
  color: {PALETTE["sidebar_text"]};
  border-left: 4px solid transparent;
}}
.sbp-sidebar-active {{
  background: {PALETTE["sidebar_active"]};
  border-left-color: {PALETTE["sidebar_indicator"]};
  font-weight: 750;
}}
section[data-testid="stSidebar"] .stButton>button {{
  width: 100%;
  justify-content: flex-start;
  background: transparent;
  color: {PALETTE["sidebar_text"]};
  border: 1px solid transparent;
  border-radius: 10px;
  padding: .72rem .78rem;
  margin: .08rem 0;
  font-weight: 650;
}}
section[data-testid="stSidebar"] .stButton>button:hover {{
  background: {PALETTE["sidebar_hover"]};
  border-color: rgba(203, 213, 225, .25);
  color: {PALETTE["sidebar_text"]};
}}
.sbp-hero {{
  border: 1px solid var(--sbp-border);
  border-radius: 12px;
  padding: 1.25rem 1.35rem;
  background: #FFFFFF;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
  margin-bottom: 1rem;
}}
.sbp-hero h1 {{
  margin-bottom: .35rem;
}}
.sbp-card {{
  border: 1px solid var(--sbp-border);
  border-radius: 12px;
  padding: 1rem 1.05rem;
  background: #FFFFFF;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
  height: 100%;
}}
.sbp-card:hover {{
  border-color: var(--sbp-border-strong);
}}
.sbp-card h3 {{
  margin-top: 0;
  font-size: 1.05rem;
  color: var(--sbp-text);
}}
.sbp-card p {{
  color: var(--sbp-body);
}}
.sbp-muted {{
  color: var(--sbp-muted);
}}
.sbp-badge {{
  display: inline-block;
  border: 1px solid;
  border-radius: 999px;
  padding: .12rem .48rem;
  font-size: .78rem;
  font-weight: 650;
  line-height: 1.35;
  margin-right: .25rem;
}}
.sbp-step {{
  border: 1px solid #BFDBFE;
  border-left: 7px solid var(--sbp-primary-dark);
  padding: .78rem .95rem;
  background: #EFF6FF;
  border-radius: 10px;
  margin: 1.25rem 0 .8rem 0;
  color: var(--sbp-primary-dark);
  font-weight: 750;
}}
.sbp-step-number {{
  display: inline-flex;
  justify-content: center;
  align-items: center;
  width: 1.65rem;
  height: 1.65rem;
  border-radius: 999px;
  background: var(--sbp-primary-dark);
  color: #FFFFFF;
  margin-right: .55rem;
  font-size: .9rem;
}}
.sbp-flow {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: .6rem;
  margin: .75rem 0 1rem 0;
}}
.sbp-flow div {{
  border: 1px solid var(--sbp-border);
  border-radius: 12px;
  padding: .75rem;
  background: #FFFFFF;
  text-align: center;
  font-weight: 650;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}}
.sbp-governance-alert {{
  border: 1px solid {PALETTE["alert_warning_border"]};
  border-left: 7px solid {PALETTE["alert_warning_border"]};
  background: {PALETTE["alert_warning_bg"]};
  color: {PALETTE["alert_warning_text"]};
  padding: .95rem 1rem;
  border-radius: 12px;
  margin-bottom: .6rem;
}}
.sbp-final-warning {{
  border: 1px solid {PALETTE["alert_warning_border"]};
  background: {PALETTE["alert_warning_bg"]};
  color: {PALETTE["alert_warning_text"]};
  padding: .85rem 1rem;
  border-radius: 12px;
  margin-top: .85rem;
  font-weight: 650;
}}
.sbp-final-warning::before {{
  content: "";
}}
.sbp-governance-alert strong {{
  color: {PALETTE["alert_warning_text"]};
}}
.sbp-section {{
  border: 1px solid var(--sbp-border);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin: 1rem 0;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}}
.sbp-section-blue {{
  background: #EFF6FF;
}}
.sbp-section-neutral {{
  background: #F8FAFC;
}}
.sbp-section-teal {{
  background: #F0FDFA;
}}
.sbp-governance-card-title {{
  margin-bottom: .75rem;
}}
.sbp-governance-card-title h3 {{
  margin: 0 0 .25rem 0;
  color: var(--sbp-text);
  font-size: 1.18rem;
}}
.sbp-governance-card-title p {{
  margin: 0;
  color: var(--sbp-body);
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
  border: 1px solid var(--sbp-border);
  border-radius: 12px;
  background: #FFFFFF;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}}
.sbp-equal-card-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: stretch;
  gap: 1rem;
  margin: .8rem 0 1rem 0;
}}
.sbp-equal-card-grid .sbp-card {{
  background: #F8FAFC;
  min-height: 190px;
}}
.sbp-comparison-table {{
  width: 100%;
  border-collapse: collapse;
  border: 1px solid var(--sbp-border);
  border-radius: 12px;
  overflow: hidden;
  background: #FFFFFF;
}}
.sbp-comparison-table th {{
  background: #E2E8F0;
  color: #0F172A;
  border-bottom: 1px solid var(--sbp-border);
  padding: .65rem .75rem;
  text-align: left;
}}
.sbp-comparison-table td {{
  border-bottom: 1px solid var(--sbp-border);
  color: #334155;
  padding: .65rem .75rem;
}}
.sbp-comparison-table tr:nth-child(even) td {{
  background: #F8FAFC;
}}
.stButton>button:focus, .stDownloadButton>button:focus {{
  outline: 3px solid {PALETTE["primary_dark"]};
  outline-offset: 2px;
}}
@media (max-width: 760px) {{
  .main .block-container {{
    padding-left: .85rem;
    padding-right: .85rem;
  }}
  .sbp-equal-card-grid {{
    grid-template-columns: 1fr;
  }}
  .sbp-equal-card-grid .sbp-card {{
    min-height: auto;
  }}
}}
</style>
"""


INSTITUTIONAL_DESCRIPTION = (
    "A plataforma auxilia atividades de desenvolvimento de software, testes, engenharia de requisitos, "
    "demonstrações técnicas, pesquisa acadêmica, treinamento de modelos de Inteligência Artificial e "
    "validação de pipelines de dados. Embora utilize técnicas de geração de dados sintéticos, não oferece "
    "garantia absoluta de anonimização nem elimina completamente a possibilidade de coincidências estatísticas "
    "com informações reais."
)


GOVERNANCE_WARNING = (
    "Os dados gerados são sintéticos e não foram consultados ou validados em bases oficiais. "
    "A validade estrutural de documentos não comprova existência, regularidade ou associação a uma pessoa real. "
    "Os dados não devem ser utilizados para fraude, autenticação, identificação real ou acesso a serviços."
)


ANONYMIZATION_WARNING = (
    "Dados sintéticos devem permanecer identificados como sintéticos e não substituem avaliação de governança, "
    "privacidade ou uso responsável."
)
