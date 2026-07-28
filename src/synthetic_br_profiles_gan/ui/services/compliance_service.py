"""Matriz educacional de conformidade regulatória para a interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synthetic_br_profiles_gan.ui.services.governance_service import GovernanceSnapshot
from synthetic_br_profiles_gan.ui.ui_config import UIConfig


IMPLEMENTADO = "Implementado"
PARCIAL = "Parcialmente implementado"
NAO_EVIDENCIADO = "Não evidenciado"
REQUER_AVALIACAO = "Requer avaliação institucional"
NAO_APLICAVEL = "Não aplicável ao cenário avaliado"

EVIDENCE_STATUSES = (IMPLEMENTADO, PARCIAL, NAO_EVIDENCIADO, REQUER_AVALIACAO, NAO_APLICAVEL)

LEGAL_DISCLAIMER = (
    "Esta página oferece orientação técnica e educacional. Ela não constitui parecer jurídico, "
    "certificação regulatória, auditoria formal ou garantia de conformidade."
)


@dataclass(frozen=True)
class ComplianceReference:
    """Referência oficial exibida na página de conformidade."""

    label: str
    url: str
    source: str


def build_compliance_matrix(config: UIConfig, snapshot: GovernanceSnapshot) -> list[dict[str, Any]]:
    """Cria uma matriz de evidências sem calcular uma nota geral de conformidade."""
    has_history = bool(snapshot.history)
    has_audit = bool(snapshot.audit_events)
    latest_validation = _latest_validation_status(snapshot)
    return [
        {
            "tema": "LGPD: finalidade e minimização",
            "status": PARCIAL if has_history else NAO_EVIDENCIADO,
            "apoia": "Uso de dados sintéticos para reduzir exposição operacional em testes e desenvolvimento.",
            "evidência": "Manifestos de geração e treinamento." if has_history else "Nenhum manifesto local encontrado.",
            "requer_processo_institucional": "Definição de finalidade, base legal, papéis e retenção.",
            "nao_avaliado": "Conformidade jurídica do caso de uso específico.",
        },
        {
            "tema": "LGPD: segurança e prevenção",
            "status": PARCIAL if has_audit else NAO_EVIDENCIADO,
            "apoia": "Auditoria sanitizada de eventos operacionais sem registrar valores gerados.",
            "evidência": "events.jsonl sanitizado." if has_audit else "Sem eventos de auditoria registrados.",
            "requer_processo_institucional": "Controles de acesso, segregação de ambientes e revisão periódica.",
            "nao_avaliado": "Maturidade completa de segurança institucional.",
        },
        {
            "tema": "Validade estrutural sem consulta oficial",
            "status": IMPLEMENTADO if latest_validation else NAO_EVIDENCIADO,
            "apoia": "Validação local de schema, documentos e coerência sem consultar bases reais.",
            "evidência": "validation.is_valid em manifesto recente." if latest_validation else "Sem validação recente evidenciada.",
            "requer_processo_institucional": "Bloqueio de uso em serviços reais, fraude, autenticação ou identificação.",
            "nao_avaliado": "Existência, regularidade ou associação real de identificadores.",
        },
        {
            "tema": "ECA Digital: proteção de crianças e adolescentes",
            "status": REQUER_AVALIACAO,
            "apoia": "Avisos de uso responsável e ausência de finalidade de identificação real.",
            "evidência": "Conteúdo de governança da interface.",
            "requer_processo_institucional": "Avaliação institucional quando a ferramenta for usada em contextos com crianças ou adolescentes.",
            "nao_avaliado": "Conformidade material com todos os deveres da Lei nº 15.211/2025 e do Decreto nº 12.880/2026.",
        },
        {
            "tema": "Anonimização",
            "status": REQUER_AVALIACAO,
            "apoia": "Métricas de privacidade e diversidade como indicadores de risco.",
            "evidência": "Relatórios experimentais quando disponíveis.",
            "requer_processo_institucional": "Avaliação técnica e jurídica específica para cada base e finalidade.",
            "nao_avaliado": "Garantia absoluta de anonimização.",
        },
    ]


def legal_references() -> tuple[ComplianceReference, ...]:
    """Retorna referências oficiais usadas na documentação da interface."""
    return (
        ComplianceReference(
            label="Lei Geral de Proteção de Dados Pessoais, Lei nº 13.709/2018",
            url="https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm",
            source="Planalto",
        ),
        ComplianceReference(
            label="Lei nº 15.211/2025, conhecida como ECA Digital",
            url="https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2025/lei/l15211.htm",
            source="Planalto",
        ),
        ComplianceReference(
            label="Decreto nº 12.880/2026, regulamentação da Lei nº 15.211/2025",
            url="https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2026/decreto/d12880.htm",
            source="Planalto",
        ),
        ComplianceReference(
            label="Materiais educativos e publicações da ANPD",
            url="https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes",
            source="ANPD",
        ),
        ComplianceReference(
            label="Documentos técnicos e orientativos da ANPD",
            url="https://www.gov.br/anpd/pt-br/centrais-de-conteudo/documentos-tecnicos-orientativos",
            source="ANPD",
        ),
    )


def compliance_summary(config: UIConfig, snapshot: GovernanceSnapshot) -> dict[str, Any]:
    """Resume a matriz sem produzir nota, ranking ou percentual de conformidade."""
    matrix = build_compliance_matrix(config, snapshot)
    counts = {status: 0 for status in EVIDENCE_STATUSES}
    for row in matrix:
        counts[str(row["status"])] += 1
    return {
        "legal_content_last_review": config.legal_content_last_review,
        "statuses": counts,
        "matrix_rows": len(matrix),
        "disclaimer": LEGAL_DISCLAIMER,
    }


def _latest_validation_status(snapshot: GovernanceSnapshot) -> bool:
    for record in snapshot.history:
        validation = record.manifest.get("validation") if isinstance(record.manifest, dict) else None
        if isinstance(validation, dict) and validation.get("is_valid") is not None:
            return bool(validation.get("is_valid"))
    return False
