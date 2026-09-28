"""Compatibilidade temporária para serviços de auditoria da UI."""

from synthetic_br_profiles_gan.services.audit_service import AuditWriteResult, read_audit_events, sanitize_event, write_audit_event

__all__ = ["AuditWriteResult", "read_audit_events", "sanitize_event", "write_audit_event"]
