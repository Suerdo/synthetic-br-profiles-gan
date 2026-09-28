# Registro Histórico da Migração de Interface

Migração concluída.

Este documento registra que a interface React + FastAPI substituiu a interface visual anterior após validação de paridade suficiente para o congelamento da versão do artigo.

## Resultado

| Funcionalidade | Situação no React + FastAPI |
| --- | --- |
| Gerar Dados | Equivalente |
| Programático | Equivalente |
| CTGAN | Equivalente, com seleção por `artifact_id` administrado |
| GAN simples | Equivalente, com seleção por `artifact_id` administrado |
| Presets | Equivalente |
| Seleção personalizada de colunas | Equivalente |
| Seed e quantidade | Equivalente |
| CSV, JSON e Parquet | Equivalente |
| Preview | Equivalente |
| Validação | Equivalente |
| Download de dataset e manifesto | Equivalente |
| Modelos | Substituída por página React integrada ao `ModelRegistry` |
| Governança | Substituída por página React com evidências por modelo e dados sanitizados da API |
| Execuções recentes | Equivalente |
| Auditoria | Equivalente, via serviços compartilhados e API |
| Estados vazios | Equivalente |
| Erros amigáveis | Equivalente |
| Exposição de caminhos e ambiente local | Deliberadamente removida |
| Estado visual persistente de sessão | Deliberadamente removido |
| Treinamento pela interface | Deliberadamente fora de escopo |

## Arquitetura Final

```text
React
  → FastAPI
  → services
  → models / pipeline / evaluation / artifacts
```

O pacote Python não contém mais a camada visual antiga. Serviços reutilizáveis permanecem em `synthetic_br_profiles_gan.services`, o catálogo de modelos fica em `synthetic_br_profiles_gan.services.model_catalog` e a API usa `configs/api.yaml`.

## Segurança Mantida

A interface oficial não aceita `model_path`, `output_path`, upload de modelos, upload de datasets nem treinamento HTTP. A API usa `artifact_id`, sessão efêmera e respostas sanitizadas.
