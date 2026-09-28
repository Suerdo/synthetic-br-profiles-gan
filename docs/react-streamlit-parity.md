# Paridade entre React e Streamlit

Este documento registra a situação de paridade entre a interface React + FastAPI e a interface Streamlit. A existência do frontend React não remove o Streamlit nesta fase.

## Princípio

As duas interfaces devem reutilizar os mesmos serviços do pacote:

- `GenerationService`;
- `ModelRegistry`;
- catálogo de colunas e presets;
- validação estrutural;
- exportação e manifestos;
- serviços compartilhados de histórico, governança e auditoria.

Nenhuma interface deve duplicar regras de geração, validação, vocabulário, renda, geografia, quality gates ou seleção de artefatos.

## Cobertura atual do React

O React cobre:

- geração programática direta;
- seleção de artefato neural por `artifact_id`;
- seleção de colunas por preset ou lista explícita;
- exportação em CSV, JSON e Parquet;
- prévia e downloads por sessão efêmera;
- página `Modelos` com os três sintetizadores;
- página `Governança` com decisão técnica, estratégias, modelo neural recomendado, proveniência, quality gates, privacidade, diversidade, memorização, realismo estatístico, execuções recentes, auditoria recolhida e glossário recolhido.

A cadeia conceitual da Governança é:

```text
configuração → geração → validação → avaliação → evidências → decisão → artefato → rastreabilidade
```

## Separação de estados

O React apresenta separadamente:

- `evaluation_status`: resultado da avaliação técnica;
- `recommendation_status`: recomendação neural;
- `general_default`: padrão geral da plataforma;
- `production_status`: decisão de produção, quando existir.

Essa separação evita confundir a CTGAN aprovada internamente com o padrão geral da plataforma. O `ProgrammaticSynthesizer` permanece como padrão geral; a CTGAN aprovada é o artefato neural recomendado; a GAN simples permanece como baseline experimental.

## Sanitização

A API usada pelo React não expõe:

- hostname;
- usuário local;
- caminhos absolutos;
- IP;
- user agent;
- hardware;
- variáveis de ambiente;
- caminhos CUDA;
- stack traces;
- datasets completos;
- valores individuais gerados.

Campos ausentes permanecem como `null`, e a interface exibe `Não avaliado` sem converter ausência em zero.

## Streamlit preservado

O Streamlit continua disponível para uso local e comparação. Seus wrappers em `synthetic_br_profiles_gan.ui.services` apontam para os serviços compartilhados, mantendo compatibilidade sem duplicar regra de negócio.

## Antes de remover o Streamlit

Antes de qualquer remoção futura, ainda é necessário validar:

- smoke manual do React em larguras de 375 px, 768 px, 1024 px e 1440 px;
- estabilidade da API com múltiplas sessões;
- política institucional de retenção de arquivos em `artifacts/web_sessions`;
- estratégia de distribuição dos artefatos neurais aprovados;
- autenticação e autorização, caso a aplicação seja exposta fora de ambiente local;
- observabilidade operacional fora do armazenamento em memória;
- revisão institucional dos textos de governança.
