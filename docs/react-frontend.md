# Frontend React

O frontend oficial usa React, TypeScript e Vite. Ele consome a API FastAPI e não duplica regras de geração, validação, presets, dependências internas, governança ou exportação.

## Execução

Inicie a API em um terminal:

```bash
python -m uvicorn synthetic_br_profiles_gan.api.app:app --host 127.0.0.1 --port 8000
```

Em outro terminal:

```bash
cd frontend
npm install
npm run dev
```

A URL padrão é `http://localhost:5173`. Para build:

```bash
cd frontend
npm run build
```

## Estrutura

```text
frontend/
  public/
    brand/
  src/
    api/
    components/
    hooks/
    pages/
    types/
    utils/
```

A marca visual fica em `frontend/public/brand/dados-sinteticos-br-logo.png` e também é usada como favicon.

## Páginas

- `Gerar Dados`: geração assíncrona, presets, seleção explícita de colunas, formato, preview e downloads.
- `Modelos`: papéis dos três sintetizadores, artefatos válidos, artefato recomendado e limitações.
- `Governança`: decisão técnica, estratégias, proveniência, quality gates, privacidade, diversidade, realismo condicional, execuções recentes, auditoria e glossário.

## Contrato com a API

O frontend usa:

- `GET /api/health`;
- `GET /api/columns`;
- `GET /api/models`;
- `GET /api/models/{model}`;
- `GET /api/models/{model}/artifacts`;
- `GET /api/models/{model}/recommended`;
- `GET /api/governance` e endpoints seccionados;
- `POST /api/generations`;
- endpoints de status, preview, manifesto e download por `generation_id`.

Todas as gerações enviam `X-UI-Session-ID`, criado em memória pelo React. O frontend não persiste esse identificador e não lê arquivos diretamente de `artifacts/`.

## Modelos e Artefatos

O programático está sempre disponível. CTGAN e GAN simples dependem de artefatos válidos listados pela API. O usuário não informa caminhos locais nem faz upload de `.pkl`, `.keras` ou `.json`.

Quando existir um artefato CTGAN `approved` com `recommended_for_neural_generation = true`, ele é apresentado como artefato neural recomendado. O programático continua sendo o padrão geral da plataforma.

## Valores Ausentes

Valores científicos ausentes permanecem como `null` na API e são exibidos como `Não avaliado`. Zero só é mostrado quando a evidência registra zero real.

## Limitações desta Versão

- não há histórico persistente;
- o estado dos jobs fica em memória no servidor;
- a governança depende dos manifestos e artefatos agregados já existentes;
- execuções antigas sem métricas novas aparecem como `Não avaliado`;
- não há autenticação, banco de dados, upload de modelos, treinamento pela interface ou filas distribuídas.
