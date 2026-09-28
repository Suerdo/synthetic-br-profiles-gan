# Frontend React

Esta fase adiciona uma interface React + TypeScript + Vite sem remover a interface Streamlit. O React reutiliza a API FastAPI, que por sua vez chama os serviços existentes do pacote.

## Instalação

Instale a API opcional:

```bash
pip install -e ".[api]"
```

Instale as dependências do frontend:

```bash
cd frontend
npm install
```

Configure a URL da API com base em `.env.example`:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Execução local

Em um terminal, inicie a API:

```bash
python -m uvicorn synthetic_br_profiles_gan.api.app:app --host 127.0.0.1 --port 8000
```

Em outro terminal:

```bash
cd frontend
npm run dev
```

Abra a URL exibida pelo Vite, normalmente `http://localhost:5173`.

## Estrutura

```text
frontend/
  src/
    api/
    components/
    hooks/
    pages/
    types/
```

A lógica de geração, validação, presets, dependências internas e exportação não fica no React. A interface apenas monta a solicitação, chama a API, acompanha o job assíncrono e apresenta prévia, validação e downloads.

Os serviços reutilizáveis de histórico, governança e auditoria ficam em `synthetic_br_profiles_gan.services`. A camada Streamlit mantém wrappers compatíveis em `synthetic_br_profiles_gan.ui.services`, enquanto a API FastAPI usa os serviços compartilhados diretamente e não depende de Streamlit.

## Páginas

A navegação possui três páginas:

- `Gerar dados`: página inicial, com formulário completo de geração;
- `Modelos`: visão didática dos três sintetizadores e artefatos disponíveis;
- `Governança`: resumo de rastreabilidade e explicação de indicadores.

`Visão geral` e `Conformidade regulatória` não fazem parte da navegação React.

A página `Modelos` consome `/api/models`, `/api/models/{model}` e `/api/models/{model}/recommended`. Ela apresenta os três sintetizadores, seus papéis, resumos simples e técnicos, usos recomendados, limitações, notas de governança e artefatos válidos. Artefatos ausentes ou métricas não calculadas aparecem como `Não avaliado`.

A página `Governança` consome `/api/governance` e os endpoints seccionados de governança. Ela apresenta `Resumo Operacional`, `Modelo Neural Recomendado`, `Qualidade dos Dados`, `Diversidade e Memorização`, `Realismo Condicional`, `Execuções Recentes`, `Auditoria` e glossário. Os filtros atuam sobre os metadados agregados retornados pela API; o frontend não lê datasets completos nem arquivos do diretório `artifacts/` diretamente.

## Sessão efêmera

O React cria um UUID em memória e o envia em `X-UI-Session-ID` nos endpoints de geração. Esse identificador não é persistido em storage ou cookie. Ao recarregar a página, a sessão muda.

## Seleção de modelos

O modelo programático gera diretamente. CTGAN e GAN simples só usam artefatos válidos listados pela API. O usuário não informa caminhos locais nem faz upload de `.pkl`, `.keras` ou `.json`.

Quando existir um artefato CTGAN `approved` com `recommended_for_neural_generation = true`, ele é apresentado como artefato neural recomendado. O programático continua sendo o padrão geral da plataforma.

## Seleção de colunas e formatos

A página `Gerar dados` consome o catálogo retornado por `/api/columns`. A seleção pode usar preset ou lista personalizada. O serviço continua gerando e validando internamente o schema completo antes de exportar o subconjunto solicitado.

Formatos suportados:

- CSV;
- JSON;
- Parquet.

## Downloads

Os downloads são feitos por `generation_id` e sessão. A interface oferece:

- baixar dataset;
- baixar manifesto.

Os nomes de arquivos são descritivos e não expõem caminhos internos.

## Limitações desta fase

- Não há histórico persistente;
- o estado dos jobs fica em memória no servidor;
- a governança React depende dos manifestos e artefatos agregados já existentes; execuções antigas sem métricas novas aparecem como `Não avaliado`;
- não há autenticação;
- não há banco de dados;
- não há upload de modelos;
- não há treinamento pela interface.

A interface Streamlit permanece disponível e não foi removida.
