# Interface Web

A interface oficial do projeto é composta por React + TypeScript + Vite no frontend e FastAPI no backend. A interface é apenas uma camada de apresentação: ela monta solicitações, acompanha jobs assíncronos e exibe resultados, enquanto geração, validação, seleção de colunas, carregamento de modelos, manifestos e governança permanecem nos serviços Python reutilizáveis.

Fluxo arquitetural:

```text
React
  → FastAPI
  → services
  → models / pipeline / evaluation / artifacts
```

## Instalação

Instale o pacote Python com a API:

```bash
pip install -e ".[api]"
```

Instale as dependências do frontend:

```bash
cd frontend
npm install
```

## Execução Local

Use dois processos durante o desenvolvimento local.

Backend:

```bash
python -m uvicorn synthetic_br_profiles_gan.api.app:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Acesse:

```text
http://localhost:5173
```

Para gerar o build estático:

```bash
cd frontend
npm run build
```

## Configuração

A configuração operacional da API fica em `configs/api.yaml`. Ela define título, diretórios administrados, limites interativos por modelo, defaults de geração, raiz de artefatos neurais e caminho de auditoria.

Os limites de linhas são operacionais para uso interativo. Eles não representam capacidade máxima absoluta dos modelos e não substituem benchmarks de capacidade.

## Navegação

A aplicação possui três páginas:

- `Gerar Dados`: formulário principal de geração;
- `Modelos`: apresentação dos três sintetizadores e dos artefatos válidos;
- `Governança`: evidências, quality gates, privacidade, diversidade, realismo condicional, execuções recentes, auditoria e glossário.

## Geração

A página `Gerar Dados` permite escolher modelo, quantidade de registros, seed, formato e colunas exportadas. O modelo `programmatic` gera diretamente. `ctgan` e `simple_gan` usam apenas `artifact_id` de artefatos válidos listados pelo `ModelRegistry`; o usuário não informa `model_path`, `output_path` nem caminhos arbitrários.

A seleção de colunas é aplicada somente depois da geração interna das 18 colunas finais e da validação estrutural completa:

```text
geração interna das 18 colunas
  → validação estrutural completa
  → projeção das colunas solicitadas
  → exportação
```

Formatos suportados:

- `csv`: `utf-8-sig`, sem índice e com separador `;`;
- `json`: lista de objetos UTF-8 com `ensure_ascii=False`;
- `parquet`: preservação de tipos sempre que possível.

Após a conclusão, a interface mostra resumo, preview limitado, validação, colunas exportadas e downloads de dataset e manifesto.

## Sessão Efêmera

O React cria um UUID em memória e o envia em `X-UI-Session-ID`. Esse identificador não é persistido em `localStorage`, `sessionStorage` ou cookie.

Cada geração usa:

```text
artifacts/web_sessions/<ui-session-id>/<generation-id>/
```

Recarregar a página cria uma nova sessão visual. Gerações anteriores continuam nos artefatos locais, mas a interface não recupera downloads de sessões antigas nesta versão.

## Modelos

Papéis operacionais:

- `programmatic`: padrão geral, rápido, controlado e adequado a grandes volumes locais;
- `ctgan`: modelo neural recomendado quando o artefato aprovado está instalado localmente;
- `simple_gan`: baseline acadêmico experimental.

Artefatos neurais aparecem quando o manifesto pode ser lido, o modelo é reconhecido, os arquivos obrigatórios existem, o schema é compatível ou possui tratamento de compatibilidade, e o diretório está dentro da raiz administrada por `artifacts/models`.

`artifacts/` não é distribuído pelo Git. A CTGAN aprovada e a GAN simples dependem de artefatos locais; o programático não depende de artefato neural.

## Segurança

A API não oferece upload de modelos, upload de datasets, treinamento HTTP nem seleção por caminho local. Modelos serializados com `pickle`, Keras ou formatos equivalentes devem ser produzidos ou aprovados pela própria aplicação.

As respostas são sanitizadas: não expõem hostname, usuário local, IP, user agent, caminhos absolutos, detalhes de hardware, variáveis de ambiente, caminhos CUDA, stack traces completos, datasets brutos ou valores individuais gerados.

## Auditoria

Eventos sanitizados são registrados em:

```text
artifacts/web_audit/events.jsonl
```

Eventos não incluem CPF, nomes, telefone, linhas geradas, datasets, IP, user agent, identidade de usuário ou traceback completo. Falhas de auditoria não invalidam uma geração.

## Treinamento e Benchmarking

Treinamento e benchmarking permanecem operações administrativas via CLI. Esta versão não possui endpoint HTTP público para treinamento, benchmark de capacidade, retreinamento neural ou promoção automática de artefatos.

## Histórico da Migração

A interface visual anterior foi removida após validação de paridade funcional. O registro histórico da migração está em `docs/react-streamlit-parity.md`. A documentação operacional atual deve tratar React + FastAPI como a interface oficial.
