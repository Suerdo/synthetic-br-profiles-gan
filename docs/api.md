# API HTTP para a interface React

A API FastAPI é uma camada fina sobre os serviços existentes do projeto. Ela não implementa regras próprias de geração, validação, presets, carregamento de modelos ou exportação.

A configuração operacional padrão fica em `configs/api.yaml`.

## Instalação e execução

Instale o extra opcional:

```bash
pip install -e ".[api]"
```

Inicie a API a partir da raiz do repositório:

```bash
python -m uvicorn synthetic_br_profiles_gan.api.app:app --host 127.0.0.1 --port 8000
```

O CORS é restrito a origens explícitas. Por padrão, a API aceita `http://localhost:5173` e `http://127.0.0.1:5173`. Para alterar:

```bash
SYNTHETIC_BR_PROFILES_GAN_CORS_ORIGINS=http://localhost:5173
```

Não use `*` em produção ou homologação.

## Sessão efêmera

Todos os endpoints de geração exigem:

```text
X-UI-Session-ID: <uuid>
```

A interface React cria esse UUID apenas em memória. Ele não é salvo em `localStorage`, `sessionStorage` ou cookie. Ao recarregar a página, uma nova sessão é criada e gerações anteriores da sessão antiga não são recuperadas pela interface.

## Endpoints

### `GET /api/health`

Retorna saúde operacional, versão da API, defaults da interface e disponibilidade resumida dos modelos. A resposta não expõe caminhos locais.

### `GET /api/columns`

Retorna o catálogo das 18 colunas finais, grupos, dependências internas e presets. A interface usa esse endpoint para evitar listas duplicadas em TypeScript.

### `GET /api/models`

Retorna os três modelos, suas descrições, disponibilidade, limites operacionais da interface e o artefato recomendado quando aplicável.

### `GET /api/models/{model}`

Retorna a ficha detalhada de um modelo, incluindo resumos simples e técnicos, usos recomendados, limitações, notas de governança e artefatos válidos administrados pela aplicação. A resposta não expõe caminhos absolutos nem permite escolher arquivos fora da raiz configurada.

### `GET /api/models/{model}/artifacts`

Lista artefatos tecnicamente válidos de `ctgan`, `simple_gan` ou `programmatic`. A resposta usa `artifact_id`; não expõe `artifact_path`.

### `GET /api/models/{model}/recommended`

Retorna o artefato recomendado segundo a política do `ModelRegistry`. Para `ctgan`, a prioridade é: artefato `approved` com `recommended_for_neural_generation = true`, outro aprovado, `recommended_candidate`, `candidate` e, somente por seleção manual, artefatos `experimental`, `smoke` ou legados. O endpoint não confunde recência com recomendação.

### Governança

```text
GET /api/governance
GET /api/governance/summary
GET /api/governance/quality
GET /api/governance/privacy
GET /api/governance/income
GET /api/governance/executions
GET /api/governance/audit
```

Esses endpoints entregam o mesmo conjunto de evidências usado pela interface React em `Governança`: resumo operacional, qualidade dos dados, privacidade, diversidade e memorização, realismo condicional, execuções recentes e auditoria sanitizada.

Campos ausentes permanecem como `null` na API para que o frontend exiba `Não avaliado` sem transformar ausência em zero. As respostas de governança são sanitizadas: não expõem hostname, usuário local, caminhos absolutos, IP, user agent, variáveis de ambiente, caminhos CUDA, stack traces ou linhas individuais geradas.

#### Campos semânticos de governança

`GET /api/governance` também inclui blocos semânticos para a interface React:

- `governance_decision`: decisão técnica interna, escopo, benchmark, contagem de gates obrigatórios, ressalvas e separação entre `evaluation_status`, `recommendation_status`, `general_default` e `production_status`;
- `available_strategies`: as três estratégias do projeto, separando estratégia disponível de artefato disponível;
- `provenance`: itens de proveniência, grupos e linha do tempo sanitizada;
- `quality_gates`: linhas normalizadas com métrica, valor observado, operador, threshold, obrigatoriedade, resultado e fonte.

O frontend deve tratar esses campos como fonte de verdade para a página `Governança`. Valores científicos, thresholds e contagens não devem ser hardcoded no React.

### `POST /api/generations`

Agenda uma geração assíncrona e retorna `202 Accepted`.

Exemplo:

```json
{
  "model": "programmatic",
  "num_rows": 100,
  "output_format": "csv",
  "seed": 41,
  "column_preset": "minimo"
}
```

Para modelos neurais, informe somente `artifact_id` previamente retornado pelo registry:

```json
{
  "model": "ctgan",
  "artifact_id": "ctgan/20260730T123208Z-income-v3-geo-v2-approved",
  "num_rows": 100,
  "output_format": "csv",
  "seed": 41,
  "selected_columns": ["Nome", "Idade", "Estado", "CPF"]
}
```

O contrato HTTP não aceita `model_path` nem `output_path`.

### `GET /api/generations/{generation_id}`

Consulta o status da geração da sessão atual. Jobs de outra sessão retornam `404`.

Quando um job falha, a resposta pública inclui apenas tipo de erro e mensagem amigável. Tracebacks e detalhes internos ficam restritos aos logs do servidor.

### `GET /api/generations/{generation_id}/preview`

Retorna uma amostra limitada do dataset exportado.

### `GET /api/generations/{generation_id}/manifest`

Retorna manifesto sanitizado, sem caminhos locais, ambiente completo ou commit.

### Downloads

```text
GET /api/generations/{generation_id}/download/dataset
GET /api/generations/{generation_id}/download/manifest
```

Os downloads exigem o mesmo `X-UI-Session-ID`.

## Isolamento de arquivos

Cada geração usa um diretório exclusivo:

```text
artifacts/web_sessions/<ui-session-id>/<generation-id>/
```

O cliente não informa caminhos de saída. O servidor gera nomes descritivos e preserva dataset, manifesto e prévia da geração. Esta primeira fase usa armazenamento local e estado de jobs em memória; em reinício do servidor, o histórico em memória é perdido.

## Segurança de modelos

A API nunca aceita upload ou caminho arbitrário de modelo. Artefatos neurais são selecionados por `artifact_id` a partir de `artifacts/models`, validado pelo `ModelRegistry`.

Modelos serializados com `pickle`, Keras ou formatos equivalentes devem ser produzidos ou previamente aprovados pela aplicação.
