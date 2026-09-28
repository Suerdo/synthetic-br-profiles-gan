# API HTTP para a interface React

A API FastAPI é uma camada fina sobre os serviços existentes do projeto. Ela não implementa regras próprias de geração, validação, presets, carregamento de modelos ou exportação.

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

### `GET /api/models/{model}/artifacts`

Lista artefatos tecnicamente válidos de `ctgan`, `simple_gan` ou `programmatic`. A resposta usa `artifact_id`; não expõe `artifact_path`.

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
