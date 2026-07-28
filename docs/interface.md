# Interface Streamlit

A interface Streamlit é a primeira camada visual do projeto. Ela reutiliza os serviços existentes de geração, catálogo de colunas, presets, validação estrutural, exportação e manifestos. A página não implementa regras próprias de documentos, pós-processamento, validação ou serialização.

## Instalação

A dependência da interface é opcional. Para instalar:

```bash
pip install -e ".[ui]"
```

A instalação básica do pacote continua disponível sem Streamlit:

```bash
pip install -e .
```

## Inicialização

Execute:

```bash
streamlit run app/streamlit_app.py
```

O ponto de entrada fica em `app/streamlit_app.py`. A lógica reutilizável da interface fica em `src/synthetic_br_profiles_gan/ui/`.

## Configuração

A configuração fica em `configs/ui.yaml`.

Ela define:

- título da aplicação;
- quantidade de linhas exibidas na amostra;
- diretório administrado de modelos em `artifacts/models`;
- diretório de sessões em `artifacts/ui_sessions`;
- quantidade padrão de registros;
- limites operacionais por modelo;
- modelo, preset, formato e seed padrão.

Os limites de linhas da interface são limites operacionais de uso interativo. Eles não representam a capacidade máxima absoluta dos modelos e não devem ser confundidos com benchmarks de capacidade.

## Telas

### Gerar dados

A tela principal permite escolher:

- modelo;
- artefato treinado, quando necessário;
- quantidade de registros;
- formato de saída;
- seed;
- preset ou seleção personalizada de colunas.

Ao acionar `Gerar dados sintéticos`, a interface cria um diretório exclusivo, constrói um `GenerationRequest` e chama o `GenerationService`.

### Conheça os modelos

Apresenta os três sintetizadores:

| Modelo | Treinamento | Custo | Situação |
| --- | --- | --- | --- |
| Programático | Não exige | Baixo | Recomendado |
| CTGAN | Exige | Alto | Avançado |
| GAN simples | Exige | Médio | Experimental |

Essas descrições são didáticas e não transformam resultados experimentais em garantias universais.

### Sobre e governança

Explica a finalidade acadêmica, o fluxo de geração, a seleção de colunas, a validade estrutural, a segurança de modelos serializados e as limitações de uso.

## Modelos e artefatos

O modelo `programmatic` está sempre disponível e não exige treinamento prévio.

`ctgan` e `simple_gan` exigem artefatos previamente treinados. A interface lista somente diretórios válidos encontrados dentro de `artifacts/models`, com `training_manifest.json` compatível e arquivos obrigatórios do modelo.

A interface não permite upload de `.pkl`, `.keras`, `.json` ou outros artefatos de modelo. Também não permite que o usuário informe caminhos arbitrários. Modelos serializados devem ser produzidos ou previamente aprovados pela própria aplicação.

Quando um artefato neural foi treinado com a versão anterior do vocabulário categórico, a interface mostra um aviso. A saída terá acentuação normalizada por aliases legados, mas o modelo não conhecerá automaticamente as ocupações adicionadas na versão atual. Para obter toda a diversidade do vocabulário `2`, o artefato precisa ser treinado novamente com a base de calibração atual.

## Seleção de colunas

Os três modelos continuam gerando internamente as 11 colunas-base. O pós-processamento produz as 18 colunas finais e a validação estrutural é executada sobre o schema completo.

A seleção escolhida pelo usuário é aplicada somente depois da validação:

```text
geração interna das 18 colunas
  → validação estrutural completa
  → projeção das colunas solicitadas
  → exportação
```

No modo `Preset`, a interface usa os presets existentes em `column_catalog.py`:

- `completo`;
- `demografico`;
- `contato`;
- `documentos`;
- `minimo`.

No modo personalizado, as colunas são apresentadas por grupo:

- Identificação sintética;
- Demografia;
- Localização e contato;
- Perfil socioeconômico.

Dependências internas, como `Telefone` depender de `Estado` e `DDD`, continuam sendo geradas para preservar a coerência dos perfis. Elas não são adicionadas automaticamente ao arquivo exportado.

## Formatos e downloads

Formatos disponíveis:

- `csv`: compatível com planilhas e ferramentas de análise, gravado em `utf-8-sig` e separador `;`;
- `json`: adequado para integrações e desenvolvimento;
- `parquet`: indicado para análise de dados com preservação de tipos.

Após a geração, a interface apresenta:

- resumo da execução;
- amostra limitada por `preview_rows`;
- colunas exportadas;
- visão amigável da validação estrutural;
- botão para baixar o dataset;
- botão para baixar o manifesto.

## Diretórios temporários

Cada geração usa um diretório exclusivo em:

```text
artifacts/ui_sessions/<session-id>/<generation-id>/
```

Esta primeira versão não implementa histórico persistente nem limpeza automática. A equipe responsável pode remover os diretórios de sessões periodicamente. Os artefatos gerados pela interface não devem ser versionados.

## Governança

Os dados gerados são sintéticos e não foram consultados ou validados em bases oficiais. A validade estrutural de documentos não comprova existência, regularidade ou associação a uma pessoa real.

A ferramenta auxilia testes, ensino e pesquisa, mas não oferece garantia absoluta de anonimização ou ausência de coincidências com informações reais.

Os dados não devem ser usados para fraude, autenticação, identificação real, criação de contas, engenharia social ou acesso a serviços.

## Vocabulário e labels

Os nomes técnicos das colunas continuam sem acento por compatibilidade com modelos, CLI, manifestos, presets e código externo. A interface apresenta labels em português brasileiro, como `Gênero`, `Região`, `Município`, `Ocupação`, `Título de eleitor`, `Estado civil` e `Data de nascimento`.

Os valores categóricos exportados usam português brasileiro canônico, normalização Unicode NFC e vocabulário categórico `2`. Isso inclui escolaridade, estado civil, ocupações e municípios com acentuação. A interface não remove acentos nem converte os dados para ASCII.

## Limitações

- Não há treinamento pela interface nesta fase.
- Não há upload de modelos ou datasets.
- Não há autenticação, banco de dados, histórico persistente, filas ou geração assíncrona.
- A interface não executa benchmarks de capacidade.
- Modelos neurais podem apresentar variações conforme backend, hardware e versões das bibliotecas.
- A referência metodológica continua sendo a base de calibração sintética controlada.

## Preparação para implantação institucional

Antes de uma implantação institucional, recomenda-se definir:

- política de retenção e limpeza dos arquivos em `artifacts/ui_sessions`;
- processo formal de aprovação dos artefatos em `artifacts/models`;
- limites operacionais por ambiente;
- controle de acesso;
- monitoramento local de recursos;
- revisão de governança e uso responsável.
