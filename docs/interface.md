# Interface Streamlit

A interface Streamlit é a camada visual da plataforma. Ela reutiliza `GenerationService`, `ModelRegistry`, catálogo de colunas, presets, validação estrutural, exportação e manifestos. A aplicação não implementa regras próprias de geração de CPF, documentos, pós-processamento, validação ou serialização.

## Instalação

A dependência da interface é opcional:

```bash
pip install -e ".[ui]"
```

A instalação básica continua disponível sem Streamlit:

```bash
pip install -e .
```

## Inicialização

Execute:

```bash
streamlit run app/streamlit_app.py
```

O ponto de entrada fica em `app/streamlit_app.py`. Os componentes reutilizáveis ficam em `src/synthetic_br_profiles_gan/ui/`.

## Configuração

A configuração fica em `configs/ui.yaml`. Ela define título, subtítulo, diretórios administrados, limites operacionais da interface, seed padrão, formato padrão, artefatos neurais aprovados, caminho da auditoria e data de revisão do conteúdo regulatório.

Os limites de linhas são operacionais para uso interativo. Eles não representam a capacidade máxima absoluta dos modelos e não devem ser confundidos com benchmarks de capacidade.

## Navegação

A aplicação possui três áreas:

- `Gerar dados`: formulário principal de geração.
- `Modelos`: explicação didática e técnica dos três sintetizadores.
- `Governança`: histórico, indicadores, quality gates, auditoria sanitizada e glossário de interpretação.

`Gerar dados` é a página inicial. A navegação usa menu lateral com fundo azul-marinho, item ativo destacado e a marca visual `Dados Sintéticos Brasileiro`. A sidebar termina após os itens de navegação, sem rodapé informativo.

Títulos e subtítulos estruturais usam capitalização com iniciais maiúsculas nas palavras principais, como `Resumo Operacional`, `Qualidade dos Dados`, `Execuções Recentes`, `Resumo Simples` e `Resumo Técnico`. Labels funcionais de formulário podem permanecer em frase natural, como `Quantidade de registros`.

## Geração

A tela `Gerar dados` organiza o fluxo em seis etapas:

1. Escolha do modelo.
2. Volume e Reprodutibilidade.
3. Seleção de colunas.
4. Formato de saída.
5. Revisão antes da execução.
6. Execução e downloads.

Ao clicar em `Gerar dados sintéticos`, a interface cria um diretório exclusivo, constrói um `UIGenerationRequest` e chama o serviço de geração. O Streamlit não duplica lógica de negócio.

## Modelos e artefatos

O modelo `programmatic` fica sempre disponível e não exige treinamento.

`ctgan` e `simple_gan` ficam disponíveis na tela de geração quando há artefatos tecnicamente válidos no diretório administrado. Um artefato aparece quando o manifesto pode ser lido, o tipo de modelo é reconhecido, os arquivos obrigatórios existem, o schema é compatível e o diretório permanece dentro de `artifacts/models`.

A ausência de status `approved` não bloqueia automaticamente a exibição. Em vez disso, a interface mostra a finalidade do artefato:

- `Aprovado`;
- `Candidato`;
- `Experimental`;
- `Smoke`;
- `Legado`;
- `Sem classificação`.

Ao selecionar `ctgan` ou `simple_gan`, o artefato mais recente do modelo é pré-selecionado por `created_at_utc` do manifesto. Quando não houver data temporal no manifesto, a aplicação usa metadados temporais disponíveis e, por último, uma data segura do arquivo. O badge `Mais recente` indica apenas recência, não melhor qualidade nem aprovação.

A interface não permite upload de `.pkl`, `.keras`, `.json` ou outros artefatos de modelo. Também não permite que o usuário informe caminhos arbitrários. Modelos serializados devem ser produzidos ou previamente aprovados pela própria aplicação.

Avisos por finalidade:

- `Smoke`: treinado apenas para validação técnica e não representa modelo de produção.
- `Experimental`: finalidade experimental; métricas devem ser avaliadas antes de uso crítico.
- `Legado`: treinado com versão anterior do vocabulário; a saída será normalizada, mas pode apresentar menor diversidade de ocupações.
- `Candidato`: artefato em avaliação, ainda não definido como modelo neural padrão.

## Seleção de colunas

Os três modelos continuam gerando internamente as 11 colunas-base. O pós-processamento produz as 18 colunas finais e a validação estrutural é executada sobre o schema completo.

A seleção escolhida pelo usuário é aplicada somente depois da validação:

```text
geração interna das 18 colunas
  → validação estrutural completa
  → projeção das colunas solicitadas
  → exportação
```

No modo `Preset`, a interface usa os presets existentes em `column_catalog.py`: `completo`, `demografico`, `contato`, `documentos` e `minimo`.

No modo personalizado, as colunas são agrupadas em `Identificação sintética`, `Demografia`, `Localização e contato` e `Perfil socioeconômico`. Dependências internas continuam sendo geradas para preservar a coerência dos perfis, mas não são adicionadas automaticamente ao arquivo exportado.

## Formatos e downloads

Formatos disponíveis:

- `csv`: gravado em `utf-8-sig`, sem índice e com separador `;`;
- `json`: lista de objetos em UTF-8 com `ensure_ascii=False`;
- `parquet`: preserva tipos sempre que possível.

Após a geração, a interface apresenta resumo, amostra limitada por `preview_rows`, colunas exportadas, validação estrutural e botões para baixar o dataset e o manifesto.

Os botões de download são apresentados lado a lado, com largura visual semelhante e próximos entre si para reduzir deslocamento visual.

## Campos do formulário

Os campos de configuração da geração usam primeiro o tema oficial do Streamlit, definido em `.streamlit/config.toml`. A opção `secondaryBackgroundColor = "#E8EEF7"` dá aos widgets um fundo azul-acinzentado claro, `borderColor = "#64748B"` define a borda visível e `showWidgetBorder = true` mantém a borda mesmo sem foco. A cor `primaryColor = "#1E3A8A"` orienta o foco e elementos selecionados.

Os `number_input`, como `Quantidade de registros` e `Seed`, devem aparecer como caixas delimitadas, com steppers visíveis. Os `selectbox`, como `Preset`, `Formato`, `Tipo`, `Modelo` e `Status`, usam fundo e borda do tema, mantendo seta de dropdown contrastada e área clicável confortável.

O CSS interno da interface é usado para layout, sidebar, cards, seções e destaques. Ele não redefine genericamente `input`, `selectbox`, `multiselect` ou elementos internos BaseWeb. Qualquer CSS adicional para widgets deve ser tratado como fallback restrito e documentado.

## Auditoria

A interface registra eventos sanitizados em:

```text
artifacts/ui_audit/events.jsonl
```

Eventos registrados incluem `session_started`, `page_viewed`, `model_selected`, `generation_requested`, `generation_succeeded`, `generation_failed`, `dataset_download_requested` e `manifest_download_requested`.

A auditoria não registra linhas geradas, CPF, nomes, telefones, datasets, traceback completo, IP, user agent ou identidade de usuário. Falhas de escrita da auditoria não invalidam uma geração.

## Diretórios temporários

Cada geração usa um diretório exclusivo em:

```text
artifacts/ui_sessions/<session-id>/<generation-id>/
```

Esta primeira versão não implementa histórico persistente nem limpeza automática. A equipe responsável deve definir uma política de retenção e remoção desses arquivos em ambiente institucional.

## Governança

A página de governança lê evidências reais de manifestos e eventos locais. Quando não houver dado suficiente, exibe `Não disponível`, `Não avaliado` ou `Sem execução registrada`. Cada bloco informa a origem esperada, como `manifesto de execução`, `validation.json`, `quality_gates.json`, `evaluation.json` ou `generation.json`.

A página foi simplificada para concentrar `Resumo Operacional`, `Modelo Neural Recomendado`, `Qualidade dos Dados`, `Privacidade`, `Execuções Recentes`, `Auditoria` e `Como interpretar os indicadores`. As seções principais usam containers com borda, fundo claro, padding consistente e títulos no topo. O histórico visual de artefatos e a matriz regulatória não fazem parte dessa tela. Consulte `docs/governance.md` para o glossário operacional e `docs/compliance.md` para referências regulatórias.

## Artefato neural recomendado

A CTGAN aprovada é selecionada pelo `ModelRegistry` por uma regra explícita: primeiro artefatos `approved` com `recommended_for_neural_generation = true`, depois outros aprovados, candidatos recomendados e candidatos. Artefatos `smoke`, `experimental` e `legacy` continuam visíveis quando tecnicamente válidos, mas não são pré-selecionados como recomendados.

O artefato `artifacts/models/ctgan/20260730T123208Z-income-v3-geo-v2-approved/` é o artefato neural recomendado. Ele usa vocabulário v2, renda v3 e geografia v2. O modelo programático permanece como padrão geral da plataforma; a CTGAN aprovada é recomendada para geração neural avaliada; a GAN simples permanece experimental.

A aprovação é técnica e interna. Ela não é certificação externa, não garante anonimização e não representa validação populacional oficial.

## Limitações

- Não há treinamento pela interface nesta fase.
- Não há upload de modelos ou datasets.
- Não há autenticação, banco de dados, histórico persistente, filas ou geração assíncrona.
- A interface não executa benchmarks de capacidade.
- Modelos neurais podem variar conforme backend, hardware e versões das bibliotecas.
- A referência metodológica continua sendo a base de calibração sintética controlada.
## Indicadores de diversidade e renda

A interface apresenta métricas de duplicidade de combinações-base, correspondência exata com treino, correspondência exata com holdout e realismo condicional da renda na página `Governança`.

Quando uma execução foi criada antes dessas métricas, o valor aparece como `Não avaliado`. Zero é exibido somente quando o artefato registra zero real.

Esses indicadores usam as colunas-base do modelo e excluem identificadores derivados. Eles apoiam avaliação de risco e qualidade, mas não garantem anonimização nem conformidade regulatória.
