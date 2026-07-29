# Governança, LGPD e uso responsável

O projeto é um artefato de pesquisa e experimentação. Ele gera dados sintéticos locais e não consulta Receita Federal, cartórios, operadoras, bases governamentais ou serviços externos.

Princípios adotados:

- finalidade restrita a pesquisa, teste, homologação e demonstração;
- minimização de dados e ausência de bases pessoais reais;
- identificadores fictícios gerados localmente;
- validação estrutural sem consulta oficial;
- avaliação de utilidade e indicadores de risco de memorização;
- rastreabilidade por configuração, run id, hashes e manifestos.
- auditoria sanitizada dos eventos da interface, sem registrar valores individuais gerados.

Dados sintéticos não devem ser automaticamente considerados anonimizados. As métricas de privacidade do pipeline são indicadores e não substituem avaliação jurídica, relatório de impacto, governança institucional ou auditoria.

Quality gates obrigatórios ausentes, inválidos ou `NaN` reprovam a execução quando são necessários para aprovação. Smoke tests pequenos servem para validar o funcionamento técnico do pipeline e são colocados em quarentena quando não atingem o tamanho mínimo configurado.

## Quality gates

| Gate | Métrica | Unidade | Limite padrão | Obrigatório | Interpretação e comportamento quando ausente |
| --- | --- | --- | --- | --- | --- |
| `min_evaluation_rows` | `row_counts.synthetic` | linhas | `100` | em `approval` | Tamanho mínimo para sustentar aprovação estatística; ausente, inválido ou abaixo do limite rejeita em `approval` e coloca em quarentena nos demais modos. |
| `invalid_rows_max` | `validation.invalid_rows` | linhas | `0` | sim | Linhas finais estruturalmente inválidas; ausente, inválido ou acima do limite rejeita. |
| `duplicated_identifier_max` | contagem de motivos `*_duplicado` para identificadores | linhas | `0` | sim | Colisões de identificadores gerados; ausente, inválido ou acima do limite rejeita. |
| `null_required_fields_max` | `validation.reason_counts.null_required_fields` | campos | `0` | sim | Campos obrigatórios nulos; ausente, inválido ou acima do limite rejeita. |
| `exact_train_match_rate_max` | `evaluation.privacy.exact_train_match_rate` | taxa | `0.01` | sim | Match exato com treino sobre atributos de modelo; ausente, inválido ou acima do limite rejeita. |
| `total_variation_distance_max` | maior TVD categórica contra holdout | distância `[0, 1]` | `0.25` | não | Deriva de distribuições categóricas; ausente, inválido ou acima do limite coloca em quarentena quando opcional. |
| `correlation_difference_max` | maior diferença absoluta de correlação contra holdout | diferença absoluta | `0.30` | não | Deriva da matriz de correlação numérica; ausente, inválido ou acima do limite coloca em quarentena quando opcional. |

Uso proibido:

- fraude ou falsificação documental;
- simulação de identidade real;
- criação indevida de contas;
- engenharia social;
- interação com serviços reais;
- tomada de decisão sobre pessoas.

## Governança da interface

A interface Streamlit possui uma página `Governança` dedicada a consolidar evidências locais. Ela lê manifestos, histórico de modelos, relatórios de validação, quality gates e eventos sanitizados. Quando uma informação não existe, a interface usa explicitamente `Não disponível`, `Não avaliado` ou `Sem execução registrada`.

Os indicadores exibidos são evidências operacionais, não certificações. Exemplos:

- status do pipeline com base nos manifestos encontrados;
- artefatos de modelo e sua situação (`approved`, `smoke`, `experimental`, `candidate` ou `legacy`);
- versão do vocabulário categórico;
- validação estrutural mais recente;
- indicadores de duplicidade e privacidade quando disponíveis;
- histórico filtrável por tipo, modelo e status;
- trilha de auditoria sanitizada.

## Origem das métricas

Cada bloco da página informa a fonte esperada:

- resumo operacional: manifesto de execução e `quality_gates.json`;
- qualidade dos dados: `validation.json`, `quality_gates.json` e manifesto de execução;
- privacidade: `evaluation.json` e métricas de privacidade disponíveis;
- execuções recentes: manifestos de execução, benchmark, treinamento e geração da interface;
- auditoria: `artifacts/ui_audit/events.jsonl`.

Ausência de métrica não é exibida como zero. A interface usa `Não avaliado` e informa que a execução não produziu aquela métrica. Zero é reservado para valores reais registrados como zero.

## Glossário dos indicadores

**Execuções registradas:** quantidade de manifestos de execução identificados pela aplicação.

**Aprovadas:** execuções sem falha nos quality gates obrigatórios.

**Em quarentena:** execuções tecnicamente concluídas, mas com alertas ou falhas em métricas informativas.

**Rejeitadas:** execuções que falharam em pelo menos um gate obrigatório.

**Linhas válidas:** linhas que passaram por todas as validações estruturais do schema final.

**Linhas inválidas:** linhas com problemas de domínio, consistência, nulidade, documento ou relacionamento estrutural.

**Identificadores duplicados:** repetições encontradas em CPF, CNH, RG, título de eleitor ou telefone dentro da mesma geração.

**Cobertura de ocupações:** proporção das ocupações canônicas reproduzidas pelo modelo na amostra avaliada.

**Distância de variação total:** diferença entre distribuições categóricas do conjunto sintético e da referência. Quanto menor, mais próximas estão as distribuições comparadas.

**Diferença de correlação:** maior diferença observada nas relações entre variáveis numéricas.

**Exact train match rate:** proporção de registros sintéticos cujas colunas-base coincidem exatamente com registros do treinamento.

**Risco de privacidade:** classificação derivada de métricas explícitas disponíveis. Não constitui garantia de anonimização.

**Status operacional:** situação técnica da execução mais recente identificada.

## Auditoria sanitizada

Eventos da interface são registrados em:

```text
artifacts/ui_audit/events.jsonl
```

Eventos previstos:

- `session_started`;
- `page_viewed`;
- `model_selected`;
- `generation_requested`;
- `generation_succeeded`;
- `generation_failed`;
- `dataset_download_requested`;
- `manifest_download_requested`.

A auditoria registra apenas metadados operacionais, como modelo, quantidade solicitada, formato, seed, modo de seleção de colunas, duração e tipo de erro. Ela não registra CPF, CNH, RG, título de eleitor, telefone, nome, linhas geradas, dataset, traceback completo, IP, user agent ou identidade de usuário.

Falhas de escrita do arquivo de auditoria são registradas em log e não invalidam a geração. Em ambiente institucional, a equipe responsável deve definir retenção, proteção, rotação e revisão desses arquivos.

## Artefatos de modelo

Artefatos neurais aparecem na tela `Gerar dados` quando são tecnicamente válidos: manifesto legível, modelo reconhecido, arquivos obrigatórios presentes, schema compatível e diretório dentro da raiz administrada. A aprovação não é usada como bloqueio automático de exibição; ela aparece como status do artefato.

Artefatos `smoke`, `experimental`, `candidate` ou `legacy` recebem avisos específicos. O artefato mais recente de cada modelo neural é selecionado por padrão na tela de geração, mas recência não significa melhor qualidade, aprovação ou recomendação.

A página `Governança` não exibe mais a seção visual `Modelos e versões`. Essa remoção é apenas visual e não altera `ModelRegistry`, manifestos ou seleção de artefatos.

## Conteúdo regulatório

O conteúdo regulatório detalhado permanece documentado em `docs/compliance.md`. A página `Governança` da interface foi simplificada e não exibe matriz regulatória ou seção separada de conformidade.
## Diversidade, memorização e realismo condicional na interface

A página `Governança` apresenta os cards `Diversidade e Memorização` e `Realismo Condicional` quando existem métricas em `evaluation.json`. Execuções antigas exibem `Não avaliado`, sem converter ausência de dado em zero.

As fontes exibidas incluem `evaluation.json → privacy`, `evaluation.json → conditional_income`, `quality_gates.json` e `manifest.json`. A interface não carrega datasets completos para montar o histórico.

Correspondência exata com treino é um indicador de possível memorização, mas não constitui prova definitiva de vazamento. Correspondência com holdout ajuda a interpretar coincidências inerentes à distribuição.

O `income_model_version` é exibido separadamente da versão do vocabulário. A versão 3 refina a calibração sintética de renda e deve ser interpretada como parâmetro experimental do projeto, não como dado salarial oficial.

O artefato CTGAN criado após a confirmação independente da candidate_c possui finalidade `recommended_candidate`. Esse status indica que o artefato pode ser considerado em uma etapa posterior de aprovação, mas não significa `approved`, `default` ou `production`.
