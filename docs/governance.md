# Governanca, LGPD e Uso Responsavel

O projeto e um artefato de pesquisa e experimentacao. Ele gera dados sinteticos locais e nao consulta Receita Federal, cartorios, operadoras, bases governamentais ou servicos externos.

Principios adotados:

- finalidade restrita a pesquisa, teste, homologacao e demonstracao;
- minimizacao de dados e ausencia de bases pessoais reais;
- identificadores ficticios gerados localmente;
- validacao estrutural sem consulta oficial;
- avaliacao de utilidade e indicadores de risco de memorizacao;
- rastreabilidade por configuracao, run id, hashes e manifestos.

Dados sinteticos nao devem ser automaticamente considerados anonimizados. As metricas de privacidade do pipeline sao indicadores e nao substituem avaliacao juridica, relatorio de impacto, governanca institucional ou auditoria.

Quality gates obrigatorios ausentes, invalidos ou `NaN` reprovam a execucao quando sao necessarios para aprovacao. Smoke tests pequenos servem para validar o funcionamento tecnico do pipeline e sao colocados em quarentena quando nao atingem o tamanho minimo configurado.

## Quality gates

| Gate | Metrica | Unidade | Limite padrao | Obrigatorio | Interpretacao e comportamento quando ausente |
| --- | --- | --- | --- | --- | --- |
| `min_evaluation_rows` | `row_counts.synthetic` | linhas | `100` | em `approval` | Tamanho minimo para sustentar aprovacao estatistica; ausente, invalido ou abaixo do limite rejeita em `approval` e coloca em quarentena nos demais modos. |
| `invalid_rows_max` | `validation.invalid_rows` | linhas | `0` | sim | Linhas finais estruturalmente invalidas; ausente, invalido ou acima do limite rejeita. |
| `duplicated_identifier_max` | contagem de motivos `*_duplicado` para identificadores | linhas | `0` | sim | Colisoes de identificadores gerados; ausente, invalido ou acima do limite rejeita. |
| `null_required_fields_max` | `validation.reason_counts.null_required_fields` | campos | `0` | sim | Campos obrigatorios nulos; ausente, invalido ou acima do limite rejeita. |
| `exact_train_match_rate_max` | `evaluation.privacy.exact_train_match_rate` | taxa | `0.01` | sim | Match exato com treino sobre atributos de modelo; ausente, invalido ou acima do limite rejeita. |
| `total_variation_distance_max` | maior TVD categorica contra holdout | distancia `[0, 1]` | `0.25` | nao | Deriva de distribuicoes categoricas; ausente, invalido ou acima do limite coloca em quarentena quando opcional. |
| `correlation_difference_max` | maior diferenca absoluta de correlacao contra holdout | diferenca absoluta | `0.30` | nao | Deriva da matriz de correlacao numerica; ausente, invalido ou acima do limite coloca em quarentena quando opcional. |

Uso proibido:

- fraude ou falsificacao documental;
- simulacao de identidade real;
- criacao indevida de contas;
- engenharia social;
- interacao com servicos reais;
- tomada de decisao sobre pessoas.
