# Governança, LGPD e uso responsável

O projeto é um artefato de pesquisa e experimentação. Ele gera dados sintéticos locais e não consulta Receita Federal, cartórios, operadoras, bases governamentais ou serviços externos.

Princípios adotados:

- finalidade restrita a pesquisa, teste, homologação e demonstração;
- minimização de dados e ausência de bases pessoais reais;
- identificadores fictícios gerados localmente;
- validação estrutural sem consulta oficial;
- avaliação de utilidade e indicadores de risco de memorização;
- rastreabilidade por configuração, run id, hashes e manifestos.

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
