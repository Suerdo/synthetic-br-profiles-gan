# Realismo condicional da renda

O modelo sintético de renda passou a usar perfis condicionais por ocupação. A renda continua sendo influenciada por idade, escolaridade, ocupação, região e variação estocástica, sem usar gênero.

A versão da calibração de renda é registrada separadamente como `income_model_version`. Essa versão não altera o schema nem a versão do vocabulário categórico.

## Caso observado

Foi observado um perfil com `Escolaridade = Ensino Médio`, `Ocupacao = Mecânico` e renda próxima de R$ 11.000,00. Essa combinação não é bloqueada automaticamente: um mecânico experiente, especializado, proprietário de oficina ou com renda variável pode atingir valores altos.

O problema avaliado é a frequência da cauda, sua relação com idade, região, escolaridade e ocupação, não a existência isolada de uma linha.

## Modelo de renda versão 2

O `income_model_version = 2` adiciona parâmetros sintéticos por ocupação:

- variabilidade da renda;
- probabilidade de cauda superior;
- escala da cauda superior;
- força do efeito de escolaridade;
- força do efeito de experiência;
- fator de localização.

Esses parâmetros são heurísticos e configuráveis. Eles não representam estatísticas oficiais do mercado de trabalho brasileiro.

Para `Mecânico`, a renda elevada continua possível, mas a probabilidade de cauda muito alta foi reduzida. Não há teto rígido específico por ocupação.

## Métricas

O relatório condicional calcula estatísticas por:

- `Ocupacao`;
- `Ocupacao + Escolaridade`;
- `Ocupacao + Faixa_Etaria`;
- `Ocupacao + Regiao`;
- `Ocupacao + Escolaridade + Faixa_Etaria`.

Para grupos com amostra suficiente, são registrados média, mediana, desvio padrão, p05, p25, p75, p90, p95, p99, mínimo, máximo, intervalo interquartil e taxas de cauda.

Os artefatos principais são:

- `conditional_income_summary.csv`;
- `conditional_income_summary.parquet`;
- `conditional_income_comparison.csv`;
- `conditional_income_tail_events.csv`;
- `income_plausibility_summary.json`.

Eventos de cauda não incluem nomes, CPF, telefone ou documentos.

## Interpretação

O realismo condicional avalia se as distribuições permanecem plausíveis dentro de contextos específicos. A média isolada pode esconder caudas excessivas; por isso p95 e p99 são monitorados.

Essas métricas não garantem representatividade da população brasileira, anonimização absoluta ou aderência a dados oficiais.

## Modelo de renda versão 3

O `income_model_version = 3` registra o refinamento conceitual da renda v2.1. A versão do vocabulário categórico permanece `categorical_vocabulary_version = 2`; a mudança é exclusiva da calibração sintética de renda.

Foram comparadas quatro versões: `income_v1`, `income_v2`, `income_v3_candidate_a` e `income_v3_candidate_b`. A seleção usou as seeds `41`, `42` e `43`; a confirmação da CTGAN usou seeds independentes `44`, `45` e `46`, que não participaram da escolha da calibração.

A versão selecionada foi `income_v3_candidate_b`, registrada como `selected_calibration`. Ela aumentou a dispersão em relação à v2 sem retornar à cauda excessiva da v1. Os parâmetros continuam sintéticos, configuráveis e não representam estatísticas oficiais do mercado de trabalho brasileiro.

Diagnóstico controlado de `Mecânico`, com `Ensino Médio`, 48 anos, região `Sudeste` e 5.000 amostras por seed:

| Versão | Mediana | P90 | P95 | P99 | Máximo | Taxa acima de R$ 10.000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `income_v1` | 3.949 | 8.455 | 10.370 | 15.319 | 34.526 | 14,5% |
| `income_v2` | 2.987 | 5.046 | 5.853 | 7.580 | 13.319 | 1,2% |
| `income_v3_candidate_a` | 3.026 | 5.271 | 6.197 | 8.275 | 15.042 | 2,0% |
| `income_v3_candidate_b` | 3.047 | 5.420 | 6.399 | 8.520 | 15.113 | 2,3% |

O valor de R$ 10.000,00 é usado apenas para diagnosticar o caso observado originalmente. Ele não é teto universal nem regra de validação por ocupação.

### Sobreposição e compressão

A seleção da v3 considera compressão de mediana, p95, p99, desvio padrão e intervalo interquartil em relação à v1, além da mudança de sobreposição em relação à v1 e à v2. Esses indicadores são informativos: não há quality gate universal para compressão nesta etapa.

O objetivo foi preservar sobreposição entre ocupações e manter extremos raros possíveis. Uma distribuição sem sobreposição seria artificialmente rígida; por outro lado, caudas muito amplas podem gerar perfis frequentes demais em regiões altas da distribuição.

### Confirmação da CTGAN candidate_c

A configuração CTGAN candidate_c foi congelada no perfil `ctgan_income_v3_recommended_candidate`, com `epochs: 20`, `batch_size: 500`, `generator_lr: 0.0001`, `discriminator_lr: 0.0001`, `generator_decay: 0.000001`, `discriminator_decay: 0.000001`, `discriminator_steps: 1`, `log_frequency: false` e `pac: 10`. A seed continua sendo definida por execução.

A confirmação independente foi executada com `configs/benchmark-ctgan-candidate-c-confirmation.yaml`, usando `income_model_version = 3`, vocabulário 2, treino de 20.000 linhas, holdout de 5.000 linhas e 20.000 sintéticos por seed.

| Seed | Status | Wasserstein norm. renda | KS renda | Duplicidade-base | Match exato com treino | Pico de RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 44 | `approved` | 0,213 | 0,122 | 0,000 | 0,000 | 4.059 MB |
| 45 | `approved` | 0,111 | 0,056 | 0,000 | 0,000 | 4.067 MB |
| 46 | `approved` | 0,067 | 0,032 | 0,000 | 0,000 | 3.799 MB |

O artefato `artifacts/models/ctgan/20260729T231900Z-income-v3-recommended-candidate/` foi criado com finalidade `recommended_candidate`. Ele não foi marcado como `approved`, `default` ou `production`.

As métricas `raw_evaluation.json`, `final_evaluation.json` e `raw_final_comparison.json` permanecem separadas para mostrar quanto da qualidade final veio diretamente da amostra bruta e quanto dependeu do pós-processamento e da seleção global.

## Diagnóstico da validade bruta da CTGAN income v3

Foi realizada uma investigação estritamente diagnóstica do artefato `artifacts/models/ctgan/20260729T231900Z-income-v3-recommended-candidate/`, sem treinamento novo, sem alteração da configuração candidate_c e sem sobrescrever os artefatos `raw_evaluation.json`, `final_evaluation.json` ou `raw_final_comparison.json` das execuções de confirmação.

Os artefatos diagnósticos foram gravados em:

```text
artifacts/diagnostics/ctgan-income-v3-raw-validity-20260730T001234Z/
```

### Semântica das taxas

As métricas foram formalizadas em `metric_semantics.json`:

| Métrica | Numerador | Denominador | Unidade | Estágio |
| --- | --- | --- | --- | --- |
| `raw_structural_validity_rate` | linhas brutas selecionadas que passam nas regras estruturais das colunas-base | linhas brutas selecionadas | taxa | `raw` |
| `final_structural_validity_rate` | linhas finais selecionadas que passam na validação estrutural completa | linhas finais selecionadas | taxa | `final` |
| `postprocessing_repair_rate` | ganho positivo de validade entre `final` e `raw` sobre as linhas selecionadas | linhas finais selecionadas | taxa | comparação `raw`/`final` |
| `postprocessing_rejection_rate` | candidatos inválidos depois do pós-processamento e da validação global | candidatos pós-processados gerados em todos os batches | taxa | pós-processamento |
| `candidate_acceptance_rate` | candidatos aceitos pelas regras locais de batch | candidatos gerados | taxa | geração em batches |
| `global_acceptance_rate` | candidatos aceitos pela validação global | candidatos gerados | taxa | seleção global |

Nesta análise, `postprocessing_repair_rate` é uma taxa histórica agregada de recuperação de validade estrutural, não uma contagem de campos modificados. Para evitar ambiguidade, o diagnóstico separa também `repair`, `replacement` e `rejection` por linha e por campo.

### Validade por regra

Nas seeds independentes `44`, `45` e `46`, a CTGAN preservou bem valores individuais e relações profissionais, mas aprendeu mal relações geográficas conjuntas:

| Seed | Validade bruta global | Categorias conhecidas | Validade profissional | Validade geográfica | Validade não relacional | Regra dominante |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 44 | 0,205% | 100,000% | 92,705% | 0,210% | 92,325% | `Estado × Município` |
| 45 | 0,180% | 100,000% | 92,920% | 0,190% | 92,770% | `Estado × DDD` |
| 46 | 0,285% | 100,000% | 94,230% | 0,305% | 94,205% | `Estado × Município` |

A validade global bruta é próxima de 0,2% porque ela representa a interseção de todas as regras. A regra dominante foi geográfica: a maior parte das linhas continha `Estado`, `Município` e/ou `DDD` individualmente conhecidos, mas em combinações incompatíveis.

### Pós-processamento por campo

Os campos mais alterados foram `Municipio`, `DDD` e `Regiao`. `Estado`, `Escolaridade`, `Ocupacao` e `Estado_Civil` não precisaram de alteração nas linhas selecionadas.

| Seed | Linhas selecionadas | Linhas inalteradas | Linhas reparadas | Linhas substituídas | Linhas rejeitadas | Linhas com múltiplas alterações |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 44 | 20.000 | 41 | 19.765 | 194 | 3.930 | 19.101 |
| 45 | 20.000 | 36 | 19.871 | 93 | 4.245 | 19.147 |
| 46 | 20.000 | 57 | 19.691 | 252 | 3.911 | 19.054 |

`repair` significa ajuste de campos relacionados preservando o núcleo semântico principal da linha. `replacement` indica substituição integral de algum campo semântico central. `rejection` indica candidato não aproveitado após validação global.

### Ocupação ausente

Nas seeds `44` e `45`, a ocupação ausente no resultado selecionado foi `Diretor`. Ela era rara no treino e no holdout e apareceu entre candidatos rejeitados ou excedentes, sem evidência de remoção pelo pós-processamento. Na seed `46`, nenhuma ocupação ficou ausente.

### Distância de renda

A distância de Wasserstein da renda passou a ser registrada em duas escalas:

- `wasserstein_distance_absolute_brl`: distância absoluta em reais;
- `wasserstein_distance_normalized`: distância normalizada pelo IQR da referência, com fallback para desvio-padrão.

| Seed | Raw absoluto | Final absoluto | Raw normalizado | Final normalizado |
| ---: | ---: | ---: | ---: | ---: |
| 44 | R$ 446,33 | R$ 434,71 | 0,208 | 0,203 |
| 45 | R$ 253,40 | R$ 238,31 | 0,122 | 0,114 |
| 46 | R$ 166,22 | R$ 159,06 | 0,078 | 0,075 |

### Interpretação

A dependência do pós-processamento foi classificada como `alta`, com critérios explícitos: validade bruta global abaixo de 50% e taxa de linhas selecionadas com reparo próxima de 99%. Ela não foi classificada como `crítica` porque a taxa de rejeição ficou abaixo de 20%.

Esse diagnóstico não promove nem descarta o artefato. Ele mostra que a CTGAN candidate_c aprendeu adequadamente categorias individuais e relações profissionais, mas depende fortemente do pós-processamento para coerência geográfica e validade estrutural final.
