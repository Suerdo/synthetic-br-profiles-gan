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
