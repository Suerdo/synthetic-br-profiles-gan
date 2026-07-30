# Privacidade e diversidade

Esta etapa amplia a avaliação de diversidade e de indicadores de memorização sem usar identificadores derivados. As métricas são calculadas somente sobre as 11 colunas-base do modelo:

`Idade`, `Genero`, `Regiao`, `Estado`, `Municipio`, `Escolaridade`, `Estado_Civil`, `Ocupacao`, `Renda`, `Dependentes` e `DDD`.

As colunas `Nome`, `Data_Nascimento`, `CPF`, `CNH`, `RG`, `Titulo_Eleitor` e `Telefone` são excluídas porque são derivadas por pós-processamento e tornariam as linhas artificialmente únicas.

## Métricas

**Duplicidade de combinações-base** mede repetições exatas das colunas-base. O relatório diferencia ocorrências duplicadas, grupos duplicados, maior grupo duplicado e linhas pertencentes a grupos duplicados.

**Correspondência exata com treino** conta registros sintéticos cujas colunas-base coincidem com pelo menos um registro do treino. Essa métrica pode indicar possível memorização, mas também pode ocorrer por baixa cardinalidade, arredondamento da renda, coincidência estatística ou pela própria natureza programática da referência.

**Correspondência exata com holdout** é uma métrica de controle contra registros que não foram usados no treinamento. Ela ajuda a separar sinais de memorização de coincidências esperadas na distribuição.

**Similaridade com registros de referência** usa DCR e NNDR nas colunas-base. Esses indicadores apoiam auditoria, mas não garantem anonimização.

## Evidências

Os artefatos de evidência usam hashes SHA-256 de uma representação canônica das combinações-base. A representação normaliza textos em Unicode NFC, aplica aliases legados, arredonda `Renda` para duas casas decimais, normaliza inteiros e trata nulos como `<NA>`.

Os arquivos principais são:

- `memorization_metrics.json`;
- `duplicate_base_rows.json`;
- `exact_train_matches.json`;
- `exact_holdout_matches.json`.

Os hashes não devem ser tratados como prova absoluta de privacidade. Eles servem para rastrear evidências sem expor linhas completas.

## Quality gates

O gate `exact_train_match_rate_max` permanece obrigatório com limite inicial de `0.01`.

O novo gate `duplicate_base_row_rate_max` foi incluído como informativo, também com valor inicial de `0.01`. A ausência da métrica em execuções antigas é exibida como `Não avaliado`, não como zero.

Esses limites são parâmetros exploratórios do projeto, não limiares científicos universais.

## Avaliação raw e final

Nos benchmarks recentes, as métricas de diversidade e memorização são persistidas antes e depois do pós-processamento:

- `raw_evaluation.json`: avalia a amostra bruta das colunas-base geradas pelo sintetizador;
- `final_evaluation.json`: avalia o dataset final exportável;
- `raw_final_comparison.json`: resume validade bruta, validade final, reparos, rejeições e mudanças de distribuição.

Essa separação evita atribuir ao modelo neural uma qualidade criada pelo pipeline. Correções de pós-processamento não são penalizadas automaticamente, mas sua participação fica explícita.

## CTGAN candidate_c e renda v3

A confirmação independente da CTGAN candidate_c com `income_model_version = 3` concluiu três seeds de confirmação sem duplicidade de combinações-base e sem correspondência exata com treino. Esses resultados apoiam a classificação `recommended_candidate`, mas não aprovam automaticamente o artefato nem garantem anonimização.

As evidências continuam sanitizadas. Artefatos como `duplicate_base_rows.json`, `exact_train_matches.json` e `exact_holdout_matches.json` registram hashes, contagens e índices, sem expor nomes, CPF, telefone, documentos ou linhas completas.

## Diversidade geográfica

A representação `geography_model_version = 2` introduz métricas específicas para a chave interna `Geo_Key`, sem alterar as colunas exportadas. Essas métricas avaliam se a coerência geográfica foi preservada e se a distribuição não colapsou em poucas combinações.

Métricas principais:

- `geography_key_coverage`: proporção das chaves geográficas canônicas observadas na amostra;
- `geography_key_unique_count`: quantidade de chaves distintas observadas;
- `geography_key_duplicate_rate`: taxa de repetição de chaves, usada como indicador de concentração;
- `state_coverage`, `municipality_coverage` e `ddd_coverage`: cobertura marginal dos componentes geográficos;
- `region_distribution_tvd`, `state_distribution_tvd`, `municipality_distribution_tvd` e `geography_key_distribution_tvd`: distância de variação total entre referência e sintético;
- `rare_geography_key_coverage`: cobertura de combinações geográficas raras.

Na confirmação `ctgan-income-v3-geo-v2-confirmation-20260730T012716Z-d44f6686`, a cobertura de `Geo_Key`, estados, municípios e DDDs foi 100% nas três seeds. A TVD de `Geo_Key` ficou entre 0,098 e 0,111. Assim, a chave composta eliminou a incoerência geográfica bruta observada na representação independente, mas a distribuição geográfica continua sendo avaliada separadamente.
