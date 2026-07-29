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
