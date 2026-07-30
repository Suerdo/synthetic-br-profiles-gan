# Saídas geradas

As execuções são versionadas por `run_id` em `artifacts/runs/<run_id>/`.

```text
artifacts/
  models/
    <model>/
      <run_id>/
        model/
  runs/
    <run_id>/
      approved/
      quarantine/
      manifest.json
      config.yaml
```

O dataset aprovado fica em `approved/`. Resultados reprovados ou em observação ficam em `quarantine/`.

Arquivos principais:

- `dataset.parquet`: formato principal de processamento.
- `dataset.xlsx`: exportação opcional para usuários.
- `validation.json`: validação estrutural.
- `evaluation.json`: métricas contra treino e holdout.
- `quality_gates.json`: status e falhas dos gates.
- `generation.json`: contabilidade de candidatos, aceitos, rejeitados e excedentes válidos.
- `manifest.json`: ambiente, hashes, modelo, seed, status e commit Git quando disponível.
- `train.parquet` e `holdout.parquet`: splits usados para avaliação.

Arquivos de nomes fixos em `data/outputs/` permanecem apenas para compatibilidade do script legado.

## Modelos reutilizáveis

O comando `train` grava artefatos reutilizáveis no diretório informado por `--output`.

```text
artifacts/
  models/
    ctgan-default/
      model.pkl
      metadata.json
      metadata_ctgan.json
      training_manifest.json
      training_config.yaml
    simple-gan-default/
      generator.keras
      discriminator.keras
      preprocessor.pkl
      metadata.json
      config.json
      training_history.json
      training_manifest.json
      training_config.yaml
    programmatic-default/
      config.json
      metadata.json
      training_manifest.json
      training_config.yaml
```

`training_manifest.json` identifica o tipo de artefato, modelo, seed, tamanhos de treino, holdout e calibração, colunas esperadas, configuração resolvida, ambiente, tempos e tamanho do modelo. Para `programmatic`, o campo `training_required` é `false`.

O manifesto de treinamento também registra:

- `data_locale`: idioma canônico dos valores textuais, atualmente `pt-BR`;
- `unicode_normalization`: forma Unicode utilizada, atualmente `NFC`;
- `categorical_vocabulary_version`: versão do vocabulário categórico, atualmente `2`.
- `income_model_version`: versão da calibração sintética de renda, independente do vocabulário;
- `geography_model_version`: versão da representação geográfica usada pelo artefato;
- `geography_catalog_version` e `geography_catalog_checksum`: versão e checksum do catálogo quando a CTGAN usa `Geo_Key`.

## Datasets gerados sob demanda

O comando `generate` exporta um arquivo no formato solicitado e cria um manifesto ao lado dele:

```text
artifacts/
  generations/
    programmatic-10000.csv
    programmatic-10000.manifest.json
```

Formatos suportados:

- `csv`: `utf-8-sig`, sem índice, com separador `;`, para preservar acentos e facilitar abertura no Excel para Windows.
- `json`: lista de objetos, UTF-8, `ensure_ascii=False`.
- `parquet`: formato principal para preservar tipos sempre que possível.

O manifesto da geração registra modelo, artefato de origem quando houver, quantidade de linhas, colunas finais, formato, seed, caminho de saída, tamanho do arquivo, tempos, validação estrutural e aviso de governança. A exportação é bloqueada quando a validação estrutural final falha.

Também são registrados:

- `data_locale`;
- `unicode_normalization`;
- `source_model_vocabulary_version`;
- `output_vocabulary_version`;
- `legacy_value_normalization_applied`.
- `source_model_geography_version`;
- `output_geography_model_version`;
- `geography_catalog_checksum`, quando o artefato neural usa geografia v2.

Esses campos permitem identificar quando um modelo neural legado gerou valores com vocabulário anterior e a saída foi normalizada para o vocabulário atual.

Modelos CTGAN com `geography_model_version = 2` também registram a representação geográfica usada no treinamento. `Geo_Key` é uma coluna interna: ela não aparece no dataset exportado, mas seu catálogo e checksum ficam preservados no artefato do modelo para rastreabilidade.

## Seleção de colunas na exportação

O comando `generate` sempre gera internamente as 18 colunas finais e executa a validação estrutural sobre esse schema completo. A seleção de colunas é aplicada somente depois dessa validação.

Fluxo:

```text
geração das 11 colunas-base
  → pós-processamento das 18 colunas finais
  → validação estrutural completa
  → projeção das colunas solicitadas
  → exportação
```

Quando `--columns` é usado, a ordem exportada segue exatamente a ordem solicitada. Colunas repetidas, inexistentes, vazias ou com capitalização incorreta são rejeitadas. Quando `--preset` é usado, a ordem segue o preset definido no catálogo.

Presets disponíveis:

- `completo`: todas as 18 colunas;
- `demografico`: dados demográficos, localização e perfil socioeconômico;
- `contato`: nome, localização e telefone;
- `documentos`: nome, data de nascimento e identificadores sintéticos;
- `minimo`: `Nome`, `Idade`, `Estado` e `CPF`.

Dependências internas registradas no catálogo não são adicionadas automaticamente ao arquivo exportado. Por exemplo, `Telefone` depende de `Estado` e `DDD`, mas uma exportação com `--columns Nome Telefone CPF` contém somente `Nome`, `Telefone` e `CPF`.

O manifesto de geração preserva os campos existentes e acrescenta:

- `requested_columns`: colunas solicitadas explicitamente ou pelo preset;
- `exported_columns`: colunas presentes no arquivo exportado;
- `internally_generated_columns`: colunas geradas e validadas internamente;
- `column_selection_mode`: `all`, `explicit` ou `preset`;
- `column_preset`: preset utilizado, quando houver;
- `internal_dependencies`: dependências registradas no catálogo para as colunas exportadas;
- `validation.validated_columns`: colunas validadas no schema completo;
- `validation.projection_after_validation`: indica que a projeção ocorreu depois da validação.

## Segurança dos modelos serializados

`SimpleTabularGAN` e `CTGANSynthesizer` usam artefatos serializados, como `pickle` ou formatos equivalentes. O carregamento deve ficar restrito a diretórios de modelos produzidos ou previamente aprovados pela aplicação. Esta fase não implementa upload arbitrário de modelos nem seletor para arquivos `.pkl` enviados por usuários.
## Artefatos de diversidade e renda condicional

Novas execuções de pipeline podem produzir:

- `memorization_metrics.json`;
- `duplicate_base_rows.json`;
- `exact_train_matches.json`;
- `exact_holdout_matches.json`;
- `conditional_income_summary.csv`;
- `conditional_income_summary.parquet`;
- `conditional_income_comparison.csv`;
- `conditional_income_tail_events.csv`;
- `income_plausibility_summary.json`.

Esses arquivos complementam `evaluation.json` e preservam campos legados como `duplicate_row_rate`, `exact_train_match_rate`, `exact_holdout_match_rate`, `unique_combinations` e `unique_combination_rate`.

Os arquivos de evidência usam hashes das combinações-base e não devem incluir nomes, CPF, telefone ou documentos.

## Artefatos geográficos

Modelos CTGAN treinados com `geography_model_version = 2` salvam:

- `metadata_ctgan_internal.json`: metadados da representação interna com `Geo_Key`;
- `geography_catalog.json`: catálogo determinístico de `Geo_Key`, `Regiao`, `Estado`, `Municipio` e `DDD`;
- `metadata_ctgan.json`: versão geográfica, checksum do catálogo, colunas externas e colunas internas de treinamento.

Benchmarks recentes também registram métricas geográficas em `evaluation.json`, `raw_evaluation.json`, `final_evaluation.json` e `raw_final_comparison.json`, incluindo validade geográfica bruta, cobertura de chaves, TVD geográfica e taxa de chave conhecida.
