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

## Datasets gerados sob demanda

O comando `generate` exporta um arquivo no formato solicitado e cria um manifesto ao lado dele:

```text
artifacts/
  generations/
    programmatic-10000.csv
    programmatic-10000.manifest.json
```

Formatos suportados:

- `csv`: UTF-8, sem índice, com separador `;`.
- `json`: lista de objetos, UTF-8, `ensure_ascii=False`.
- `parquet`: formato principal para preservar tipos sempre que possível.

O manifesto da geração registra modelo, artefato de origem quando houver, quantidade de linhas, colunas finais, formato, seed, caminho de saída, tamanho do arquivo, tempos, validação estrutural e aviso de governança. A exportação é bloqueada quando a validação estrutural final falha.
