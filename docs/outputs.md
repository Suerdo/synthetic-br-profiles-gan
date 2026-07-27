# Saidas Geradas

As execucoes sao versionadas por `run_id` em `artifacts/runs/<run_id>/`.

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

O dataset aprovado fica em `approved/`. Resultados reprovados ou em observacao ficam em `quarantine/`.

Arquivos principais:

- `dataset.parquet`: formato principal de processamento.
- `dataset.xlsx`: exportacao opcional para usuarios.
- `validation.json`: validacao estrutural.
- `evaluation.json`: metricas contra treino e holdout.
- `quality_gates.json`: status e falhas dos gates.
- `generation.json`: contabilidade de candidatos, aceitos, rejeitados e excedentes validos.
- `manifest.json`: ambiente, hashes, modelo, seed, status e commit Git quando disponivel.
- `train.parquet` e `holdout.parquet`: splits usados para avaliacao.

Arquivos de nomes fixos em `data/outputs/` permanecem apenas para compatibilidade do script legado.
