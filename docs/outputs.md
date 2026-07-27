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
