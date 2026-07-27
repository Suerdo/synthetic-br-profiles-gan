# Benchmark experimental

Este documento descreve o benchmark-piloto para comparar os sintetizadores `programmatic`, `simple_gan` e `ctgan`.

O benchmark reutiliza as abstrações existentes do projeto: geração da base de calibração, divisão entre conjunto de treinamento e conjunto de holdout, treinamento dos modelos, geração de dados sintéticos, validação estrutural, métricas, quality gates, artefatos e manifestos.

## Desenho experimental

A matriz piloto definida em `configs/benchmark.yaml` executa:

| Parâmetro | Valor |
| --- | --- |
| Modelos | `programmatic`, `simple_gan`, `ctgan` |
| Seeds | `11`, `22`, `33` |
| Base de calibração | `5000` registros |
| Conjunto de holdout | `20%` |
| Dados sintéticos por execução | `2000` registros |
| Modo de avaliação | `experimental` |

Total esperado:

```text
3 modelos × 3 seeds = 9 execuções
```

Para cada seed, o benchmark cria uma única base de calibração e um único par treinamento/holdout. Esses mesmos dados são reutilizados por todos os modelos naquela seed. Essa decisão evita que diferenças entre modelos sejam confundidas com diferenças entre bases de calibração.

## Como reproduzir

Executar o piloto completo:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark.yaml
```

Sobrescrever modelos e seeds:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark.yaml \
  --models programmatic simple_gan ctgan \
  --seeds 11 22 33
```

Executar apenas a validação rápida programática:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-programmatic.yaml
```

## Artefatos

Cada benchmark recebe um `benchmark_id` próprio e salva resultados em:

```text
artifacts/
  benchmarks/
    <benchmark_id>/
      benchmark_config.yaml
      benchmark_manifest.json
      runs.json
      results.parquet
      results.csv
      run_summary.parquet
      run_summary.csv
      summary.json
      failures.json
      calibration/
        seed-11/
          calibration.parquet
          train.parquet
          holdout.parquet
          metadata.json
      runs/
        <model>/
          seed-<seed>/
            run-reference.json
      diagnostics/
```

Os artefatos completos de cada execução continuam no sistema existente de `artifacts/runs/<run_id>/`. O diretório do benchmark guarda referências para esses `run_id`, não cópias completas dos datasets individuais.

## Métricas consolidadas

O arquivo `results.parquet` usa formato longo. Cada linha identifica `benchmark_id`, `run_id`, modelo, seed, status, grupo de métrica, nome da métrica, coluna, valor, referência, diferença e detalhes.

O arquivo `run_summary.parquet` resume as principais métricas por execução:

- Wasserstein normalizado da renda;
- KS da renda;
- TVD de gênero;
- diferença média de correlação;
- taxa de duplicidade;
- match exato com treinamento;
- Distance to Closest Record;
- Nearest Neighbor Distance Ratio;
- duração de treinamento;
- duração de geração.

O arquivo `summary.json` agrega resultados por modelo, incluindo contagens de status e estatísticas descritivas das métricas principais: média, mediana, desvio-padrão, mínimo, máximo e intervalo de confiança exploratório de 95% quando há dados suficientes.

## Estados e falhas

Cada execução individual preserva os estados do pipeline:

- `approved`: todos os critérios aplicáveis foram atendidos;
- `quarantined`: critérios obrigatórios passaram, mas há alertas ou critérios informativos não atendidos;
- `rejected`: pelo menos um critério obrigatório falhou.

O benchmark usa também o estado `failed` para execuções que não concluíram por erro técnico, como dependência opcional ausente ou falha de treinamento.

O status geral do benchmark pode ser:

- `completed`: todas as execuções concluíram;
- `completed_with_failures`: parte das execuções concluiu e parte falhou;
- `failed`: nenhuma execução válida foi concluída.

Quando `continue_on_error: true`, falhas individuais são registradas em `failures.json` e em arquivos de diagnóstico, e as demais execuções continuam. Quando `continue_on_error: false`, o benchmark interrompe na primeira falha.

## Decisões metodológicas

- A comparação usa a mesma calibração, o mesmo conjunto de treinamento e o mesmo conjunto de holdout para todos os modelos dentro de uma mesma seed.
- As mesmas métricas, validadores, quality gates e colunas excluídas das métricas de privacidade são aplicados a todos os modelos.
- `execution.parallelism` permanece `1` por padrão para evitar competição por memória entre TensorFlow, CTGAN e PyTorch.
- O ranking exploratório existe apenas como configuração futura e permanece desativado por padrão.

## Limitações

O benchmark-piloto é uma ferramenta exploratória. Três seeds ainda não produzem evidência estatística definitiva. Menor distância estatística não implica maior privacidade, e nenhum modelo é necessariamente superior em todos os critérios.

Os resultados dependem da base de calibração controlada. Como a calibração não representa perfeitamente a população brasileira, as conclusões devem ser interpretadas como comparação experimental dentro deste ambiente, não como medida final de qualidade populacional.
