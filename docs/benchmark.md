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

## Sensibilidade e escalabilidade

O benchmark de sensibilidade estende o piloto para comparar os três sintetizadores em conjuntos de treinamento com tamanhos exatos:

| Cenário | Treino | Holdout | Calibração total |
| --- | ---: | ---: | ---: |
| Dados reduzidos | 1.000 | 250 | 1.250 |
| Referência operacional | 5.000 | 1.250 | 6.250 |
| Limite superior do experimento atual | 20.000 | 5.000 | 25.000 |

Nesse modo, o parâmetro principal é `train_sizes`. Ele representa o tamanho exato do conjunto de treinamento, não o total da base de calibração. O total da calibração é calculado a partir de:

```text
holdout_rows = train_rows × holdout_fraction / (1 - holdout_fraction)
```

Com `holdout_fraction = 0.20`, isso equivale a `holdout_rows = train_rows × 0.25`.

Para cada combinação de seed e tamanho, o benchmark cria uma única calibração, um único conjunto de treinamento e um único conjunto de holdout. Esses mesmos dados são reutilizados pelos modelos `programmatic`, `simple_gan` e `ctgan`.

### Validação técnica

`configs/benchmark-scaling-smoke.yaml` executa:

```text
3 modelos × 3 tamanhos × 1 seed = 9 execuções
```

Essa configuração usa poucas épocas e serve para validar caminhos técnicos, artefatos, métricas, manifesto, monitoramento de memória e consolidação. Ela não deve ser interpretada como avaliação estatística conclusiva.

### Experimento principal

`configs/benchmark-scaling.yaml` executa:

```text
3 modelos × 3 tamanhos × 3 seeds = 27 execuções
```

Os hiperparâmetros dos modelos permanecem constantes entre os tamanhos. Isso significa que conjuntos maiores produzem mais batches por época e, no caso da GAN tabular densa simples, mais atualizações do gerador e do discriminador.

Para a GAN tabular densa simples com `batch_size: 128` e `epochs: 20`, o histórico real deve registrar:

| Treino | Batches por época | Updates do gerador | Updates do discriminador |
| ---: | ---: | ---: | ---: |
| 1.000 | 8 | 160 | 320 |
| 5.000 | 40 | 800 | 1.600 |
| 20.000 | 157 | 3.140 | 6.280 |

Para a CTGAN com `batch_size: 500`, o benchmark registra batches inferidos a partir de tamanho de treino, batch e épocas. Essa inferência é identificada como tal e não afirma que a rotina interna da biblioteca seja idêntica à rotina da GAN tabular densa simples.

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

Executar a validação de escalabilidade:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-scaling-smoke.yaml
```

Executar o experimento principal de escalabilidade:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-scaling.yaml
```

Sobrescrever tamanhos de treinamento:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-scaling.yaml \
  --train-sizes 1000 5000 20000
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
      environment.json
      resource_limits.json
      aggregate_by_model_and_size.json
      marginal_gains.json
      scalability_limits.json
      failures.json
      calibration/
        seed-11/
          calibration.parquet
          train.parquet
          holdout.parquet
          metadata.json
          train-1000/
            calibration.parquet
            train.parquet
            holdout.parquet
            metadata.json
      runs/
        <model>/
          seed-<seed>/
            run-reference.json
            train-1000/
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

No benchmark de escalabilidade, `results.parquet` e `run_summary.parquet` incluem também `train_size` e `holdout_size`. O arquivo `aggregate_by_model_and_size.json` agrega as métricas por modelo e tamanho de treinamento. O arquivo `marginal_gains.json` registra as mudanças entre `1000_to_5000`, `5000_to_20000` e `1000_to_20000`.

Valores negativos em métricas de distância, como Wasserstein normalizado da renda, KS da renda, TVD de gênero e diferença de correlação, indicam melhora relativa nessa métrica. Essa leitura não substitui a análise por métrica, porque nenhum resultado único resume qualidade, privacidade, custo e validade estrutural.

## Monitoramento de recursos

O benchmark registra:

- memória residente antes e depois do treinamento;
- pico aproximado de memória residente do processo;
- tempo de treinamento, geração, validação, avaliação e exportação;
- tamanho do modelo salvo;
- tamanho dos artefatos da execução;
- CPU e quantidade de threads disponíveis quando a medição está disponível.

A medição de memória usa `psutil`. Como TensorFlow e PyTorch podem alocar memória nativa, o valor deve ser interpretado como aproximação do processo observado, não como medição perfeita de todos os recursos do sistema.

`resource_limits` permite configurar limites operacionais opcionais. Valores `null` significam monitoramento sem interrupção automática. Quando `stop_larger_sizes_after_resource_failure` está ativo, tamanhos maiores do mesmo modelo podem ser pulados depois de uma falha de recurso em tamanho menor.

## Warm-up e ordem de execução

`execution.warmup_backends` carrega minimamente os backends opcionais antes das execuções e registra `backend_warmup_seconds`. Esse tempo não entra em `training_seconds`.

`execution.rotate_model_order_by_seed` reduz viés de cache alternando a ordem dos modelos por seed:

```text
seed 11: programmatic, simple_gan, ctgan
seed 22: simple_gan, ctgan, programmatic
seed 33: ctgan, programmatic, simple_gan
```

A ordenação final dos arquivos consolidados continua por tamanho, modelo e seed.

## Limites observados

`scalability_limits.json` informa, para cada modelo, os tamanhos testados, os tamanhos concluídos tecnicamente, o maior tamanho testado com sucesso e a primeira falha observada quando houver.

Uma execução `approved`, `quarantined` ou `rejected` concluiu tecnicamente. `rejected` indica falha de quality gate obrigatório, não necessariamente falha de escalabilidade. Falhas técnicas e limites operacionais são registrados separadamente como `failed` ou `resource_limited`.

A conclusão deve ser lida como limite observado neste ambiente. Exemplo adequado:

```text
A CTGAN foi executada com sucesso com até 20.000 registros de treinamento neste experimento.
```

Isso não significa que 20.000 seja o limite máximo da CTGAN.

## Capacidade operacional

O benchmark de capacidade operacional é separado do benchmark de qualidade. Seu objetivo é observar o maior tamanho de conjunto de treinamento executado com sucesso neste ambiente, sem declarar que esse tamanho seja o limite máximo absoluto de qualquer modelo.

As configurações são:

- `configs/benchmark-capacity-smoke.yaml`: validação técnica com 50.000 registros de treinamento, uma época e geração pequena;
- `configs/benchmark-capacity.yaml`: execução principal com 50.000, 100.000 e 200.000 registros de treinamento, cinco épocas e geração reduzida.

Executar o smoke de capacidade:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-capacity-smoke.yaml
```

Executar o benchmark principal de capacidade:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-capacity.yaml
```

Com `holdout_fraction: 0.20`, os tamanhos exatos são:

| Treino | Holdout | Calibração total |
| ---: | ---: | ---: |
| 50.000 | 12.500 | 62.500 |
| 100.000 | 25.000 | 125.000 |
| 200.000 | 50.000 | 250.000 |

O modo de capacidade usa `benchmark.type: capacity`. Nesse modo, cada combinação de modelo e tamanho é executada em um subprocesso separado. O processo principal cria os splits, inicia o subprocesso com uma lista de argumentos, monitora memória residente, registra `stdout.log`, `stderr.log`, código de saída e lê um `result.json` estruturado.

A memória reportada é a memória residente total observada para o processo do worker e seus filhos. Mesmo com isolamento por subprocesso, a medição continua dependente do sistema operacional e de como TensorFlow, PyTorch e bibliotecas nativas alocam memória.

Quando `execution.warmup_backends` está ativo, o worker carrega minimamente o backend do modelo antes de iniciar o pipeline. Esse tempo é registrado em `backend_warmup_seconds` e não entra em `training_seconds`.

### Progressão por modelo

A progressão é controlada separadamente por modelo. Se um modelo falhar por recurso ou erro técnico em 100.000 registros, 200.000 registros são pulados apenas para esse modelo. Os demais modelos continuam.

Estados técnicos usados nesse benchmark:

- `completed`: treinamento, geração e exportação concluíram com quality gates aprovados;
- `quality_quarantined`: execução técnica concluída, mas os quality gates deixaram o resultado em quarentena;
- `quality_rejected`: execução técnica concluída, mas houve falha obrigatória de quality gate;
- `resource_limited`: limite operacional de tempo ou memória foi excedido;
- `failed`: exceção técnica ou término inesperado;
- `skipped_after_failure`: tamanho pulado porque um tamanho anterior daquele modelo falhou;
- `backend_unavailable`: dependência opcional ausente.

Uma execução em `quality_quarantined` ou `quality_rejected` conta como concluída tecnicamente para capacidade operacional. Quality gates avaliam validade e qualidade, mas não determinam, sozinhos, se o modelo conseguiu operar naquele tamanho.

### Artefatos de capacidade

Os artefatos específicos incluem:

```text
artifacts/
  benchmarks/
    <capacity-benchmark-id>/
      benchmark_config.yaml
      benchmark_manifest.json
      capacity_summary.json
      capacity_results.parquet
      capacity_results.csv
      runs.json
      failures.json
      scalability_limits.json
      subprocesses/
        <model>/
          train-50000/
            run_config.yaml
            stdout.log
            stderr.log
            result.json
```

`capacity_results.parquet` e `capacity_results.csv` registram tamanho de treinamento, holdout, status técnico, status dos quality gates, duração, memória inicial, pico de memória residente, memória incremental, tamanho do modelo serializado, tamanho dos artefatos, batches e updates quando disponíveis.

`capacity_summary.json` e `scalability_limits.json` registram, por modelo, tamanhos testados, tamanhos concluídos, tamanhos pulados, primeira falha observada e uma conclusão textual restrita ao ambiente atual.

Poucas épocas validam capacidade técnica de execução, não convergência do modelo. Resultados de capacidade dependem de hardware, sistema operacional, versões das bibliotecas, CPU/GPU disponível e carga do ambiente.

## Fronteira superior de capacidade observada

Esta etapa amplia o benchmark de capacidade operacional para localizar a primeira falha observada dos modelos que concluíram tecnicamente até 200.000 registros de treinamento. O experimento executa apenas `programmatic` e `ctgan`; a `simple_gan` não é executada nesta etapa, porque já possui sucesso observado em 100.000 registros e falha observada em 200.000 registros no experimento anterior.

As configurações são:

- `configs/benchmark-capacity-upper-smoke.yaml`: validação técnica com `programmatic` e `ctgan`, uma seed, 400.000 registros de treinamento e 100 registros sintéticos;
- `configs/benchmark-capacity-upper.yaml`: execução principal com `programmatic` e `ctgan`, uma seed, tamanhos de 400.000, 800.000 e 1.600.000 registros de treinamento e 1.000 registros sintéticos.

Executar o smoke da fronteira superior:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-capacity-upper-smoke.yaml
```

Executar o benchmark principal da fronteira superior:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-capacity-upper.yaml
```

Com `holdout_fraction: 0.20`, os tamanhos exatos são:

| Treino | Holdout | Calibração total |
| ---: | ---: | ---: |
| 400.000 | 100.000 | 500.000 |
| 800.000 | 200.000 | 1.000.000 |
| 1.600.000 | 400.000 | 2.000.000 |

A progressão permanece independente por modelo. Se a CTGAN falhar em 800.000 registros, 1.600.000 registros serão marcados como `skipped_after_failure` apenas para a CTGAN. O modelo programático continuará a progressão configurada, desde que o tamanho anterior desse modelo tenha sido tecnicamente concluído.

São considerados tecnicamente concluídos os status `completed`, `quality_quarantined` e `quality_rejected`. Os status `resource_limited`, `failed`, `skipped_after_failure` e `backend_unavailable` não contam como conclusão técnica. Uma execução em quarentena ou rejeitada por quality gate pode, portanto, contar como concluída para capacidade operacional.

Os resultados estruturados são gravados em `capacity_summary.json`, `scalability_limits.json`, `capacity_results.csv`, `capacity_results.parquet` e `failures.json`. Cada falha registra, quando disponível, estágio, tipo, mensagem, código de saída, sinal nativo, duração até a falha, memória inicial, pico de RSS, memória incremental, caminhos de `stdout.log` e `stderr.log`, disponibilidade de `result.json`, último evento registrado pelo worker, sistema operacional, versão do Python, versão do backend, CPU e GPU.

Os resultados representam a capacidade observada no ambiente testado. O maior tamanho concluído não deve ser interpretado como limite máximo absoluto do modelo; o limite máximo absoluto não foi determinado.

### Registro de falhas observadas

Execução documentada:

- `benchmark_id`: `capacity-upper-bound-20260727T220738Z-32ffebd0`;
- sistema operacional: Windows 11 (`Windows-11-10.0.26200-SP0`);
- Python: `3.13.13`;
- CTGAN: `0.12.1`;
- TensorFlow: `2.21.0`;
- PyTorch: `2.13.0`;
- CPU lógica reportada: `32`;
- GPU CUDA reportada pelo worker da CTGAN: indisponível;
- status geral: `completed`;
- execuções esperadas: `6`;
- execuções concluídas tecnicamente: `6`;
- falhas registradas em `failures.json`: `0`.

| Modelo | Treino | Holdout | Status técnico | Status de qualidade | Tempo de treino | Duração total | Pico de RSS | Tamanho do modelo |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `programmatic` | 400.000 | 100.000 | `completed` | `approved` | 0,030 s | 5,998 s | 3.501,5 MiB | 0,01 MiB |
| `ctgan` | 400.000 | 100.000 | `quality_quarantined` | `quarantined` | 99,801 s | 106,756 s | 8.343,9 MiB | 26,28 MiB |
| `programmatic` | 800.000 | 200.000 | `completed` | `approved` | 0,055 s | 9,319 s | 3.773,6 MiB | 0,01 MiB |
| `ctgan` | 800.000 | 200.000 | `completed` | `approved` | 201,622 s | 211,657 s | 9.587,8 MiB | 50,69 MiB |
| `programmatic` | 1.600.000 | 400.000 | `completed` | `approved` | 0,112 s | 16,460 s | 4.035,4 MiB | 0,01 MiB |
| `ctgan` | 1.600.000 | 400.000 | `completed` | `approved` | 408,202 s | 425,602 s | 14.607,5 MiB | 99,52 MiB |

#### ProgrammaticSynthesizer

- Maior tamanho concluído: 1.600.000 registros de treinamento.
- Primeira falha observada: não observada nesta execução.
- Tamanhos pulados: nenhum.
- Status dos quality gates: `approved` em todos os tamanhos.
- Interpretação: o modelo programático foi executado com sucesso com pelo menos 1.600.000 registros neste ambiente. O limite máximo absoluto não foi determinado.
- Intervalo operacional observado: limite inferior observado de 1.600.000 registros; limite superior não determinado.

#### CTGANSynthesizer

- Maior tamanho concluído: 1.600.000 registros de treinamento.
- Primeira falha observada: não observada nesta execução.
- Tamanhos pulados: nenhum.
- Status dos quality gates: `quality_quarantined` em 400.000 registros e `completed` com qualidade `approved` em 800.000 e 1.600.000 registros.
- Evidências: `result.json` foi produzido para todos os tamanhos, `exit_code` foi `0` em todos os subprocessos e `failures.json` não registrou falhas.
- Interpretação: a CTGAN foi executada com sucesso com pelo menos 1.600.000 registros neste ambiente. O limite máximo absoluto não foi determinado.
- Intervalo operacional observado: limite inferior observado de 1.600.000 registros; limite superior não determinado.

Rodadas adicionais executadas em configurações separadas:

- `capacity-upper-bound-20260727T223209Z-8cc6eb01`: teste exclusivo de 3.200.000 registros de treinamento, com 800.000 registros de holdout e 4.000.000 registros de calibração;
- `capacity-upper-bound-20260727T233618Z-c6a38d16`: teste exclusivo de 6.400.000 registros de treinamento, com 1.600.000 registros de holdout e 8.000.000 registros de calibração.

| Modelo | Treino | Holdout | Status técnico | Status de qualidade | Tempo de treino | Duração total | Pico de RSS | Tamanho do modelo |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `programmatic` | 3.200.000 | 800.000 | `completed` | `approved` | 0,232 s | 31,872 s | 5.843,4 MiB | 0,01 MiB |
| `ctgan` | 3.200.000 | 800.000 | `completed` | `approved` | 2.446,593 s | 2.486,856 s | 20.905,9 MiB | 197,17 MiB |
| `programmatic` | 6.400.000 | 1.600.000 | `completed` | `approved` | 0,490 s | 61,440 s | 8.180,7 MiB | 0,01 MiB |
| `ctgan` | 6.400.000 | 1.600.000 | `failed` | não aplicável | não concluído | 858,097 s até a falha | 22.140,0 MiB | não produzido |

#### ProgrammaticSynthesizer, situação consolidada

- Maior tamanho concluído: 6.400.000 registros de treinamento.
- Primeira falha observada: não observada.
- Tamanhos pulados: nenhum.
- Status dos quality gates: `approved` em todos os tamanhos executados nesta etapa.
- Interpretação: o modelo programático foi executado com sucesso com pelo menos 6.400.000 registros neste ambiente. O limite máximo absoluto não foi determinado.
- Intervalo operacional observado: limite inferior observado de 6.400.000 registros; limite superior não determinado.

#### CTGANSynthesizer, situação consolidada

- Maior tamanho concluído: 3.200.000 registros de treinamento.
- Primeira falha observada: 6.400.000 registros de treinamento.
- Status da primeira falha: `failed`.
- Estágio registrado: `pipeline`.
- Tipo da falha: `OSError`.
- Código de saída do worker: `4`.
- Evidências: `result.json` foi produzido, `failure_message` registrou `[WinError 1450] Não existem recursos de sistema suficientes para concluir o serviço solicitado`, o pico de RSS observado foi de 22.140,0 MiB e `failures.json` registrou a falha.
- Interpretação: a CTGAN foi executada com sucesso até 3.200.000 registros neste ambiente. A primeira falha foi observada em 6.400.000 registros. O limite máximo absoluto não foi determinado.
- Intervalo operacional observado: entre 3.200.000 e 6.400.000 registros neste ambiente, sob os hiperparâmetros e versões utilizados.

Para refinar a fronteira operacional observada da CTGAN, o próximo tamanho intermediário recomendado é 4.800.000 registros de treinamento. Para o modelo programático, que ainda não apresentou falha até 6.400.000 registros, o próximo patamar de exploração seria 12.800.000 registros. Nenhum desses tamanhos foi executado automaticamente nesta documentação.

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
- No modo de escalabilidade, a comparação usa a mesma calibração, o mesmo conjunto de treinamento e o mesmo conjunto de holdout para todos os modelos dentro de cada combinação de seed e `train_size`.
- As mesmas métricas, validadores, quality gates e colunas excluídas das métricas de privacidade são aplicados a todos os modelos.
- `execution.parallelism` permanece `1` por padrão para evitar competição por memória entre TensorFlow, CTGAN e PyTorch.
- O ranking exploratório existe apenas como configuração futura e permanece desativado por padrão.
- O experimento com três seeds é exploratório. Ele ajuda a observar tendências, mas não prova saturação definitiva nem superioridade geral de um modelo.

## Limitações

O benchmark-piloto é uma ferramenta exploratória. Três seeds ainda não produzem evidência estatística definitiva. Menor distância estatística não implica maior privacidade, e nenhum modelo é necessariamente superior em todos os critérios.

Os resultados dependem da base de calibração controlada. Como a calibração não representa perfeitamente a população brasileira, as conclusões devem ser interpretadas como comparação experimental dentro deste ambiente, não como medida final de qualidade populacional.

O cenário de 1.000 registros é útil como limite inferior, mas o holdout de 250 linhas não oferece evidência robusta para categorias raras. O cenário de 5.000 registros funciona como referência operacional do piloto. O cenário de 20.000 registros é apenas o limite superior do experimento atual, não um limite máximo absoluto dos modelos.
