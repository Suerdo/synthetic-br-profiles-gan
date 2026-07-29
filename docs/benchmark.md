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
- `configs/benchmark-capacity-upper.yaml`: execução principal com `programmatic` e `ctgan`, uma seed, tamanhos de 400.000, 800.000 e 1.600.000 registros de treinamento e 1.000 registros sintéticos;
- `configs/benchmark-capacity-ctgan-refinement.yaml`: refinamento final exclusivo da CTGAN com 4.800.000 registros de treinamento, 1.200.000 registros de holdout e 6.000.000 registros de calibração.

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

Executar o refinamento final exclusivo da CTGAN:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-capacity-ctgan-refinement.yaml
```

Com `holdout_fraction: 0.20`, os tamanhos exatos são:

| Treino | Holdout | Calibração total |
| ---: | ---: | ---: |
| 400.000 | 100.000 | 500.000 |
| 800.000 | 200.000 | 1.000.000 |
| 1.600.000 | 400.000 | 2.000.000 |
| 4.800.000 | 1.200.000 | 6.000.000 |

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
- `capacity-upper-bound-20260727T233618Z-c6a38d16`: teste exclusivo de 6.400.000 registros de treinamento, com 1.600.000 registros de holdout e 8.000.000 registros de calibração;
- `capacity-ctgan-refinement-20260728T001319Z-1dd7fa14`: refinamento final exclusivo da CTGAN com 4.800.000 registros de treinamento, 1.200.000 registros de holdout e 6.000.000 registros de calibração.

| Modelo | Treino | Holdout | Status técnico | Status de qualidade | Tempo de treino | Duração total | Pico de RSS | Tamanho do modelo |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `programmatic` | 3.200.000 | 800.000 | `completed` | `approved` | 0,232 s | 31,872 s | 5.843,4 MiB | 0,01 MiB |
| `ctgan` | 3.200.000 | 800.000 | `completed` | `approved` | 2.446,593 s | 2.486,856 s | 20.905,9 MiB | 197,17 MiB |
| `ctgan` | 4.800.000 | 1.200.000 | `completed` | `approved` | 1.473,487 s | 1.525,077 s | 24.305,2 MiB | 294,83 MiB |
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

- Maior tamanho concluído: 4.800.000 registros de treinamento.
- Primeira falha observada: 6.400.000 registros de treinamento.
- Status registrado no artefato histórico da primeira falha: `failed`; interpretação operacional: limitação de recursos.
- Estágio registrado: `pipeline`.
- Tipo da falha: `OSError`.
- Código de saída do worker: `4`.
- Evidências da falha em 6.400.000: `result.json` foi produzido, `failure_message` registrou `[WinError 1450] Não existem recursos de sistema suficientes para concluir o serviço solicitado`, o pico de RSS observado foi de 22.140,0 MiB e `failures.json` registrou a falha. Essa mensagem indica limitação operacional de recursos, preservando `OSError` como tipo original da exceção.
- Evidências do sucesso em 4.800.000: `result.json` foi produzido, `exit_code` foi `0`, o status técnico foi `completed`, o status de qualidade foi `approved`, o tempo de treinamento foi de 1.473,487 s e o pico de RSS observado foi de 24.305,2 MiB.
- Interpretação: a CTGAN foi executada com sucesso até 4.800.000 registros neste ambiente. A primeira falha por insuficiência de recursos foi observada em 6.400.000 registros. O limite máximo absoluto não foi determinado.
- Intervalo operacional observado: entre 4.800.000 e 6.400.000 registros neste ambiente, sob os hiperparâmetros e versões utilizados.
- Encerramento: a busca de capacidade da CTGAN foi encerrada após esse refinamento final. Não foram executados tamanhos intermediários adicionais.

A CTGAN foi executada com sucesso com até 4.800.000 registros de treinamento no ambiente avaliado. A primeira falha por insuficiência de recursos foi observada em 6.400.000 registros. Assim, a fronteira operacional observada ficou entre 4.800.000 e 6.400.000 registros. Esse intervalo depende do hardware, do sistema operacional, das versões das bibliotecas e dos hiperparâmetros utilizados, não devendo ser interpretado como limite máximo absoluto do modelo.

### Situação consolidada dos três modelos

| Modelo | Maior sucesso observado | Primeira falha observada | Interpretação |
| --- | ---: | ---: | --- |
| Programático | 6.400.000 | nenhuma | Busca encerrada por decisão metodológica |
| CTGAN | 4.800.000 | 6.400.000 | Fronteira refinada |
| GAN simples | 100.000 | 200.000 | Falha técnica observada |

### Encerramento da busca de capacidade do sintetizador programático

O `ProgrammaticSynthesizer` foi executado com sucesso com 6.400.000 registros de treinamento equivalentes no ambiente avaliado. Como esse sintetizador não possui etapa real de treinamento neural, sua principal limitação está associada à materialização e ao processamento dos dados em memória, e não à capacidade de aprendizagem de um modelo generativo.

O consumo observado nesse modelo decorre principalmente da criação do dataset de calibração, da materialização de DataFrames, do armazenamento de strings e colunas categóricas, da criação do conjunto de holdout, da serialização em Parquet, do pós-processamento dos perfis e da disponibilidade geral de memória do ambiente.

Por esse motivo, não será realizado, nesta fase, o teste com 12.800.000 registros. Encontrar o ponto em que o ambiente deixa de materializar DataFrames, strings, colunas categóricas, holdout e artefatos Parquet possui menor valor científico para os objetivos atuais do projeto do que caracterizar a fronteira operacional da CTGAN.

Essa decisão é metodológica, não uma limitação conhecida do código. O valor de 6.400.000 registros representa apenas o maior tamanho testado com sucesso. O limite máximo absoluto do sintetizador programático não foi determinado.

## Qualidade dos sintetizadores com o vocabulário 2

O benchmark de qualidade do vocabulário 2 avalia o espaço categórico em português brasileiro, com normalização Unicode `NFC` e 37 ocupações estruturadas. O objetivo é medir cobertura, coerência e diversidade do novo vocabulário, especialmente nas relações entre `Ocupacao`, `Escolaridade`, `Idade` e `Renda`. Essa etapa não mede capacidade máxima e não ajusta pesos, multiplicadores ou regras sintéticas.

A avaliação distingue dois estágios:

- `raw`: saída diagnóstica imediatamente produzida por `synthesizer.sample(...)`, antes de aliases, normalização linguística, pós-processamento e validação final;
- `final`: resultado após normalização pt-BR, aliases, normalização estrutural, pós-processamento e validação final.

Essa separação evita atribuir ao modelo neural uma qualidade que tenha sido introduzida depois pelo pipeline. O resultado `final` demonstra a qualidade do dataset exportável; o resultado `raw` mostra quanto o sintetizador aprendeu diretamente.

Configurações adicionadas:

- `configs/benchmark-quality-vocab-v2-smoke.yaml`: validação técnica pequena com os três modelos, uma seed e 1.000 registros de treinamento;
- `configs/benchmark-quality-vocab-v2.yaml`: benchmark principal com os três modelos, seeds `41`, `42` e `43`, 20.000 registros de treinamento, 5.000 registros de holdout e 20.000 registros sintéticos por execução.

Com `holdout_fraction: 0.20`, cada seed do benchmark principal usa:

| Conjunto | Registros |
| --- | ---: |
| Treinamento | 20.000 |
| Holdout | 5.000 |
| Calibração total | 25.000 |

Execução principal documentada:

- `benchmark_id`: `quality-vocab-v2-20260728T131154Z-364a35cb`;
- sistema operacional: Windows 11 (`Windows-11-10.0.26200-SP0`);
- Python: `3.13.13`;
- `ctgan`: `0.12.1`;
- TensorFlow: `2.21.0`;
- PyTorch: `2.13.0`;
- `pandas`: `2.3.3`;
- `pyarrow`: `24.0.0`;
- CPU lógica reportada: `32`;
- GPU CUDA reportada: indisponível.

Hiperparâmetros do benchmark principal:

| Modelo | Épocas | Batch | Observação |
| --- | ---: | ---: | --- |
| `programmatic` | não aplicável | não aplicável | Não possui treinamento neural |
| `simple_gan` | 20 | 128 | GAN tabular densa simples em Keras |
| `ctgan` | 20 | 500 | CTGAN da biblioteca `ctgan` |

Resumo por execução:

| Modelo | Seed | Status | Linhas válidas | Cobertura raw | Cobertura final | Validade escolaridade-ocupação raw | Validade final | Wasserstein norm. renda | KS renda |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `programmatic` | 41 | `approved` | 20.000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,025 | 0,017 |
| `programmatic` | 42 | `approved` | 20.000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,022 | 0,011 |
| `programmatic` | 43 | `rejected` | 19.999 | 1,000 | 1,000 | 1,000 | 1,000 | 0,020 | 0,015 |
| `simple_gan` | 41 | `quarantined` | 2.259 | 0,081 | 0,135 | 0,020 | 1,000 | 0,883 | 0,677 |
| `simple_gan` | 42 | `quarantined` | 20.000 | 0,189 | 0,162 | 0,943 | 1,000 | 0,657 | 0,426 |
| `simple_gan` | 43 | `rejected` | 19.999 | 0,027 | 0,027 | 1,000 | 1,000 | 0,852 | 0,643 |
| `ctgan` | 41 | `approved` | 20.000 | 1,000 | 1,000 | 0,914 | 1,000 | 0,118 | 0,116 |
| `ctgan` | 42 | `approved` | 20.000 | 1,000 | 1,000 | 0,912 | 1,000 | 0,115 | 0,040 |
| `ctgan` | 43 | `rejected` | 19.999 | 1,000 | 1,000 | 0,913 | 1,000 | 0,324 | 0,139 |

Agregação exploratória por modelo:

| Modelo | Cobertura raw média | Cobertura final média | Distância ocupação raw | Distância ocupação final | Entropia raw | Entropia final | Maior ocupação raw | Maior ocupação final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `programmatic` | 1,000 | 1,000 | 0,034 | 0,036 | 3,251 | 3,249 | 0,077 | 0,079 |
| `simple_gan` | 0,099 | 0,108 | 0,912 | 0,885 | 0,394 | 0,534 | 0,837 | 0,690 |
| `ctgan` | 1,000 | 1,000 | 0,105 | 0,129 | 3,291 | 3,077 | 0,088 | 0,105 |

O `ProgrammaticSynthesizer` e a `CTGANSynthesizer` cobriram as 37 ocupações nas três seeds, tanto em `raw` quanto em `final`. A `SimpleTabularGAN` apresentou forte concentração em poucas categorias: a cobertura final média foi de aproximadamente 10,8%, com 31 a 36 ocupações ausentes por seed.

Para ocupações raras, definidas como ocupações com participação inferior a 1% no conjunto de holdout, `programmatic` e `ctgan` reproduziram todas as categorias raras observadas nas três seeds. A `simple_gan` reproduziu 0 de 13 ou 14 ocupações raras no resultado final das três seeds; na saída `raw`, reproduziu uma ocupação rara apenas na seed `42`.

As combinações `Escolaridade`-`Ocupacao` foram totalmente válidas no resultado `final` de todos os modelos, porque a validação e a normalização estrutural atuam antes da exportação. Na saída `raw`, a CTGAN preservou cerca de 91,3% de combinações válidas. A GAN simples variou bastante: uma seed apresentou validade bruta de apenas 2,0%, com concentração em combinações como `Fundamental + Técnico`.

As combinações `Idade`-`Ocupacao` também foram totalmente válidas no resultado `final`. Na saída `raw`, a CTGAN ficou entre 97,7% e 98,4% de validade, principalmente por produzir casos de `Aposentado` em idades abaixo do limite obrigatório. A GAN simples ficou próxima de 100% nessa métrica específica, embora com baixa diversidade ocupacional.

A comparação de renda por ocupação confirma a vantagem estrutural do modelo programático, porque a referência também é gerada pelas mesmas regras sintéticas. O modelo programático preservou melhor as diferenças esperadas entre pares como `Médico` versus `Atendente` e `Gerente` versus `Auxiliar Administrativo`. A CTGAN reproduziu cobertura ampla, mas reduziu fortemente a separação média entre algumas ocupações qualificadas e operacionais. A GAN simples não teve amostra suficiente para várias comparações, pois não cobriu a maior parte do catálogo.

A auditoria de gênero registra uma regra metodológica: gênero não é utilizado como parâmetro no cálculo sintético da renda. Eventuais diferenças amostrais por gênero são diagnósticas e não representam uma regra implementada.

Os quality gates específicos do vocabulário 2 consideram bloqueantes a quantidade final de linhas, o schema final, categorias canônicas, ausência de categorias legadas no resultado final, compatibilidade estrutural, renda dentro dos limites e normalização Unicode `NFC`. Métricas como cobertura ocupacional, entropia, distância da distribuição e coerência bruta permanecem diagnósticas nesta primeira versão, pois ainda não há base empírica suficiente para limiares definitivos.

Status observados:

- `programmatic`: duas execuções `approved` e uma `rejected`;
- `ctgan`: duas execuções `approved` e uma `rejected`;
- `simple_gan`: duas execuções `quarantined` e uma `rejected`.

As rejeições de `programmatic`, `ctgan` e `simple_gan` na seed `43` foram causadas por quality gates obrigatórios (`invalid_rows_max` e `duplicated_identifier_max`) com uma linha inválida ou identificador duplicado. Não houve falhas técnicas no benchmark principal; `failures.json` permaneceu vazio.

### Correção da unicidade entre batches

A execução original da seed `43` revelou um defeito na garantia de unicidade entre lotes. O problema estava na etapa compartilhada de pós-processamento e seleção, não nos sintetizadores individualmente.

O diagnóstico isolado foi executado com `configs/benchmark-quality-vocab-v2-seed43-debug.yaml`, usando apenas `programmatic`, seed `43`, 20.000 registros de treinamento, 5.000 registros de holdout, 25.000 registros de calibração total, 20.000 registros sintéticos solicitados e `batch_size: 4096`.

Execução diagnóstica original:

- `benchmark_id`: `quality-vocab-v2-seed43-debug-20260728T161102Z-0efe1fcc`;
- `run_id`: `20260728T161111Z-66910641`;
- status: `rejected`;
- `invalid_rows`: `1`;
- `duplicated_identifier`: `1`;
- motivo estrutural: `CPF_duplicado`;
- artefato diagnóstico: `artifacts/runs/20260728T161111Z-66910641/quarantine/seed43_duplicate_diagnostic.json`.

O identificador afetado foi `CPF`. O valor foi registrado no diagnóstico apenas em forma mascarada e por SHA-256:

- valor mascarado: `***.***.***-04`;
- SHA-256: `e0944332c9e751272da37d171622adc172c2d5974b9cf5337bb3ff2e824924ab`;
- ocorrências: índice global `7764`, batch `1`, posição `3668`; índice global `9991`, batch `2`, posição `1799`;
- tipo de colisão: entre batches diferentes, não dentro do mesmo batch.

A comparação das máscaras confirmou a causa raiz:

| Métrica | Valor |
| --- | ---: |
| Máscara válida por batch | 20.480 |
| Máscara concatenada por batch | 20.480 |
| Máscara válida global | 20.479 |
| Selecionados | 20.000 |
| Índice válido por batch e inválido globalmente | `9991` |

Antes da correção, `finalizar_perfis_sinteticos` criava conjuntos locais de identificadores a cada chamada. Como `generate_profiles` chamava o pós-processamento uma vez por batch, a unicidade de `CPF`, `CNH`, `RG`, `Titulo_Eleitor` e `Telefone` era garantida apenas dentro do lote. Além disso, a seleção final usava a máscara válida por batch, embora a validação global já detectasse a colisão.

A correção aplicada passou a:

- criar um estado de identificadores uma única vez por geração;
- passar esse estado compartilhado para todos os batches de `finalizar_perfis_sinteticos`;
- preservar o comportamento antigo quando `finalizar_perfis_sinteticos` é chamada sem estado externo;
- validar a presença das cinco chaves esperadas no estado;
- selecionar registros com base em `full_validation.valid_mask`, isto é, a máscara da validação global;
- continuar gerando batches até obter a quantidade solicitada de registros globalmente válidos ou falhar claramente ao esgotar `max_batches`;
- registrar contadores separados para aceitação por batch e por regras globais.

Contadores adicionados ao accounting da geração:

- `accepted_by_batch_rules`;
- `accepted_by_global_rules`;
- `rejected_by_batch_rules`;
- `rejected_by_global_rules`;
- `batch_acceptance_rate`;
- `global_acceptance_rate`;
- `per_batch_valid_mask_count`;
- `concatenated_valid_mask_count`;
- `global_valid_mask_count`;
- `global_mask_disagreeing_count`;
- `global_mask_disagreeing_indices`;
- `cross_batch_identifier_duplicates`.

Após a correção, o diagnóstico programático foi repetido somente com a seed `43`:

- `benchmark_id`: `quality-vocab-v2-seed43-debug-20260728T161746Z-8ba0ba62`;
- `run_id`: `20260728T161750Z-5bbf24b9`;
- status: `approved`;
- `selected`: `20.000`;
- `invalid_rows`: `0`;
- `duplicated_identifier`: `0`;
- `cross_batch_identifier_duplicates`: `0`;
- `global_mask_disagreeing_count`: `0`.

Depois, a regressão restrita à seed `43` foi executada com `configs/benchmark-quality-vocab-v2-seed43-regression.yaml`, mantendo os três modelos e os hiperparâmetros do benchmark principal:

| Modelo | Status após correção | Linhas válidas | Identificador duplicado | Observação |
| --- | --- | ---: | ---: | --- |
| `programmatic` | `approved` | 20.000 | 0 | Defeito compartilhado removido |
| `ctgan` | `approved` | 20.000 | 0 | Defeito compartilhado removido |
| `simple_gan` | `quarantined` | 20.000 | 0 | Quarentena mantida por TVD e correlação, não por identificadores |

As conclusões anteriores sobre cobertura do vocabulário, categorias raras, diversidade e comparação `raw` versus `final` permanecem válidas como registro da execução original. A interpretação dos quality gates da seed `43` foi refinada: a rejeição por identificador duplicado não deve ser atribuída aos modelos, mas ao caminho compartilhado de pós-processamento e seleção existente naquela execução.

Artefatos específicos gerados em `artifacts/benchmarks/quality-vocab-v2-20260728T131154Z-364a35cb/`:

- `vocabulary_v2_metrics.csv` e `vocabulary_v2_metrics.parquet`;
- `occupation_coverage.csv` e `occupation_coverage.parquet`;
- `occupation_distribution.csv` e `occupation_distribution.parquet`;
- `occupation_income_summary.csv` e `occupation_income_summary.parquet`;
- `invalid_education_occupation.csv` e `invalid_education_occupation.parquet`;
- `invalid_age_occupation.csv` e `invalid_age_occupation.parquet`;
- `rare_occupation_coverage.csv` e `rare_occupation_coverage.parquet`;
- `raw_vs_final_summary.json`.

Interpretação cautelosa:

- `ProgrammaticSynthesizer`: permanece como opção padrão para a interface. O resultado tem vantagem estrutural porque a referência sintética é gerada pelo mesmo conjunto de regras.
- `SimpleTabularGAN`: permanece como baseline acadêmico experimental. O benchmark mostrou baixa cobertura do catálogo e sinais de concentração ocupacional.
- `CTGANSynthesizer`: foi aprovada para treinamento de um artefato candidato maior, pois cobriu as 37 ocupações e preservou parte relevante das dependências, embora ainda dependa do pós-processamento para validade final e tenha reduzido contrastes de renda por ocupação.

Os resultados anteriores não foram usados como substitutos desta avaliação, pois o vocabulário categórico e o catálogo de ocupações foram ampliados. A análise distingue a capacidade aprendida pelos modelos na saída bruta das correções aplicadas pelo pipeline no resultado final.

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
## Benchmark de realismo de renda e memorização

Os benchmarks `income_realism` avaliam diversidade, correspondência exata e plausibilidade condicional da renda sem executar testes de capacidade. As configurações principais são:

- `configs/benchmark-income-realism-baseline.yaml`, com `income_model_version: 1`;
- `configs/benchmark-income-realism-v2-smoke.yaml`, para validação técnica pequena;
- `configs/benchmark-income-realism-v2.yaml`, com `income_model_version: 2`.

Os resultados antigos permanecem históricos. A comparação entre baseline e v2 deve considerar frequência de caudas, p95, p99, duplicidade de combinações-base, match exato com treino e holdout, custo computacional e estabilidade entre seeds.

Nenhuma recomendação de artefato neural deve ser feita por uma única média. Gates obrigatórios continuam prevalecendo sobre scores agregados.
