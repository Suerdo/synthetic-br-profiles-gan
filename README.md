# Geração de perfis sintéticos brasileiros

Pipeline experimental para gerar e avaliar perfis sintéticos brasileiros sem usar dados pessoais reais, sem consultar bases oficiais e sem validar documentos contra pessoas existentes.

O projeto compara três estratégias de geração: um baseline programático, uma GAN tabular densa simples preservada do projeto original e uma CTGAN baseada na biblioteca `ctgan`.

## Estratégias de geração

| Modelo | Descrição | Dependência opcional |
| --- | --- | --- |
| `programmatic` | Baseline programático baseado em regras probabilísticas controladas | Não |
| `simple_gan` | GAN tabular densa simples em Keras | `simple-gan` |
| `ctgan` | CTGAN da biblioteca `ctgan>=0.12.1,<0.13` | `ctgan` |

O modelo `simple_gan` não é uma CTGAN. Ele é mantido como baseline neural legado para comparação experimental.

## Arquitetura

O pacote separa as responsabilidades principais:

- `calibration.py`: cria a base de calibração sintética com dependências semânticas e divisão entre conjunto de treinamento e conjunto de holdout, isto é, o subconjunto reservado exclusivamente para avaliação.
- `metadata.py`: define schema, tipos, domínios, categorias e dependências estruturais.
- `column_catalog.py`: descreve as 18 colunas finais, seus grupos, dependências internas e presets de exportação.
- `generators/`: contém o contexto de perfil e os geradores de nomes, datas, telefone e documentos fictícios.
- `models/`: contém a interface comum `TabularSynthesizer`, o baseline programático, `SimpleTabularGAN` e `CTGANSynthesizer`.
- `models/registry.py`: carrega modelos salvos a partir de `training_manifest.json` e valida arquivos obrigatórios.
- `services/`: contém os serviços reutilizáveis de treinamento e geração, independentes da CLI.
- `validators/`: centraliza a validação estrutural.
- `evaluation/`: calcula métricas estatísticas, relacionais, de diversidade, de privacidade e quality gates, ou critérios automáticos de aprovação e rejeição.
- `pipeline.py`: implementa a orquestração reutilizada pela CLI e pelo notebook.
- `artifacts.py` e `manifest.py`: cuidam do versionamento, dos hashes e dos manifestos de execução.

## Instalação

Instalação principal, sem backends neurais pesados:

```bash
pip install -e .
```

Backends opcionais:

```bash
pip install -e ".[simple-gan]"
pip install -e ".[ctgan]"
pip install -e ".[all-models]"
```

Se um backend opcional não estiver instalado, a CLI retorna um erro controlado com o comando de instalação esperado, por exemplo `pip install -e ".[ctgan]"`.

## Calibração

A base de calibração é sintética e controlada. Ela inclui:

- idade, gênero, região, estado, município e DDD;
- escolaridade, estado civil, ocupação, renda e dependentes.

As relações são probabilísticas e reproduzíveis por seed:

- estado pertence a uma região;
- município e DDD pertencem ao estado;
- escolaridade depende da idade;
- ocupação depende da idade e da escolaridade;
- renda usa uma distribuição assimétrica e depende de idade, escolaridade, ocupação e região;
- estado civil e dependentes dependem da idade e do contexto familiar.

Criar a base de calibração:

```bash
python -m synthetic_br_profiles_gan create-calibration \
  --config configs/calibration.yaml \
  --output artifacts/calibration/demo
```

## Treinamento

A CLI permite treinar e salvar um sintetizador uma única vez. O artefato salvo pode ser carregado posteriormente por `generate`, sem repetir o treinamento.

Treinar o baseline programático como artefato reutilizável:

```bash
python -m synthetic_br_profiles_gan train \
  --model programmatic \
  --config configs/train-programmatic.yaml \
  --output artifacts/models/programmatic-default
```

Treinar a GAN tabular densa simples em modo smoke:

```bash
python -m synthetic_br_profiles_gan train \
  --model simple_gan \
  --config configs/train-simple-gan-smoke.yaml \
  --output artifacts/models/simple-gan-default
```

Treinar a CTGAN em modo smoke:

```bash
python -m synthetic_br_profiles_gan train \
  --model ctgan \
  --config configs/train-ctgan-smoke.yaml \
  --output artifacts/models/ctgan-default
```

Na CTGAN, colunas categóricas e discretas são declaradas explicitamente, incluindo `DDD`. Categorias não são tratadas como números contínuos para posterior arredondamento.

O comando `train --model programmatic` não simula treinamento neural. Ele grava `config.json`, `metadata.json`, `training_config.yaml` e `training_manifest.json`, com `training_required: false`.

## Geração, validação e avaliação

Gerar dados sintéticos a partir de um modelo salvo:

```bash
python -m synthetic_br_profiles_gan generate \
  --model-path artifacts/models/ctgan-default \
  --rows 10000 \
  --output artifacts/generations/ctgan-10000.csv \
  --format csv \
  --seed 41
```

Gerar diretamente pelo baseline programático, sem artefato salvo:

```bash
python -m synthetic_br_profiles_gan generate \
  --model programmatic \
  --rows 10000 \
  --output artifacts/generations/programmatic-10000.csv \
  --format csv \
  --seed 41
```

O comando `generate` produz as 18 colunas finais, aplica pós-processamento contextual, valida estruturalmente o dataset e só exporta quando a validação bloqueante passa. Os formatos aceitos são `csv`, `json` e `parquet`. O CSV usa UTF-8, sem índice, com separador `;`, escolhido por compatibilidade prática com ferramentas brasileiras.

Também é possível exportar apenas um subconjunto das colunas finais. A seleção ocorre somente depois da geração interna das 18 colunas e depois da validação estrutural completa; os sintetizadores continuam usando o mesmo contrato interno de 11 colunas-base.

Seleção explícita, preservando a ordem informada:

```bash
python -m synthetic_br_profiles_gan generate \
  --model programmatic \
  --rows 1000 \
  --columns Nome Idade Estado CPF \
  --output artifacts/generations/perfis.csv \
  --format csv
```

A CLI também aceita uma lista separada por vírgulas em `--columns`, por exemplo `--columns Nome,Idade,Estado,CPF`. Colunas desconhecidas, repetidas ou com capitalização incorreta são rejeitadas. Não há normalização silenciosa de nomes como `Uf` para `Estado`.

Presets disponíveis:

- `completo`: todas as 18 colunas;
- `demografico`: perfil demográfico e socioeconômico sem identificadores documentais;
- `contato`: nome, localização e telefone;
- `documentos`: nome, data de nascimento e identificadores sintéticos;
- `minimo`: `Nome`, `Idade`, `Estado` e `CPF`.

Exemplo com preset:

```bash
python -m synthetic_br_profiles_gan generate \
  --model programmatic \
  --rows 1000 \
  --preset demografico \
  --output artifacts/generations/demografico.parquet \
  --format parquet
```

`--columns` e `--preset` não podem ser usados simultaneamente. Dependências internas continuam sendo usadas para gerar e validar os dados, mas não são adicionadas automaticamente ao arquivo exportado. Por exemplo, `Telefone` depende internamente de `Estado` e `DDD`; se o usuário solicitar apenas `Nome Telefone CPF`, o arquivo conterá somente essas três colunas.

Modelos salvos podem conter `pickle` ou formatos equivalentes. Por segurança, carregue apenas artefatos produzidos ou previamente aprovados pela própria aplicação. Esta fase não implementa upload arbitrário de modelos.

Validar um dataset final:

```bash
python -m synthetic_br_profiles_gan validate \
  --input artifacts/runs/<run_id>/approved/dataset.parquet \
  --config configs/pipeline.yaml
```

Avaliar contra uma referência:

```bash
python -m synthetic_br_profiles_gan evaluate \
  --reference artifacts/runs/<run_id>/approved/holdout.parquet \
  --synthetic artifacts/runs/<run_id>/approved/dataset.parquet
```

Executar o pipeline completo:

```bash
python -m synthetic_br_profiles_gan pipeline \
  --model programmatic \
  --config configs/pipeline.yaml
```

Use `--require-approved` quando o comando deve retornar código diferente de zero se os quality gates não aprovarem o resultado.

## Métricas

O relatório compara os dados sintéticos separadamente contra o conjunto de treinamento e contra o conjunto de holdout.

Para colunas numéricas, as métricas incluem média, mediana, desvio-padrão, mínimo, máximo, quantis, distância de Wasserstein absoluta, distância de Wasserstein normalizada pelo IQR da referência com fallback para desvio-padrão, KS e diferenças absolutas e relativas.

Para colunas categóricas, as métricas incluem frequências, diferenças de proporção, categorias ausentes, categorias inesperadas e distância de variação total.

As métricas relacionais incluem correlações Pearson/Spearman, diferenças entre matrizes, crosstabs de pares relevantes e renda por faixa etária, escolaridade, região e ocupação.

Os indicadores de diversidade e privacidade incluem duplicidade, match exato com o conjunto de treinamento e com o conjunto de holdout, combinações únicas, cobertura de categorias, Distance to Closest Record e Nearest Neighbor Distance Ratio. Esses indicadores não provam anonimização.

## Benchmark experimental

O benchmark experimental compara `programmatic`, `simple_gan` e `ctgan` sobre a mesma base de calibração por seed. Para cada seed, o pipeline cria uma única calibração, divide em conjunto de treinamento e conjunto de holdout, executa os modelos configurados e consolida métricas individuais em artefatos de benchmark.

O benchmark reutiliza o pipeline existente. Ele não possui uma segunda implementação de treino, geração, validação, avaliação ou quality gates.

A configuração piloto fica em `configs/benchmark.yaml`. A matriz padrão é:

- modelos: `programmatic`, `simple_gan` e `ctgan`;
- seeds: `11`, `22` e `33`;
- base de calibração: `5000` registros;
- dados sintéticos por execução: `2000` registros;
- conjunto de holdout: `20%`;
- modo de avaliação: `experimental`.

Executar o benchmark piloto:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark.yaml
```

Também é possível sobrescrever modelos e seeds pela CLI:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark.yaml \
  --models programmatic simple_gan ctgan \
  --seeds 11 22 33
```

Para uma validação rápida apenas com o baseline programático:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-programmatic.yaml
```

### Sensibilidade ao tamanho do treinamento

O benchmark também possui configurações para avaliar sensibilidade e escalabilidade com tamanhos exatos de conjunto de treinamento. Nesse modo, o parâmetro principal é `train_sizes`, e não `calibration_rows`. A base de calibração total é calculada para que, depois do split, o conjunto de treinamento e o conjunto de holdout tenham tamanhos exatos.

Com `holdout_fraction: 0.20`, os tamanhos usados são:

| Treino | Holdout | Calibração total |
| ---: | ---: | ---: |
| 1.000 | 250 | 1.250 |
| 5.000 | 1.250 | 6.250 |
| 20.000 | 5.000 | 25.000 |

A validação técnica rápida fica em `configs/benchmark-scaling-smoke.yaml`:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-scaling-smoke.yaml
```

O experimento principal fica em `configs/benchmark-scaling.yaml`:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-scaling.yaml
```

Também é possível sobrescrever os tamanhos pela CLI:

```bash
python -m synthetic_br_profiles_gan benchmark \
  --config configs/benchmark-scaling.yaml \
  --train-sizes 1000 5000 20000
```

Para cada combinação de seed e `train_size`, o benchmark cria uma única base de calibração, um único conjunto de treinamento e um único conjunto de holdout. Esses mesmos dados são reutilizados por `programmatic`, `simple_gan` e `ctgan`. Os 5.000 dados sintéticos do experimento principal permanecem constantes entre os tamanhos, para tornar as métricas comparáveis.

O modo de escalabilidade registra tempo de treinamento, tempo de geração, tempo total, pico aproximado de memória residente, tamanho do modelo salvo, tamanho dos artefatos, contadores de updates da GAN tabular densa simples, batches inferidos da CTGAN e ganhos marginais entre `1000 → 5000`, `5000 → 20000` e `1000 → 20000`.

`execution.warmup_backends` carrega minimamente backends opcionais antes das execuções e registra `backend_warmup_seconds` fora do tempo de treinamento. `execution.rotate_model_order_by_seed` alterna a ordem dos modelos por seed para reduzir viés de cache. Essas opções não alteram a ordenação final dos arquivos consolidados.

Os limites em `resource_limits` são operacionais e opcionais. Valores `null` significam monitoramento sem interrupção automática. Quando um limite configurado é excedido, a execução é marcada como `resource_limited` para análise de escalabilidade, sem transformar automaticamente uma falha de quality gate em falha técnica.

Os artefatos são salvos em `artifacts/benchmarks/<benchmark_id>/`. Eles incluem `benchmark_manifest.json`, `environment.json`, `resource_limits.json`, `runs.json`, `results.parquet`, `results.csv`, `run_summary.parquet`, `run_summary.csv`, `summary.json`, `aggregate_by_model_and_size.json`, `marginal_gains.json`, `scalability_limits.json`, `failures.json` e referências para os `run_id` individuais em `artifacts/runs/`.

O benchmark-piloto é exploratório. Três seeds ainda não produzem evidência definitiva; menor distância estatística não implica maior privacidade; e nenhum modelo é necessariamente superior em todos os critérios. Os resultados dependem da base de calibração controlada.

O tamanho de 20.000 registros é o limite superior do experimento atual, não um limite máximo absoluto dos modelos. Quando uma execução conclui nesse tamanho, a interpretação correta é que o modelo foi executado com sucesso com até 20.000 registros neste ambiente.

## Quality gates

Os quality gates configuráveis ficam em `configs/quality_gates.yaml` e no bloco `quality_gates` de `configs/pipeline.yaml`.

Modos de avaliação:

- `smoke`: verifica caminhos técnicos com amostras pequenas. Falhas de tamanho mínimo deixam o resultado em quarentena, mas não são evidência estatística.
- `experimental`: modo padrão para comparações exploratórias. Mantém gates obrigatórios e sinaliza gates informativos como quarentena.
- `approval`: exige amostra mínima e rejeita métricas obrigatórias ausentes, inválidas ou `NaN`.

Gates obrigatórios incluem linhas inválidas, identificadores duplicados, campos obrigatórios nulos e taxa de match exato com o conjunto de treinamento. Gates informativos padrão incluem distância de variação total categórica e diferença de correlação. Métricas obrigatórias ausentes ou não finitas não aprovam automaticamente a execução.

| Estado | Significado |
| --- | --- |
| `approved` | Todos os critérios aplicáveis foram atendidos |
| `quarantined` | Critérios obrigatórios passaram, mas há alertas ou critérios informativos não atendidos |
| `rejected` | Pelo menos um critério obrigatório falhou |

Quando o status não é aprovado, o dataset e os relatórios finais vão para `quarantine/` em vez de `approved/`. Com `--require-approved`, a CLI retorna código diferente de zero para `quarantined` ou `rejected`.

## Artefatos

Cada execução usa `run_id` com timestamp UTC:

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
  generations/
    programmatic-10000.csv
    programmatic-10000.manifest.json
  runs/
    <run_id>/
      approved/
      quarantine/
      manifest.json
      config.yaml
```

Dentro de `approved/` ou `quarantine/` são salvos:

- `dataset.parquet` como formato principal;
- `dataset.xlsx` quando habilitado;
- `validation.json`;
- `evaluation.json`;
- `quality_gates.json`;
- `generation.json`;
- `manifest.json`;
- `metadata.json`;
- `train.parquet` e `holdout.parquet`.

O manifesto registra `run_id`, timestamp UTC, modelo, seed, quantidades, status, versões de bibliotecas, plataforma, CPU/GPU quando o backend está carregado, duração, hash da configuração, hashes dos artefatos e commit Git quando disponível.

Modelos treinados pelo comando `train` usam `training_manifest.json`, com `schema_version`, `artifact_type`, modelo, seed, indicação de `training_required`, tamanhos de treino, holdout e calibração, colunas de modelo, colunas finais, configuração resolvida, ambiente, tempos e tamanho do artefato.

Datasets gerados pelo comando `generate` recebem um manifesto próximo ao arquivo exportado, por exemplo `programmatic-10000.manifest.json`. Esse manifesto registra modelo, caminho do artefato de modelo quando houver, linhas, colunas exportadas, colunas geradas internamente, modo de seleção (`all`, `explicit` ou `preset`), preset usado quando aplicável, dependências internas, formato, seed, arquivo de saída, validação estrutural do schema completo, tempos, ambiente e aviso de governança.

## Reprodutibilidade

A seed é centralizada. O pipeline controla `random`, NumPy e, quando o modelo exige, TensorFlow ou PyTorch/CTGAN. Variáveis como `PYTHONHASHSEED` são registradas, mas a documentação do manifesto avisa quando foram alteradas depois do início do interpretador.

Operações neurais podem variar entre CPU, GPU, drivers e versões de backend. Os testes padrão evitam exigir igualdade bit a bit de TensorFlow ou CTGAN.

No fluxo separado, `generate --seed` controla a amostragem quando o sintetizador oferece suporte, além de Faker, identificadores, data de nascimento e pós-processamento. Duas gerações programáticas com o mesmo artefato, seed, quantidade e versão do projeto devem produzir o mesmo arquivo. Backends neurais podem ter limitações adicionais de determinismo.

A seleção de colunas altera apenas a projeção exportada. Com a mesma seed, quantidade e modelo, a geração interna completa permanece a mesma; arquivos com subconjuntos diferentes terão conteúdos e manifestos diferentes porque exportam colunas distintas.

## Notebook

O notebook em `notebooks/` importa o pacote e demonstra execução, amostra, validação, métricas e comparação de modelos. Ele não contém mais uma implementação paralela do pipeline.

## Testes

```bash
python -m unittest discover -s tests
```

Os testes cobrem schema, catálogo de colunas, presets, seleção parcial na exportação, base de calibração, relações entre estado, região, município e DDD, data de nascimento, documentos, validadores, métricas, privacidade, quality gates, `run_id`, CLI, baseline programático, serviços de treinamento e geração, carregamento de modelos, reprodutibilidade e pipeline pequeno.

## Limitações

- Documentos matematicamente válidos não são consultados em bases reais.
- O projeto não garante que um número válido nunca seja atribuído a uma pessoa real.
- Os dados devem permanecer identificados como dados sintéticos.
- Os dados não devem ser usados para interagir com serviços reais.
- Métricas de privacidade são indicadores de risco, não prova automática de anonimização.
- A base de calibração programática não representa perfeitamente a população brasileira.
- Qualidade estatística não significa veracidade individual.
- A GAN antiga é uma GAN tabular densa simples; CTGAN só existe no modelo `ctgan`.

## Uso responsável

Este projeto é destinado a pesquisa, testes, homologação e experimentação. Não use os dados para fraude, falsificação documental, criação de contas indevidas, engenharia social, tomada de decisão sobre pessoas ou qualquer interação com serviços reais.
