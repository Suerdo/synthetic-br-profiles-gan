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
- `generators/`: contém o contexto de perfil e os geradores de nomes, datas, telefone e documentos fictícios.
- `models/`: contém a interface comum `TabularSynthesizer`, o baseline programático, `SimpleTabularGAN` e `CTGANSynthesizer`.
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

Treinar a GAN tabular densa simples:

```bash
python -m synthetic_br_profiles_gan train \
  --model simple_gan \
  --config configs/simple_gan.yaml \
  --calibration artifacts/calibration/demo/train.parquet
```

Treinar a CTGAN:

```bash
python -m synthetic_br_profiles_gan train \
  --model ctgan \
  --config configs/ctgan.yaml \
  --calibration artifacts/calibration/demo/train.parquet
```

Na CTGAN, colunas categóricas e discretas são declaradas explicitamente, incluindo `DDD`. Categorias não são tratadas como números contínuos para posterior arredondamento.

## Geração, validação e avaliação

Gerar dados sintéticos a partir de um modelo salvo:

```bash
python -m synthetic_br_profiles_gan generate \
  --model programmatic \
  --rows 1000 \
  --config configs/pipeline.yaml
```

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

Os artefatos são salvos em `artifacts/benchmarks/<benchmark_id>/`. Eles incluem `benchmark_manifest.json`, `runs.json`, `results.parquet`, `results.csv`, `run_summary.parquet`, `run_summary.csv`, `summary.json`, `failures.json` e referências para os `run_id` individuais em `artifacts/runs/`.

O benchmark-piloto é exploratório. Três seeds ainda não produzem evidência definitiva; menor distância estatística não implica maior privacidade; e nenhum modelo é necessariamente superior em todos os critérios. Os resultados dependem da base de calibração controlada.

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

## Reprodutibilidade

A seed é centralizada. O pipeline controla `random`, NumPy e, quando o modelo exige, TensorFlow ou PyTorch/CTGAN. Variáveis como `PYTHONHASHSEED` são registradas, mas a documentação do manifesto avisa quando foram alteradas depois do início do interpretador.

Operações neurais podem variar entre CPU, GPU, drivers e versões de backend. Os testes padrão evitam exigir igualdade bit a bit de TensorFlow ou CTGAN.

## Notebook

O notebook em `notebooks/` importa o pacote e demonstra execução, amostra, validação, métricas e comparação de modelos. Ele não contém mais uma implementação paralela do pipeline.

## Testes

```bash
python -m unittest discover -s tests
```

Os testes cobrem schema, base de calibração, relações entre estado, região, município e DDD, data de nascimento, documentos, validadores, métricas, privacidade, quality gates, `run_id`, CLI, baseline programático, reprodutibilidade e pipeline pequeno.

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
