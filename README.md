# Geracao de Perfis Sinteticos Brasileiros

Pipeline experimental para gerar e avaliar perfis sinteticos brasileiros sem usar dados pessoais reais, sem consultar bases oficiais e sem validar documentos contra pessoas existentes.

O projeto agora compara tres estrategias:

- `programmatic`: baseline puramente programatico, baseado em regras probabilisticas controladas.
- `simple_gan`: baseline neural preservado do projeto original. Ele e uma GAN tabular densa simples em Keras, nao uma CTGAN.
- `ctgan`: CTGAN real usando a biblioteca standalone `ctgan>=0.12.1,<0.13`.

## Arquitetura

O pacote separa as responsabilidades principais:

- `calibration.py`: cria base de calibracao sintetica com dependencias semanticas e split treino/holdout.
- `metadata.py`: schema, tipos, dominios, categorias e dependencias estruturais.
- `generators/`: contexto de perfil, nomes, datas, telefone e documentos ficticios.
- `models/`: interface comum `TabularSynthesizer`, baseline programatico, `SimpleTabularGAN` e `CTGANSynthesizer`.
- `validators/`: validacao estrutural centralizada.
- `evaluation/`: metricas estatisticas, relacionais, diversidade, privacidade e quality gates.
- `pipeline.py`: orquestracao reutilizada pela CLI e notebook.
- `artifacts.py` e `manifest.py`: versionamento, hashes e manifestos de execucao.

## Instalacao

Instalacao principal, sem backends neurais pesados:

```bash
pip install -e .
```

Backends opcionais:

```bash
pip install -e ".[simple-gan]"
pip install -e ".[ctgan]"
pip install -e ".[all-models]"
```

## Calibracao

A base de calibracao e sintetica e controlada. Ela inclui:

- idade, genero, regiao, estado, municipio, DDD;
- escolaridade, estado civil, ocupacao, renda e dependentes.

As relacoes sao probabilisticas e reproduziveis por seed:

- estado pertence a regiao;
- municipio e DDD pertencem ao estado;
- escolaridade depende de idade;
- ocupacao depende de idade e escolaridade;
- renda usa distribuicao assimetrica e depende de idade, escolaridade, ocupacao e regiao;
- estado civil e dependentes dependem de idade e contexto familiar.

Criar calibracao:

```bash
python -m synthetic_br_profiles_gan create-calibration \
  --config configs/calibration.yaml \
  --output artifacts/calibration/demo
```

## Treino

Treinar a GAN densa simples:

```bash
python -m synthetic_br_profiles_gan train \
  --model simple_gan \
  --config configs/simple_gan.yaml \
  --calibration artifacts/calibration/demo/train.parquet
```

Treinar CTGAN real:

```bash
python -m synthetic_br_profiles_gan train \
  --model ctgan \
  --config configs/ctgan.yaml \
  --calibration artifacts/calibration/demo/train.parquet
```

Na CTGAN, colunas categoricas e discretas sao declaradas explicitamente, incluindo `DDD`. Categorias nao sao tratadas como numeros continuos arredondados depois.
Se um backend opcional nao estiver instalado, a CLI retorna erro controlado com o comando de instalacao esperado, por exemplo `pip install -e ".[ctgan]"`.

## Geracao, Validacao e Avaliacao

Gerar a partir de um modelo salvo:

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

Avaliar contra uma referencia:

```bash
python -m synthetic_br_profiles_gan evaluate \
  --reference artifacts/runs/<run_id>/approved/holdout.parquet \
  --synthetic artifacts/runs/<run_id>/approved/dataset.parquet
```

Executar tudo:

```bash
python -m synthetic_br_profiles_gan pipeline \
  --model programmatic \
  --config configs/pipeline.yaml
```

Use `--require-approved` quando o comando deve retornar codigo diferente de zero se os quality gates nao aprovarem o resultado.

## Metricas

O relatorio compara sintetico contra treino e holdout separadamente.

Metricas numericas incluem media, mediana, desvio-padrao, min/max, quantis, distancia de Wasserstein absoluta, distancia de Wasserstein normalizada pelo IQR da referencia com fallback para desvio-padrao, KS e diferencas absolutas/relativas.

Metricas categoricas incluem frequencias, diferencas de proporcao, categorias ausentes/inesperadas e distancia de variacao total.

Metricas relacionais incluem correlacoes Pearson/Spearman, diferencas entre matrizes, crosstabs de pares relevantes e renda por faixa etaria, escolaridade, regiao e ocupacao.

Indicadores de diversidade e privacidade incluem duplicidade, match exato com treino/holdout, combinacoes unicas, cobertura de categorias, Distance to Closest Record e Nearest Neighbor Distance Ratio. Eles nao provam anonimizacao.

## Quality Gates

Os gates configuraveis ficam em `configs/quality_gates.yaml` e no bloco `quality_gates` de `configs/pipeline.yaml`.

Modos de avaliacao:

- `smoke`: verifica caminhos tecnicos com amostras pequenas. Falhas de tamanho minimo deixam o resultado em quarentena, mas nao sao evidencia estatistica.
- `experimental`: modo padrao para comparacoes exploratorias. Mantem gates obrigatorios e sinaliza gates informativos como quarentena.
- `approval`: exige amostra minima e rejeita metricas obrigatorias ausentes, invalidas ou `NaN`.

Gates obrigatorios incluem linhas invalidas, identificadores duplicados, campos obrigatorios nulos e taxa de match exato com treino. Gates informativos padrao incluem distancia de variacao total categorica e diferenca de correlacao. Metricas obrigatorias ausentes ou nao finitas nao aprovam automaticamente a execucao.

Estados possiveis:

- `approved`: gates obrigatorios e opcionais passaram.
- `quarantined`: gates obrigatorios passaram, mas algum gate opcional falhou.
- `rejected`: gate obrigatorio falhou.

Quando o status nao e aprovado, o dataset e os relatorios finais vao para `quarantine/` em vez de `approved/`. Com `--require-approved`, a CLI retorna codigo diferente de zero para `quarantined` ou `rejected`.

## Artefatos

Cada execucao usa `run_id` com timestamp UTC:

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

Dentro de `approved/` ou `quarantine/` sao salvos:

- `dataset.parquet` como formato principal;
- `dataset.xlsx` quando habilitado;
- `validation.json`;
- `evaluation.json`;
- `quality_gates.json`;
- `generation.json`;
- `manifest.json`;
- `metadata.json`;
- `train.parquet` e `holdout.parquet`.

O manifesto registra run id, timestamp UTC, modelo, seed, quantidades, status, versoes de bibliotecas, plataforma, CPU/GPU quando o backend esta carregado, duracao, hash da configuracao, hashes dos artefatos e commit Git quando disponivel.

## Reprodutibilidade

A seed e centralizada. O pipeline controla `random`, NumPy e, quando o modelo exige, TensorFlow ou PyTorch/CTGAN. Variaveis como `PYTHONHASHSEED` sao registradas, mas a documentacao do manifesto avisa quando foram alteradas depois do inicio do interpretador.

Operacoes neurais podem variar entre CPU, GPU, drivers e versoes de backend. Testes padrao evitam exigir igualdade bit a bit de TensorFlow ou CTGAN.

## Notebook

O notebook em `notebooks/` importa o pacote e demonstra execucao, amostra, validacao, metricas e comparacao de modelos. Ele nao contem mais uma implementacao paralela do pipeline.

## Testes

```bash
python -m unittest discover -s tests
```

Os testes cobrem schema, calibracao, relacoes estado/regiao/municipio/DDD, data de nascimento, documentos, validadores, metricas, privacidade, quality gates, run ID, CLI, baseline programatico, reprodutibilidade e pipeline pequeno.

## Limitacoes

- Documentos matematicamente validos nao sao consultados em bases reais.
- O projeto nao garante que um numero valido nunca seja atribuido a uma pessoa real.
- Os dados devem permanecer identificados como sinteticos.
- Os dados nao devem ser usados para interagir com servicos reais.
- Metricas de privacidade sao indicadores de risco, nao prova automatica de anonimizacao.
- A calibracao programatica nao representa perfeitamente a populacao brasileira.
- Qualidade estatistica nao significa veracidade individual.
- A GAN antiga e uma GAN tabular densa simples; CTGAN so existe no modelo `ctgan`.

## Uso Responsavel

Este projeto e destinado a pesquisa, testes, homologacao e experimentacao. Nao use os dados para fraude, falsificacao documental, criacao de contas indevidas, engenharia social, tomada de decisao sobre pessoas ou qualquer interacao com servicos reais.
