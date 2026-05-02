# Reprodutibilidade

O pipeline usa seed para reduzir variações entre execuções.

## Seed

O script fixa:

- `PYTHONHASHSEED`;
- `random.seed`;
- `numpy.random.seed`;
- `tf.random.set_seed`;
- seed do `Faker`.

Exemplo:

```bash
python scripts/run_pipeline.py --n 1000 --seed 41 --output data/outputs
```

## Data de referência

Datas de nascimento dependem de uma data de referência. Para reproduzir exatamente esse aspecto, informe:

```bash
python scripts/run_pipeline.py --n 1000 --seed 41 --reference-date 2026-05-02 --output data/outputs
```

Sem esse parâmetro, o pipeline usa a data da execução.

## Variações esperadas

Mesmo com seed, treinamentos neurais podem variar por:

- versão do TensorFlow/Keras;
- CPU, GPU e drivers;
- backend numérico;
- paralelismo interno;
- versão do Python e bibliotecas.

Por isso, o relatório JSON registra parâmetros e métricas relevantes para rastreabilidade, mas não deve ser tratado como garantia absoluta de bitwise reproducibility.

