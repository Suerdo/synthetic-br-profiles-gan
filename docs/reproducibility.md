# Reprodutibilidade

A seed e centralizada no pipeline e propagada para:

- `random`;
- NumPy;
- TensorFlow, quando `simple_gan` e usado;
- PyTorch/CTGAN, quando `ctgan` e usado;
- Faker;
- split treino/holdout;
- baseline programatico.

`PYTHONHASHSEED` e registrado no manifesto. Quando ele e alterado depois que o interpretador Python ja iniciou, o manifesto inclui a limitacao.

Datas de nascimento dependem de `reference_date`. Para reproduzir exatamente uma execucao, mantenha a mesma configuracao:

```bash
python -m synthetic_br_profiles_gan pipeline \
  --model programmatic \
  --config configs/pipeline.yaml
```

Treinos neurais podem variar por CPU/GPU, drivers, versoes de bibliotecas e operacoes nao deterministicas do backend. Por isso, testes padrao usam datasets pequenos e nao exigem igualdade bit a bit para TensorFlow ou CTGAN.
