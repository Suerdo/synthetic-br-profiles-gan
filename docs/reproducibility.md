# Reprodutibilidade

A seed é centralizada no pipeline e propagada para:

- `random`;
- NumPy;
- TensorFlow, quando `simple_gan` é usado;
- PyTorch/CTGAN, quando `ctgan` é usado;
- Faker;
- split treino/holdout;
- baseline programático.

`PYTHONHASHSEED` é registrado no manifesto. Quando ele é alterado depois que o interpretador Python já iniciou, o manifesto inclui a limitação.

Datas de nascimento dependem de `reference_date`. Para reproduzir exatamente uma execução, mantenha a mesma configuração:

```bash
python -m synthetic_br_profiles_gan pipeline \
  --model programmatic \
  --config configs/pipeline.yaml
```

Treinos neurais podem variar por CPU/GPU, drivers, versões de bibliotecas e operações não determinísticas do backend. Por isso, testes padrão usam datasets pequenos e não exigem igualdade bit a bit para TensorFlow ou CTGAN.
