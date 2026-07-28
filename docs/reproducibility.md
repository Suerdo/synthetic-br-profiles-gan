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

## Fluxo separado `train` e `generate`

O comando `train` grava `training_manifest.json` com seed, configuração resolvida, tamanhos de treino e holdout, ambiente e versões de bibliotecas. Esse manifesto é usado pelo carregador de modelos para validar o artefato antes da geração.

O comando `generate --seed` controla:

- amostragem do modelo, quando suportada pelo sintetizador;
- Faker;
- geração dos identificadores;
- geração da data de nascimento;
- pós-processamento contextual.

Para o `ProgrammaticSynthesizer`, duas gerações com o mesmo artefato salvo, mesma seed, mesma quantidade de linhas e mesma versão do projeto devem produzir o mesmo arquivo. Para `SimpleTabularGAN` e `CTGANSynthesizer`, a geração tenta propagar a seed para NumPy, TensorFlow ou PyTorch, mas bibliotecas neurais podem não garantir determinismo bit a bit em todos os ambientes.

O manifesto da geração registra essa limitação no campo `reproducibility`.

## Seleção de colunas

A seleção por `--columns` ou `--preset` ocorre depois da geração interna completa e depois da validação estrutural das 18 colunas finais. Portanto, ela não muda o contrato dos sintetizadores nem elimina dependências necessárias durante o pós-processamento.

Para reproduzir exatamente um arquivo exportado, mantenha também a mesma seleção de colunas, a mesma ordem e o mesmo preset. Duas execuções com a mesma seed e o mesmo modelo podem gerar internamente os mesmos perfis, mas arquivos diferentes quando exportam subconjuntos distintos.

O manifesto registra `requested_columns`, `exported_columns`, `internally_generated_columns`, `column_selection_mode`, `column_preset` e `internal_dependencies` para facilitar a auditoria posterior.

## Carregamento de modelos

Artefatos de `SimpleTabularGAN` e `CTGANSynthesizer` podem usar `pickle` ou formatos equivalentes. Para manter a reprodutibilidade e reduzir riscos operacionais, carregue apenas diretórios de modelos produzidos ou previamente aprovados pela própria aplicação. Não há suporte, nesta fase, para upload arbitrário de modelos por usuários.
