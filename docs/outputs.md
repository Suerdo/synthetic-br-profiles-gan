# Saídas Geradas

O script principal grava os artefatos em `data/outputs/`.

## Dataset

Arquivo padrão:

```text
data/outputs/dados_sinteticos_realistas.xlsx
```

Colunas geradas:

- `Nome`
- `Gênero`
- `Data_Nascimento`
- `CPF`
- `CNH`
- `RG`
- `Titulo_Eleitor`
- `Telefone`
- `Renda`

Esses campos são fictícios e destinados a ambientes controlados.

## Relatório de execução

Arquivo padrão:

```text
data/outputs/relatorio_execucao.json
```

O relatório inclui:

- tamanho-alvo da geração;
- seed;
- parâmetros da GAN;
- total de candidatos;
- taxa de aceitação;
- tempo de geração;
- throughput;
- contagem de rejeições;
- resumo univariado antes do pós-processamento;
- colisões de identificadores;
- métricas de validação final.

## Versionamento das saídas

`data/outputs/` é mantida no repositório com `.gitkeep`, mas os arquivos gerados localmente são ignorados pelo Git.

Se for necessário publicar um exemplo pequeno, use `data/samples/` e garanta que ele seja claramente identificado como sintético.

