# synthetic-br-profiles-gan

Este repositório, contêm um notebook reprodutível no Google Colab para geração de perfis sintéticos brasileiros usando uma GAN tabular (TensorFlow/Keras) e geradores determinísticos para identificadores com estrutura válida (ex.: CPF/RG/CNH). O pipeline registra métricas de execução (taxa de seleção/aceitação, vazão) e realiza validações pós-geração (formato e unicidade).

Este repositório é destinado a uso acadêmico, testes, homologação e experimentação controlada.
## O que este repositório oferece

- Notebook pronto para execução no Colab, que:
  - Treina uma GAN tabular em uma base de calibração
  - Gera candidatos sintéticos e seleciona um lote-alvo
  - Produz um conjunto final com campos estruturalmente válidos
  - Exporta um dataset `.xlsx` e um relatório `.json` de execução
- Métricas e validações para:
  - Validade de formato (CPF/RG/telefone)
  - Unicidade (ausência de colisões internas para identificadores)
  - Desempenho (tempo de geração e registros por segundo)

## Requisitos

O notebook foi projetado para execução no Google Colab (recomendado). Para execução local, utilize Python 3.12+ e os seguintes pacotes:

- tensorflow
- faker
- pandas
- numpy
- scikit-learn
- openpyxl

Um `requirements.txt` mínimo (opcional):

```

tensorflow
faker
pandas
numpy
scikit-learn
openpyxl

````

## Como executar (Google Colab)

1. Abra o notebook no Colab:
   - Se o notebook já estiver com link do Colab, use-o diretamente.
   - Ou abra pelo GitHub: Colab > Arquivo > Abrir notebook > GitHub e cole a URL deste repositório.

2. Ative GPU no runtime:
   - Ambiente de execução > Alterar tipo de ambiente de execução > Acelerador de hardware: GPU

3. Execute todas as células:
   - Ambiente de execução > Executar tudo

4. Arquivos gerados ao final (no diretório do Colab):
   - `dados_sinteticos_realistas.xlsx`
   - `relatorio_execucao.json`

## Saídas geradas

### 1) dados_sinteticos_realistas.xlsx

Dataset final voltado a testes/homologação e experimentos controlados. As colunas típicas incluem:

* Nome
* Gênero
* Data_Nascimento
* CPF
* CNH
* RG
* Titulo_Eleitor
* Telefone
* Renda

### 2) relatorio_execucao.json

Relatório de execução contendo métricas operacionais, por exemplo:

* total de candidatos gerados
* tamanho-alvo selecionado
* taxa de seleção/aceitação
* tempo total (segundos)
* vazão (registros/segundo)
* resumo univariado (média/mediana/desvio padrão)

## Validações e checagens

O notebook inclui checagens objetivas no dataset final, incluindo:

* Validade de formato:

  * CPF (regex)
  * RG (regex)
  * Telefone (regex)
* Unicidade (ausência de colisões internas):

  * CPF, CNH, RG, Título de eleitor e telefone

Essas métricas suportam a apresentação de resultados quantitativos (ex.: contagem de inválidos, duplicidades e taxa aproximada de conformidade para os critérios avaliados).

## Reprodutibilidade

* Sementes são fixadas para Python, NumPy e TensorFlow.
* Parâmetros relevantes são registrados e um relatório JSON é gerado.
* Em condições semelhantes (mesmas versões e ambiente), os resultados tendem a ser consistentes dentro da variação típica de treinamento neural.

## Nota ética e de uso

Este repositório não contém dados pessoais reais. As saídas são sintéticas e destinadas estritamente a pesquisa, testes e homologação. Não utilize os dados gerados como se representassem indivíduos reais, nem para decisões operacionais envolvendo pessoas.
