# Metodologia

O pipeline e experimental, modular e mensuravel. Ele nao usa dados pessoais reais e nao consulta bases externas.

## 1. Calibracao

A calibracao e gerada por regras probabilisticas locais. As colunas de modelo sao:

- `Idade`, `Genero`, `Regiao`, `Estado`, `Municipio`, `DDD`;
- `Escolaridade`, `Estado_Civil`, `Ocupacao`, `Renda`, `Dependentes`.

As dependencias estruturais sao explicitadas em `metadata.py` e testadas:

- `Estado` pertence a `Regiao`;
- `Municipio` e `DDD` pertencem a `Estado`;
- `Escolaridade` depende probabilisticamente de `Idade`;
- `Ocupacao` depende de `Escolaridade` e `Idade`;
- `Renda` usa distribuicao assimetrica e depende de idade, escolaridade, ocupacao e regiao;
- `Estado_Civil` e `Dependentes` dependem de idade e estado civil.

## 2. Split

A calibracao e dividida em treino e holdout com seed configuravel. Modelos treinaveis usam somente treino. A avaliacao compara o sintetico contra treino e holdout separadamente.

## 3. Modelos

- `ProgrammaticSynthesizer`: baseline programatico.
- `SimpleTabularGAN`: GAN tabular densa simples preservada do projeto original.
- `CTGANSynthesizer`: CTGAN real via biblioteca standalone `ctgan`.

## 4. Pos-processamento

As saidas do modelo viram `SyntheticProfileContext`. A partir desse contexto sao derivados nome, data de nascimento, telefone e identificadores ficticios. A data de nascimento e sorteada dentro do intervalo que produz exatamente a idade informada na data de referencia.

## 5. Validacao e avaliacao

A selecao de linhas usa schema, tipos, dominios, regras semanticas, documentos matematicamente validos e duplicidade. O discriminador da GAN simples nao e usado como probabilidade calibrada nem como filtro principal.

As metricas estatisticas, relacionais, diversidade, privacidade e quality gates sao calculadas em modulos independentes dos modelos. Distancias de vizinhanca, DCR, NNDR e matches exatos usam os atributos de modelo; identificadores derivados como CPF, RG, CNH, titulo, telefone, nome e data de nascimento ficam fora das metricas de proximidade.

Distancias de Wasserstein sao registradas em escala absoluta e tambem normalizadas pelo IQR da referencia, com fallback para desvio-padrao quando o IQR e zero. Essa normalizacao facilita comparar variaveis em escalas diferentes sem remover a metrica absoluta.
