# Metodologia

O pipeline é experimental, modular e mensurável. Ele não usa dados pessoais reais e não consulta bases externas.

## 1. Calibração

A calibração é gerada por regras probabilísticas locais. As colunas de modelo são:

- `Idade`, `Genero`, `Regiao`, `Estado`, `Municipio`, `DDD`;
- `Escolaridade`, `Estado_Civil`, `Ocupacao`, `Renda`, `Dependentes`.

As dependências estruturais são explicitadas em `metadata.py` e testadas:

- `Estado` pertence a `Regiao`;
- `Municipio` e `DDD` pertencem a `Estado`;
- `Escolaridade` depende probabilisticamente de `Idade`;
- `Ocupacao` depende de `Escolaridade` e `Idade`;
- `Renda` usa distribuição assimétrica e depende de idade, escolaridade, ocupação e região;
- `Estado_Civil` e `Dependentes` dependem de idade e estado civil.

## 2. Split

A calibração é dividida em treino e holdout com seed configurável. Modelos treináveis usam somente treino. A avaliação compara os dados sintéticos contra treino e holdout separadamente.

## 3. Modelos

- `ProgrammaticSynthesizer`: baseline programático.
- `SimpleTabularGAN`: GAN tabular densa simples preservada do projeto original.
- `CTGANSynthesizer`: CTGAN real via biblioteca standalone `ctgan`.

## 4. Pós-processamento

As saídas do modelo viram `SyntheticProfileContext`. A partir desse contexto, são derivados nome, data de nascimento, telefone e identificadores fictícios. A data de nascimento é sorteada dentro do intervalo que produz exatamente a idade informada na data de referência.

## 5. Validação e avaliação

A seleção de linhas usa schema, tipos, domínios, regras semânticas, documentos matematicamente válidos e duplicidade. O discriminador da GAN simples não é usado como probabilidade calibrada nem como filtro principal.

As métricas estatísticas, relacionais, de diversidade, de privacidade e quality gates são calculadas em módulos independentes dos modelos. Distâncias de vizinhança, DCR, NNDR e matches exatos usam os atributos de modelo; identificadores derivados como CPF, RG, CNH, título, telefone, nome e data de nascimento ficam fora das métricas de proximidade.

Distâncias de Wasserstein são registradas em escala absoluta e também normalizadas pelo IQR da referência, com fallback para desvio-padrão quando o IQR é zero. Essa normalização facilita comparar variáveis em escalas diferentes sem remover a métrica absoluta.
