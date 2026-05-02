# Metodologia

Este documento descreve o fluxo técnico do pipeline de geração de dados pessoais sintéticos brasileiros.

## 1. Base de calibração

O projeto não usa dados pessoais reais. A base de calibração é gerada de forma sintética com três variáveis numéricas:

- `Idade`: faixa adulta entre 18 e 65 anos.
- `Sexo`: variável binária usada apenas na etapa tabular.
- `Renda`: renda mensal simulada entre R$ 1.200 e R$ 25.000.

Essa base serve como referência simples para o treinamento da GAN tabular.

## 2. Pré-processamento

As variáveis numéricas são normalizadas coluna a coluna com `MinMaxScaler`, permitindo que a saída `sigmoid` do gerador opere no intervalo esperado.

O objeto `DataPreprocessor` mantém os scalers ajustados e permite reverter amostras geradas para o espaço original.

## 3. GAN tabular

A arquitetura preserva a proposta do notebook:

- gerador com camadas densas e saída `sigmoid`;
- discriminador binário com camadas densas;
- treinamento alternado entre discriminador e gerador;
- uso de ruído latente com dimensão configurável.

O objetivo é gerar candidatos tabulares coerentes antes do pós-processamento brasileiro.

## 4. Seleção de candidatos

Após a geração, cada candidato passa por:

- regras de domínio para idade, sexo e renda;
- score mínimo do discriminador;
- contagem de candidatos aceitos e rejeitados.

Essas métricas são gravadas no relatório de execução.

## 5. Pós-processamento brasileiro

Os candidatos aceitos são transformados no dataset final:

- `Sexo` é convertido para `Gênero`;
- `Data_Nascimento` é derivada da idade;
- `Nome` é gerado com Faker em `pt_BR`;
- CPF, CNH, RG, título de eleitor e telefone são gerados por funções determinísticas/fictícias;
- a coluna `Idade` é removida da exportação final.

## 6. Validações finais

O dataset final é avaliado por:

- formato de CPF, RG e telefone;
- dígitos verificadores de CPF;
- duplicidades internas de CPF, CNH, RG, título de eleitor e telefone;
- taxa aproximada de conformidade.

As validações são estruturais. Elas não equivalem a consulta oficial, validação documental governamental ou garantia de inexistência no mundo real.

