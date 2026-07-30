# Modelo Geográfico Neural

## Motivação

A confirmação da CTGAN com `income_model_version = 3` mostrou que a representação histórica, registrada como `geography_model_version = 1`, fazia a rede gerar `Regiao`, `Estado`, `Municipio` e `DDD` como colunas categóricas independentes. Essas colunas, porém, representam uma hierarquia. O resultado bruto tinha valores individuais conhecidos, mas frequentemente combinados de forma incompatível.

No diagnóstico da CTGAN candidate_c, a validade geográfica bruta ficou entre 0,19% e 0,31%, enquanto a validade profissional bruta ficou entre 92% e 94%. Isso indicou que a principal dependência do pós-processamento estava na combinação geográfica, não no vocabulário individual nem nas relações profissionais.

## `Geo_Key`

A versão `geography_model_version = 2` introduz a coluna interna `Geo_Key`. Cada chave representa uma combinação sintética permitida de:

- `Regiao`;
- `Estado`;
- `Municipio`;
- `DDD`.

As chaves usam identificadores técnicos estáveis, como `GEO_000001`, e são derivadas de forma determinística das fontes canônicas `REGION_STATES`, `STATE_MUNICIPALITIES` e `STATE_DDDS`. A saída pública continua preservando as 11 colunas-base históricas. `Geo_Key` não aparece no CSV, JSON, Parquet, manifesto de geração ou interface como coluna exportada.

## Catálogo

O catálogo geográfico é construído por `build_geography_catalog()` em `synthetic_br_profiles_gan.domain.geography`. Ele não duplica manualmente as tabelas de domínio; apenas deriva as combinações permitidas das fontes já existentes.

Na execução atual:

- cardinalidade do catálogo: 201 chaves;
- `geography_catalog_version`: `1`;
- checksum SHA-256: `0b12f8466842767c637a37cbff3939d730c1a06c87770c0846cfdeebd8ccf033`.

O carregamento de uma CTGAN com geografia v2 valida a presença do catálogo salvo, a versão, o checksum e a compatibilidade das chaves esperadas.

## Limitação do DDD

A fonte local do projeto relaciona DDDs permitidos por estado, não um DDD oficial único por município. Portanto, a `Geo_Key` representa uma combinação permitida pelo modelo sintético local:

```text
Região + Estado + Município + DDD permitido para o Estado
```

Ela não deve ser interpretada como confirmação oficial de que aquele DDD pertence ao município real.

## Treinamento e Amostragem

Na CTGAN com geografia v2, as colunas `Regiao`, `Estado`, `Municipio` e `DDD` são codificadas antes do `fit` em uma única coluna `Geo_Key`. A representação interna de treinamento passa a ser:

```text
Geo_Key
Idade
Genero
Escolaridade
Estado_Civil
Ocupacao
Renda
Dependentes
```

Depois da amostragem, a chave é validada e decodificada. A saída volta à ordem canônica:

```text
Idade
Genero
Regiao
Estado
Municipio
Escolaridade
Estado_Civil
Ocupacao
Renda
Dependentes
DDD
```

Chaves ausentes, nulas ou desconhecidas são marcadas como `unknown_geography_key` e seguem o fluxo normal de rejeição de candidatos inválidos. O sistema não substitui silenciosamente uma chave desconhecida por uma localidade aleatória.

## Compatibilidade

Modelos CTGAN antigos continuam carregando com `geography_model_version = 1`. A `SimpleTabularGAN` permanece na representação histórica e segue como baseline experimental. O `ProgrammaticSynthesizer` não foi alterado: ele continua gerando a geografia diretamente por regras explícitas e pode registrar `geography_generation_strategy = "direct_rules"`.

## Comparação v1 × v2

Foi criado o perfil `ctgan_income_v3_geo_v2_candidate`, mantendo os hiperparâmetros da CTGAN candidate_c e alterando apenas a representação geográfica neural. A confirmação independente usou seeds `47`, `48` e `49`, treino de 20.000 registros, holdout de 5.000 registros, `synthetic_rows = 20000`, vocabulário v2 e renda v3.

| Métrica | Candidate C geography v1 | Candidate geography v2 | Diferença |
| --- | ---: | ---: | ---: |
| Validade geográfica raw | 0,19% a 0,31% | 100,00% | melhora substancial |
| Validade global raw | 0,18% a 0,29% | 91,56% a 96,85% | melhora substancial |
| Validade profissional raw | 92,71% a 94,23% | 91,59% a 96,85% | sem regressão estrutural grave |
| `known_geography_key_rate` | não aplicável | 100,00% | novo controle |
| Cobertura de `Geo_Key` | não aplicável | 100,00% | cobertura completa |
| TVD de `Geo_Key` | não aplicável | 0,098 a 0,111 | distribuição ainda monitorada |
| Duplicidade-base | 0,000 | 0,000 | preservada |
| Match exato com treino | 0,000 | 0,000 | preservado |
| Status | `approved` nas seeds 44-46 | `approved` nas seeds 47-49 | confirmação independente |

A validade geográfica de 100% não implica distribuição perfeita. Por isso, a avaliação também registra cobertura de chaves, cobertura de estados, municípios e DDDs, TVD por componente e cobertura de chaves raras.

## Artefato aprovado

A confirmação gerou o artefato:

```text
artifacts/models/ctgan/20260730T123208Z-income-v3-geo-v2-approved/
```

Finalidade: `approved`.

O artefato aprovado foi criado por cópia do candidato `artifacts/models/ctgan/20260730T013320Z-income-v3-geo-v2-candidate/`, que permanece preservado. A cópia registra `approval_manifest.json`, `recommended_for_neural_generation = true` e `general_platform_default = false`.

Esse status não significa `default`, `production`, certificação externa, garantia de anonimização ou validação populacional oficial. A aprovação é uma decisão técnica interna baseada nas evidências do projeto.

## Limitações

Os resultados dependem do ambiente, das versões das bibliotecas, dos hiperparâmetros e das seeds avaliadas. A chave composta reduz a dependência do pós-processamento para geografia, mas não elimina a necessidade de validação estrutural final, pós-processamento de identificadores, controle de renda, seleção global e avaliação de privacidade.
