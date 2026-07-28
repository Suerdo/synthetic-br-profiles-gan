# Vocabulário categórico em português brasileiro

O idioma canônico dos valores textuais gerados pelo projeto é português brasileiro (`pt-BR`). A versão atual do vocabulário categórico é `2`, com normalização Unicode `NFC`.

Essa versão altera valores categóricos, municípios e ocupações, mas não altera nomes técnicos de colunas. Identificadores como `Genero`, `Regiao`, `Municipio`, `Ocupacao`, `Estado_Civil`, `Data_Nascimento` e `Titulo_Eleitor` continuam sendo contratos internos usados por modelos salvos, CLI, manifestos, validações, benchmarks e interface.

## Categorias canônicas

Escolaridade:

- `Fundamental`;
- `Ensino Médio`;
- `Superior Incompleto`;
- `Superior Completo`;
- `Pós-graduação`.

Estado civil:

- `Solteiro`;
- `Casado`;
- `União Estável`;
- `Divorciado`;
- `Viúvo`.

As categorias de gênero permanecem conceituais e não são usadas para ajustar renda ou ocupação.

## Municípios

A tabela local em `domain/brazil.py` preserva as relações entre estado, município e DDD, mas corrige grafias como `Maceió`, `Palmeira dos Índios`, `Macapá`, `Vitória da Conquista`, `Brasília`, `Ceilândia`, `Vitória`, `Goiânia`, `Anápolis`, `Aparecida de Goiânia`, `São Luís`, `Cuiabá`, `Várzea Grande`, `Rondonópolis`, `Belém`, `Santarém`, `João Pessoa`, `Maringá`, `Niterói`, `Petrópolis`, `Mossoró`, `Ji-Paraná`, `Rorainópolis`, `Caracaraí`, `Florianópolis`, `São Paulo` e `Araguaína`.

Essas tabelas continuam sendo referências sintéticas locais. Elas não consultam bases oficiais e não devem ser tratadas como comprovação de existência, endereço ou associação real.

## Catálogo de ocupações

As ocupações ficam centralizadas em `domain/occupations.py` por meio de `OccupationProfile`. Cada entrada registra:

- `name`;
- `group`;
- `allowed_education`;
- `minimum_age`;
- `maximum_age`;
- `income_multiplier`;
- `sampling_weight`;
- `description`;
- `income_variability`.

O catálogo ampliado possui mais de 30 ocupações, incluindo `Estudante`, `Estagiário`, `Serviços Gerais`, `Atendente`, `Operador de Caixa`, `Recepcionista`, `Vendedor`, `Auxiliar Administrativo`, `Motorista`, `Entregador`, `Agricultor`, `Pedreiro`, `Eletricista`, `Mecânico`, `Técnico`, `Técnico de Informática`, `Técnico de Enfermagem`, `Professor`, `Enfermeiro`, `Assistente Social`, `Designer`, `Desenvolvedor de Software`, `Analista`, `Analista de Dados`, `Analista Administrativo`, `Contador`, `Engenheiro`, `Arquiteto`, `Advogado`, `Dentista`, `Médico`, `Coordenador`, `Gerente`, `Diretor`, `Autônomo`, `Microempreendedor` e `Aposentado`.

## Coerência sintética

A amostragem de ocupação usa idade e escolaridade para filtrar ocupações elegíveis e depois sorteia com pesos contextuais. Profissões que exigem formação superior, como `Médico`, `Dentista`, `Engenheiro`, `Arquiteto`, `Advogado`, `Enfermeiro`, `Professor` e `Contador`, não são geradas com `Fundamental` ou `Ensino Médio`.

Pesos etários aumentam a tendência de:

- `Estagiário` e `Estudante` em faixas jovens;
- `Coordenador`, `Gerente` e `Diretor` em faixas de maior experiência;
- `Aposentado` em idades mais altas.

Essas regras são sintéticas e configuráveis. Elas não são estatísticas oficiais e não devem ser usadas para inferir comportamento real da população brasileira.

## Renda

A renda mensal sintética depende de idade, escolaridade, ocupação, região e variação estocástica. O catálogo de ocupações define multiplicadores de tendência e variabilidade. Ocupações de maior qualificação tendem a ter renda agregada maior, mas as distribuições mantêm sobreposição.

O cálculo não usa gênero. O projeto não introduz diferenças de renda por gênero.

Os multiplicadores de renda são parâmetros heurísticos do gerador sintético. Eles não representam dados oficiais do mercado de trabalho brasileiro.

## Compatibilidade com vocabulário legado

Valores antigos são normalizados por aliases explícitos, por exemplo:

- `Ensino Medio` → `Ensino Médio`;
- `Pos-graduacao` → `Pós-graduação`;
- `Uniao Estavel` → `União Estável`;
- `Viuvo` → `Viúvo`;
- `Servicos Gerais` → `Serviços Gerais`;
- `Tecnico` → `Técnico`;
- `Autonomo` → `Autônomo`;
- `Sao Paulo` → `São Paulo`;
- `Joao Pessoa` → `João Pessoa`.

Modelos neurais antigos continuam carregando quando o schema de colunas é compatível. A saída desses modelos recebe correção de acentuação e normalização Unicode antes da validação, mas eles não passam a produzir automaticamente ocupações adicionadas na versão `2`. Para aprender as novas ocupações, `SimpleTabularGAN` e `CTGANSynthesizer` precisam ser treinados novamente.

Configurações pequenas para validação técnica do vocabulário `2` estão disponíveis em:

- `configs/train-simple-gan-vocab-v2-smoke.yaml`;
- `configs/train-ctgan-vocab-v2-smoke.yaml`.

Essas configurações são smoke tests e não substituem treinamento experimental completo.

## Qualidade dos sintetizadores com o vocabulário 2

O primeiro benchmark de qualidade específico para o vocabulário `2` foi executado em `quality-vocab-v2-20260728T131154Z-364a35cb`. A configuração principal usou os três sintetizadores (`programmatic`, `simple_gan` e `ctgan`), seeds `41`, `42` e `43`, 20.000 registros de treinamento, 5.000 registros de holdout e 20.000 registros sintéticos por execução.

Esse benchmark avalia o vocabulário em dois estágios:

- `raw`: saída diagnóstica do sintetizador antes de aliases, normalização pt-BR, pós-processamento e validação final;
- `final`: dataset exportável após normalização, pós-processamento e validação estrutural.

A distinção é importante porque o resultado `final` pode ser corrigido pelo pipeline. A qualidade da saída `raw` indica melhor o que o modelo aprendeu diretamente sobre o vocabulário.

Resultados principais:

- `ProgrammaticSynthesizer`: cobriu as 37 ocupações nas três seeds em `raw` e `final`, preservou a validade completa entre escolaridade, idade e ocupação e manteve melhor aderência às tendências de renda por ocupação. Esse resultado possui vantagem estrutural, pois a referência sintética é gerada pelas mesmas regras.
- `CTGANSynthesizer`: cobriu as 37 ocupações nas três seeds em `raw` e `final`. A validade bruta entre escolaridade e ocupação ficou em torno de 91%, e a validade bruta entre idade e ocupação ficou acima de 97%. O resultado final ficou estruturalmente válido após o pipeline, mas a separação de renda entre algumas ocupações foi menor do que na referência.
- `SimpleTabularGAN`: permaneceu como baseline experimental. A cobertura final média ficou próxima de 10,8% das ocupações, com forte concentração em poucas categorias e ausência de 31 a 36 ocupações por seed.

Categorias raras foram definidas como ocupações com participação inferior a 1% no conjunto de holdout. `ProgrammaticSynthesizer` e `CTGANSynthesizer` reproduziram todas as ocupações raras observadas nas três seeds. `SimpleTabularGAN` não reproduziu ocupações raras no resultado final.

Os quality gates específicos do vocabulário `2` tratam como bloqueantes a quantidade final de linhas, o schema final, categorias finais canônicas, ausência de categorias legadas no resultado final, coerência estrutural final, renda dentro dos limites e normalização Unicode `NFC`. Cobertura de ocupações, entropia, concentração da ocupação dominante e coerência bruta são métricas diagnósticas nesta primeira versão.

O benchmark não altera o catálogo, os pesos, as faixas etárias nem os multiplicadores de renda. Ele mede a implementação atual. Também não substitui os benchmarks históricos, que usaram o vocabulário anterior.

Recomendação experimental:

- `ProgrammaticSynthesizer`: permanece como opção padrão da interface;
- `CTGANSynthesizer`: aprovada para treinamento de um artefato candidato maior;
- `SimpleTabularGAN`: permanece como baseline acadêmico experimental.

Modelos neurais antigos continuam compatíveis quando o schema é aceito, mas não devem ser avaliados como se conhecessem as 37 ocupações. Para reproduzir o catálogo ampliado, `SimpleTabularGAN` e `CTGANSynthesizer` precisam ser treinados novamente com o vocabulário `2`.

## Manifestos

Novos manifestos de treinamento registram:

- `data_locale`;
- `unicode_normalization`;
- `categorical_vocabulary_version`.

Novos manifestos de geração registram:

- `data_locale`;
- `unicode_normalization`;
- `source_model_vocabulary_version`;
- `output_vocabulary_version`;
- `legacy_value_normalization_applied`.

Esses campos distinguem a versão do vocabulário da versão do schema.

## Codificação

Arquivos CSV gerados pelo serviço de geração usam `utf-8-sig` e separador `;`. JSON usa UTF-8 com `ensure_ascii=False`. Parquet preserva Unicode por meio do backend tabular.
