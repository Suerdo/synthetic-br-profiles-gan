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
