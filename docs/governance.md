# Governança, LGPD e Uso Responsável

Este projeto foi organizado para apoiar uma discussão acadêmica sobre dados sintéticos, IA generativa e governança de dados.

## Princípios adotados

- **Finalidade**: uso restrito a pesquisa, testes, homologação e experimentação.
- **Minimização**: geração de campos suficientes para simular perfis, sem coletar dados reais.
- **Privacy by design**: identificadores são fictícios e gerados por regras locais.
- **Accountability**: o relatório registra seed, parâmetros, métricas e validações.
- **Rastreabilidade**: os artefatos gerados são separados em `data/outputs/`.
- **Controle de risco**: o projeto inclui avisos contra uso indevido e validações estruturais.

## LGPD

A LGPD é usada como referência conceitual para governança, segurança, prevenção e responsabilização. O projeto não deve ser interpretado como certificação de conformidade legal.

Dados sintéticos não são automaticamente anonimizados. Antes de usar dados sintéticos em ambiente produtivo ou compartilhamento externo, recomenda-se avaliar:

- origem dos dados usados para calibração;
- risco de reidentificação;
- memorização do modelo;
- finalidade e contexto de uso;
- controles de acesso e descarte;
- documentação dos parâmetros de geração.

## ECA Digital

O Estatuto Digital da Criança e do Adolescente reforça a necessidade de proteção prioritária de crianças e adolescentes em ambientes digitais. Neste pipeline, a base de calibração usa faixa adulta, de 18 a 65 anos.

Caso versões futuras gerem dados envolvendo menores de idade, recomenda-se criar salvaguardas específicas, justificativa técnica explícita e avaliação de risco reforçada.

## Uso proibido

Não use este projeto para:

- fraude ou falsificação documental;
- simulação de identidade real;
- criação indevida de contas;
- engenharia social;
- tomada de decisão sobre pessoas;
- treinamento de sistemas que busquem representar indivíduos reais.

## Referências oficiais

- Lei nº 13.709/2018 - LGPD: https://www.planalto.gov.br/ccivil_03/_Ato2015-2018/2018/Lei/L13709compilado.htm
- Lei nº 15.211/2025 - Estatuto Digital da Criança e do Adolescente: https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2025/lei/l15211.htm
- Decreto nº 12.880/2026: https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2026/Decreto/D12880.htm

