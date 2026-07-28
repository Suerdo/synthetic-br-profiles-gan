# Conformidade regulatória

Este documento descreve como a plataforma pode apoiar processos de governança, privacidade e uso responsável. Ele não constitui parecer jurídico, certificação regulatória, auditoria formal ou garantia de conformidade.

## Escopo

A plataforma gera dados sintéticos brasileiros para desenvolvimento, testes, engenharia de requisitos, demonstrações técnicas, pesquisa acadêmica, treinamento de modelos de Inteligência Artificial e validação de pipelines de dados.

Os dados gerados não são consultados nem validados em bases oficiais. A validade estrutural de documentos não comprova existência, regularidade ou associação a uma pessoa real.

## Matriz de evidências

A matriz permanece como referência documental em `docs/compliance.md` e não é exibida na página `Governança` da interface Streamlit nesta versão simplificada.

A referência documental usa uma matriz com os seguintes status:

| Status | Significado |
| --- | --- |
| `Implementado` | Há funcionalidade técnica implementada e evidência local. |
| `Parcialmente implementado` | Há apoio técnico, mas a aplicação não cobre todo o processo institucional. |
| `Não evidenciado` | Não há manifesto, auditoria ou evidência local suficiente. |
| `Requer avaliação institucional` | Depende de política, avaliação jurídica, processo ou decisão organizacional. |
| `Não aplicável ao cenário avaliado` | O item não se aplica ao uso documentado. |

A matriz não calcula nota geral, ranking, percentual de conformidade ou certificação.

## LGPD

A Lei Geral de Proteção de Dados Pessoais, Lei nº 13.709/2018, estabelece princípios e obrigações para tratamento de dados pessoais. O uso de dados sintéticos pode apoiar minimização de exposição em ambientes de teste, mas não elimina a necessidade de avaliação institucional.

A plataforma apoia:

- finalidade técnica documentada em manifestos;
- geração local sem consulta a bases reais;
- seleção de colunas na exportação;
- validação estrutural;
- indicadores de risco de memorização quando os benchmarks são executados;
- rastreabilidade por manifestos e auditoria sanitizada.

A plataforma não substitui:

- definição de base legal;
- relatório de impacto;
- governança de acesso;
- política de retenção;
- revisão jurídica;
- avaliação de anonimização.

## ECA Digital

A Lei nº 15.211/2025, conhecida como ECA Digital, e o Decreto nº 12.880/2026 tratam de deveres relacionados à proteção de crianças e adolescentes em ambientes digitais. A plataforma não foi desenhada como sistema voltado a crianças ou adolescentes e não executa avaliação jurídica desse contexto.

Quando a ferramenta for usada em projetos que envolvam crianças ou adolescentes, a equipe responsável deve realizar avaliação institucional específica. A interface apresenta avisos de uso responsável, mas isso não comprova conformidade material com todos os deveres legais.

## Auditoria e rastreabilidade

A interface registra eventos sanitizados em `artifacts/ui_audit/events.jsonl`. Esses eventos não incluem valores individuais gerados, documentos, nomes, telefones, IP, user agent, identidade de usuário ou traceback completo.

Manifestos de geração registram modelo, seed, formato, colunas exportadas, colunas geradas internamente, validação, ambiente e aviso de governança. Manifestos de treinamento registram versão do vocabulário, schema, dados de configuração, tempos e ambiente.

## Limitações

- Dados sintéticos não devem ser automaticamente considerados anonimizados.
- Métricas de privacidade são indicadores de risco, não prova de anonimização.
- A plataforma não consulta bases oficiais.
- A plataforma não comprova inexistência de documentos.
- A interface não oferece autenticação, aprovação institucional, segregação de papéis ou retenção automatizada.
- Qualidade estatística não significa veracidade individual.

## Referências oficiais

- [Lei Geral de Proteção de Dados Pessoais, Lei nº 13.709/2018](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709compilado.htm) — Planalto.
- [Lei nº 15.211/2025, conhecida como ECA Digital](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2025/lei/l15211.htm) — Planalto.
- [Decreto nº 12.880/2026](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2026/decreto/d12880.htm) — Planalto.
- [Materiais educativos e publicações da ANPD](https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes).
- [Documentos técnicos e orientativos da ANPD](https://www.gov.br/anpd/pt-br/centrais-de-conteudo/documentos-tecnicos-orientativos).

Última revisão deste conteúdo jurídico: 28/07/2026.
