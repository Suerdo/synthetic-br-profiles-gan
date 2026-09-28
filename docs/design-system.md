# Design System da Interface React

Este documento registra as decisões visuais da interface oficial React + TypeScript + Vite.

## Objetivo Visual

A aplicação deve transmitir pesquisa aplicada, governança e operação cuidadosa. O desenho privilegia legibilidade, clareza técnica e separação entre ações, evidências e limitações.

## Identidade

A marca visual fica em:

```text
frontend/public/brand/dados-sinteticos-br-logo.png
```

Ela é usada no cabeçalho lateral e como favicon. A marca textual exibida é `Dados Sintéticos Brasileiros`.

## Paleta

| Papel | Cor |
| --- | --- |
| Fundo da aplicação | `#F1F5F9` |
| Fundo principal | `#FFFFFF` |
| Fundo de seção | `#F8FAFC` |
| Texto principal | `#0F172A` |
| Texto comum | `#334155` |
| Texto secundário | `#64748B` |
| Azul principal | `#2563EB` |
| Azul escuro | `#1E3A8A` |
| Borda | `#CBD5E1` |
| Borda forte | `#94A3B8` |

Sidebar:

| Papel | Cor |
| --- | --- |
| Fundo | `#0F172A` |
| Item ativo | `#1E3A8A` |
| Hover | `#1E293B` |
| Texto principal | `#F8FAFC` |
| Texto secundário | `#CBD5E1` |
| Indicador ativo | `#60A5FA` |

## Navegação

A navegação possui somente:

- `Gerar Dados`;
- `Modelos`;
- `Governança`.

`Gerar Dados` é a página inicial. Em telas menores, o menu lateral é substituído por drawer mobile acessível.

## Componentes

Cards usam fundo claro, borda visível, raio consistente, sombra leve e espaçamento interno confortável. A página `Modelos` usa cards de resumo simples e técnico com alturas equivalentes em desktop, empilhando em telas estreitas.

Badges sempre comunicam estado por texto, não apenas por cor. Estados comuns:

- `Aprovado`;
- `Quarentena`;
- `Rejeitado`;
- `Experimental`;
- `Candidato`;
- `Smoke`;
- `Legado`;
- `Mais recente`;
- `Recomendado`.

`Mais recente` indica recência do manifesto, não qualidade nem aprovação.

## Formulários

Campos editáveis devem parecer interativos sem depender de foco: fundo contrastado, borda visível, labels claros e foco azul. Botões mantêm aparência distinta dos campos de configuração.

## Governança

A página `Governança` segue uma narrativa de evidência:

```text
configuração → geração → validação → avaliação → evidências → decisão → artefato → rastreabilidade
```

Auditoria e glossário ficam recolhidos por padrão. Valores ausentes aparecem como `Não avaliado`; zeros são exibidos somente quando registrados como zero real.

## Acessibilidade

Regras adotadas:

- contraste adequado para texto comum;
- foco visível;
- labels explícitos;
- botões e links com nomes acessíveis;
- status comunicados por texto;
- tabelas usadas quando facilitam consulta;
- conteúdo essencial preservado em modo responsivo.

## Tom de Voz

O texto usa português brasileiro formal, claro e técnico. A interface evita promessas absolutas, como garantia de anonimização, inexistência de documentos ou conformidade regulatória completa.
