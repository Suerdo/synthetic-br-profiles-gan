# Design system da interface

Este documento registra as decisões visuais da primeira versão da interface Streamlit.

## Objetivo visual

A interface deve transmitir pesquisa aplicada, governança e operação cuidadosa. O desenho privilegia legibilidade, clareza técnica e separação entre ações, evidências e limitações.

## Tema

O tema fica em `.streamlit/config.toml` e utiliza base clara:

| Papel | Cor |
| --- | --- |
| Primária | `#2563EB` |
| Fundo | `#FFFFFF` |
| Fundo da aplicação | `#F1F5F9` |
| Fundo de seção | `#F8FAFC` |
| Texto principal | `#0F172A` |
| Texto comum | `#334155` |
| Texto secundário | `#64748B` |
| Borda | `#CBD5E1` |
| Borda forte | `#94A3B8` |

A sidebar usa:

| Papel | Cor |
| --- | --- |
| Fundo | `#0F172A` |
| Fundo do item ativo | `#1E3A8A` |
| Hover | `#1E293B` |
| Texto principal | `#F8FAFC` |
| Texto secundário | `#CBD5E1` |
| Indicador ativo | `#60A5FA` |

A paleta complementar fica em `synthetic_br_profiles_gan.ui.theme`. Os testes validam contraste mínimo WCAG AA para pares principais de texto.

## Navegação

A navegação principal possui três áreas:

- `Gerar dados`;
- `Modelos`;
- `Governança`.

`Gerar dados` é a página inicial. A navegação deve parecer um menu lateral de plataforma, não um conjunto de radio buttons. O item ativo precisa ser evidente por texto, fundo e indicador lateral.

A sidebar não possui rodapé informativo nesta versão. Ela termina após `Gerar dados`, `Modelos` e `Governança`, sem status operacional, versão, vocabulário ou frase complementar.

A marca curta exibida no menu é `Dados Sintéticos Brasileiro`.

## Títulos

Títulos de páginas, subtítulos, seções e cards usam capitalização com iniciais maiúsculas nas palavras principais. Exemplos:

- `CTGAN — Modelo Tabular Avançado`;
- `GAN Simples — Experimental`;
- `Resumo Operacional`;
- `Qualidade dos Dados`;
- `Execuções Recentes`;
- `Resumo Simples`;
- `Resumo Técnico`.

Labels funcionais de campos podem usar frase natural quando isso melhora a leitura, como `Quantidade de registros`, `Modo de seleção`, `Tipo`, `Modelo` e `Status`.

## Componentes

Componentes reutilizáveis ficam em `synthetic_br_profiles_gan.ui.components`:

- cartões informativos;
- indicadores executivos;
- fluxo operacional acessível;
- badges padronizados;
- seções numeradas.

Eles são apenas componentes de apresentação. Regras de geração, validação, seleção de colunas, manifestos, auditoria e governança permanecem nos serviços do pacote.

## Estados

Status internos continuam em inglês nos manifestos e no código, mas a interface apresenta rótulos em português quando necessário:

- aprovado (`approved`);
- em quarentena (`quarantined`);
- rejeitado (`rejected`);
- falha técnica (`failed`);
- limitação de recursos (`resource_limited`);
- não disponível (`not_available`).

A cor nunca deve ser o único meio de comunicação do status. O texto do status deve estar sempre visível.

Badges padronizados:

- `Aprovado`;
- `Quarentena`;
- `Rejeitado`;
- `Experimental`;
- `Candidato`;
- `Smoke`;
- `Legado`;
- `Mais recente`;
- `Recomendado`.

`Mais recente` indica apenas recência do manifesto. Ele não significa melhor qualidade, aprovação ou recomendação.

## Cards

Cards usam fundo claro, borda visível, raio entre 10 e 12 pixels, sombra leve e padding consistente:

```css
border: 1px solid #CBD5E1;
border-radius: 12px;
box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
background: #FFFFFF;
```

Na página `Modelos`, os cards `Resumo Simples` e `Resumo Técnico` usam grid com `align-items: stretch`, a mesma classe visual e altura mínima em desktop. Em telas menores, o grid empilha os cards e remove a altura mínima.

A página `Modelos` foi simplificada para manter somente esses dois cards principais antes de `Usos Recomendados`. O histórico visual de artefatos não aparece nessa página; a seleção de artefatos permanece na tela `Gerar dados`.

## Seções numeradas

As etapas da geração usam fundo `#EFF6FF`, borda `#BFDBFE`, barra lateral azul e número em badge azul. Os títulos seguem esta ordem:

1. Modelo.
2. Volume e Reprodutibilidade.
3. Colunas.
4. Formato.
5. Revisão.

## Campos

Campos de configuração da geração e filtros da governança usam aparência inequívoca de controles editáveis por meio do tema nativo do Streamlit 1.60.0. A configuração fica em `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1E3A8A"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#E8EEF7"
textColor = "#0F172A"
borderColor = "#64748B"
showWidgetBorder = true
```

`secondaryBackgroundColor` cria o fundo azul-acinzentado dos widgets, `borderColor` define a borda visível e `showWidgetBorder` mantém a borda mesmo quando o controle não está em foco. `primaryColor` indica foco, seleção e elementos interativos destacados.

Os `number_input` mantêm caixa delimitada e steppers visíveis. Os `selectbox` mantêm seta de dropdown contrastada e padding suficiente para indicar área clicável. O CSS do pacote não redefine genericamente `input`, `selectbox`, `multiselect` ou elementos internos BaseWeb; qualquer ajuste manual deve ser fallback restrito, documentado e compatível com a estrutura do Streamlit em uso.

## Governança

As seções `Resumo Operacional`, `Qualidade dos Dados`, `Privacidade`, `Execuções Recentes` e `Auditoria` seguem o mesmo sistema visual: container com borda, fundo claro, raio de 12 pixels, sombra leve, título no topo e espaçamento interno consistente. `Execuções Recentes` usa o mesmo padrão para que os filtros `Tipo`, `Modelo` e `Status` pareçam controles interativos da seção, e não texto solto na página.

## Downloads

Os botões `Baixar dataset` e `Baixar manifesto` ficam lado a lado e próximos, com proporções que evitam ocupar metade da página cada um. Em telas estreitas, o comportamento responsivo do Streamlit pode empilhar os botões.

## Destaques

Avisos de governança e a frase final usam fundo `#FFF7ED`, borda `#FDBA74` e texto `#9A3412`. A frase final não usa exclamação nem emoji.

## Acessibilidade

Regras adotadas:

- contraste mínimo WCAG AA para texto comum;
- labels explícitos nos controles;
- títulos hierárquicos por página;
- foco visível em botões;
- estados comunicados por texto e não apenas por cor;
- tabelas usadas somente quando facilitam consulta;
- ausência de imagens externas ou dependências visuais pesadas nesta fase.

## Tom de voz

O texto deve usar português brasileiro formal, claro e técnico. A interface deve evitar promessas absolutas, como garantia de anonimização, inexistência de documentos ou conformidade regulatória completa.

## Limites

Esta versão não implementa sistema de design externo, autenticação, histórico persistente, upload de modelos ou customização por usuário. A prioridade é preparar uma base institucional reutilizável sobre os serviços já existentes.
