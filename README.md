# Geração de Dados Sintéticos com IA Generativa

Repositório do projeto acadêmico **“Geração de Dados Sintéticos com IA Generativa: Inovação Segura e Governança Alinhadas à LGPD e ao ECA Digital”**.

O projeto implementa um pipeline em Python para geração de perfis pessoais sintéticos brasileiros usando uma GAN tabular, geradores determinísticos de identificadores fictícios e validações estruturais. O objetivo é produzir dados coerentes para testes, homologação, pesquisa e experimentação, sem utilizar dados pessoais reais.

## Contexto Acadêmico

Este repositório é um artefato de pesquisa/TCC e acompanha uma proposta voltada a:

- geração de perfis sintéticos brasileiros;
- IA generativa aplicada a dados tabulares;
- validações de formato e unicidade para identificadores fictícios;
- rastreabilidade da execução;
- reprodutibilidade por seed;
- governança de dados, privacy by design e accountability;
- discussão conceitual alinhada à LGPD e ao Estatuto Digital da Criança e do Adolescente.

## Proposta

O pipeline combina duas camadas:

1. **Camada generativa tabular**: uma GAN aprende distribuições simples de uma base de calibração sintética, com atributos como idade, sexo e renda.
2. **Camada determinística e validável**: após a geração, o projeto constrói campos finais como nome, data de nascimento, CPF, CNH, RG, título de eleitor e telefone, todos fictícios e voltados a uso controlado.

Essa separação preserva a proposta científica: a GAN atua sobre a coerência estatística tabular, enquanto os identificadores são gerados por regras estruturais para evitar a aparência de dados coletados de pessoas reais.

## Funcionalidades

- Treinamento de uma GAN tabular com TensorFlow/Keras.
- Geração de perfis sintéticos com campos brasileiros.
- Geração de CPF com dígitos verificadores válidos.
- Geração de RG, CNH, título de eleitor e telefone com estrutura plausível.
- Validações de CPF, RG, telefone e duplicidades internas.
- Exportação de dataset final e relatório JSON.
- Registro de métricas de aceitação, rejeição, vazão e resumo univariado.
- Reprodutibilidade por seed em Python, NumPy, TensorFlow e Faker.
- Execução via notebook acadêmico ou script local.

## Fluxo do Pipeline

```text
Base de calibração sintética
        |
        v
Pré-processamento numérico
        |
        v
Treinamento da GAN tabular
        |
        v
Geração de candidatos sintéticos
        |
        v
Filtro por domínio e score do discriminador
        |
        v
Pós-processamento brasileiro
        |
        v
Validações, relatório e exportação
```

## Tecnologias

- Python 3.12 recomendado
- TensorFlow/Keras
- pandas
- NumPy
- scikit-learn
- Faker
- openpyxl

## Estrutura do Repositório

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── geracao_de_dados_pessoais_sinteticos_lgpd.ipynb
├── src/
│   └── synthetic_br_profiles_gan/
│       ├── generators/
│       ├── validators/
│       ├── models/
│       ├── reports/
│       ├── utils/
│       └── pipeline.py
├── scripts/
│   └── run_pipeline.py
├── data/
│   ├── samples/
│   └── outputs/
├── docs/
│   ├── governance.md
│   ├── methodology.md
│   ├── outputs.md
│   └── reproducibility.md
└── tests/
```

## Instalação Local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Em Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O notebook foi pensado para Google Colab. Para execução local, Python 3.12 tende a ser a opção mais previsível por causa da compatibilidade com TensorFlow/Keras.

## Execução

Execução principal por script:

```bash
python scripts/run_pipeline.py --n 1000 --seed 41 --output data/outputs
```

Execução menor para teste rápido:

```bash
python scripts/run_pipeline.py --n 100 --seed 41 --epochs 10 --calibration-size 2000 --output data/outputs
```

Parâmetros úteis:

- `--n`: quantidade final de registros.
- `--seed`: seed de reprodutibilidade.
- `--output`: diretório de saída.
- `--epochs`: épocas de treinamento da GAN.
- `--calibration-size`: tamanho da base de calibração.
- `--reference-date`: data fixa para cálculo de datas de nascimento, no formato `YYYY-MM-DD`.

## Execução no Google Colab

O notebook principal está em:

```text
notebooks/geracao_de_dados_pessoais_sinteticos_lgpd.ipynb
```

No Colab:

1. Abra o notebook pelo GitHub ou faça upload do arquivo.
2. Ative GPU se disponível.
3. Execute as células na ordem.
4. Ao final, o notebook exporta `dados_sinteticos_realistas.xlsx` e `relatorio_execucao.json` em `data/outputs/`.

O notebook foi preservado como artefato acadêmico. A pasta `src/` organiza a mesma lógica em módulos para execução local e testes.

## Saídas Geradas

Por padrão, o script grava em `data/outputs/`:

- `dados_sinteticos_realistas.xlsx`: dataset final fictício.
- `relatorio_execucao.json`: métricas, parâmetros e validações da execução.

Campos típicos do dataset:

- `Nome`
- `Gênero`
- `Data_Nascimento`
- `CPF`
- `CNH`
- `RG`
- `Titulo_Eleitor`
- `Telefone`
- `Renda`

## Dados Sintéticos

Os dados gerados por este projeto são fictícios. Eles não devem ser interpretados como registros de pessoas reais, nem usados para decisões operacionais sobre indivíduos.

Importante: dados sintéticos **não devem ser automaticamente considerados anonimizados** sem avaliação de risco. Dependendo do método, do conjunto de origem e do contexto de uso, podem ser necessárias análises adicionais de privacidade, utilidade e risco de reidentificação.

## Governança, LGPD e ECA Digital

O projeto foi desenhado como demonstração de privacy by design:

- não utiliza dados pessoais reais;
- gera identificadores fictícios;
- registra parâmetros e métricas da execução;
- permite reprodutibilidade por seed;
- separa dados gerados de documentação e código;
- orienta uso apenas em ambientes controlados.

A LGPD é considerada como referência conceitual para princípios como finalidade, adequação, necessidade, segurança, prevenção, responsabilização e prestação de contas. O ECA Digital é considerado no plano de governança por reforçar cuidado especial com crianças e adolescentes em ambientes digitais. Neste pipeline, a base sintética de calibração usa faixa adulta, de 18 a 65 anos.

Referências normativas oficiais:

- [Lei nº 13.709/2018 - LGPD](https://www.planalto.gov.br/ccivil_03/_Ato2015-2018/2018/Lei/L13709compilado.htm)
- [Lei nº 15.211/2025 - Estatuto Digital da Criança e do Adolescente](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2025/lei/l15211.htm)
- [Decreto nº 12.880/2026 - regulamentação da Lei nº 15.211/2025](https://www.planalto.gov.br/ccivil_03/_ato2023-2026/2026/Decreto/D12880.htm)

## Aviso Ético e Jurídico

Este projeto é destinado exclusivamente a pesquisa, testes, homologação e experimentação. Não utilize os dados gerados para fraude, falsificação documental, simulação de identidade real, criação de contas indevidas, engenharia social ou qualquer finalidade ilícita.

Identificadores gerados, ainda que tenham formato válido, são fictícios e devem permanecer em ambientes controlados. O projeto não substitui avaliação jurídica, relatório de impacto, governança institucional ou auditoria de privacidade.

## Testes

Os testes usam `unittest`:

```bash
python -m unittest discover -s tests
```

Eles cobrem validação de CPF, formato de RG e telefone, unicidade e exportação de relatório em JSON.

## Reprodutibilidade

O script fixa seeds para Python, NumPy, TensorFlow e Faker. Para reproduzir também datas de nascimento, informe uma data de referência:

```bash
python scripts/run_pipeline.py --n 1000 --seed 41 --reference-date 2026-05-02 --output data/outputs
```

Treinamentos neurais podem apresentar pequenas diferenças conforme versões de bibliotecas, hardware e backend. O relatório JSON registra parâmetros relevantes para rastreabilidade.

## Limitações Conhecidas

- A base de calibração é sintética e simplificada.
- A GAN opera sobre poucos atributos tabulares.
- As validações são estruturais e não equivalem a validação documental oficial.
- O projeto não mede, por enquanto, risco formal de reidentificação.
- O pipeline não deve ser usado para geração de identidades operacionais.
- O notebook e o script podem produzir pequenas diferenças por ambiente.

## Trabalhos Futuros

- Adicionar métricas de utilidade estatística entre base de calibração e dados gerados.
- Avaliar risco de privacidade com técnicas específicas para dados sintéticos.
- Incluir testes automatizados para mais regras semânticas.
- Documentar experimentos comparativos com outras técnicas de geração tabular.
- Criar exemplos pequenos em `data/samples/` para demonstração pública.
- Definir licença formal antes da publicação pública.
