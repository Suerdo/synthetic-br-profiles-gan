"""Catálogo descritivo dos modelos exibidos na interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCatalogEntry:
    """Descrição de um sintetizador para apresentação didática e técnica."""

    name: str
    label: str
    category: str
    status: str
    recommended: bool
    experimental: bool
    complexity_level: str
    short_description: str
    detailed_description: str
    simple_summary: str
    technical_summary: str
    objective: str
    generated_data: str
    recommended_use_cases: tuple[str, ...]
    benefits: tuple[str, ...]
    limitations: tuple[str, ...]
    realism_profile: str
    privacy_considerations: tuple[str, ...]
    compliance_considerations: tuple[str, ...]
    requires_training: bool
    requires_saved_artifact: bool
    capacity_notes: str
    benchmark_notes: str

    @property
    def status_label(self) -> str:
        """Mantém compatibilidade com a primeira versão da interface."""
        return self.status

    @property
    def best_for(self) -> tuple[str, ...]:
        """Mantém compatibilidade com código que usava `best_for`."""
        return self.recommended_use_cases


MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        name="programmatic",
        label="Programático — Recomendado",
        category="Baseline Programático",
        status="Recomendado",
        recommended=True,
        experimental=False,
        complexity_level="Baixa",
        short_description="Gera perfis por regras probabilísticas explícitas e dependências controladas.",
        detailed_description=(
            "Gera dados por regras probabilísticas explícitas e dependências controladas. "
            "Não exige treinamento e oferece geração rápida, previsível e estruturalmente consistente."
        ),
        simple_summary=(
            "O sistema combina regras configuradas, relações entre variáveis e pós-processamento "
            "para produzir perfis coerentes sem treinar uma rede neural."
        ),
        technical_summary=(
            "Utiliza amostragem probabilística explícita, catálogo estruturado de ocupações, "
            "dependências entre região, estado, município, escolaridade, ocupação, idade e renda, "
            "além de pós-processamento contextual para identificadores e campos derivados."
        ),
        objective="Produzir um baseline rápido, reprodutível e estruturalmente consistente.",
        generated_data="Colunas-base sintéticas e 18 colunas finais após pós-processamento e validação.",
        recommended_use_cases=(
            "desenvolvimento",
            "testes funcionais",
            "demonstrações técnicas",
            "validação de pipelines",
            "ensino",
            "grandes volumes locais",
        ),
        benefits=(
            "Baixo custo computacional.",
            "Não exige artefato treinado.",
            "Alta previsibilidade operacional.",
            "Forte validade estrutural quando a configuração é válida.",
        ),
        limitations=(
            "Não aprende uma distribuição externa.",
            "O realismo depende da qualidade das regras configuradas.",
            "Pode reproduzir vieses das parametrizações sintéticas.",
            "Não representa automaticamente a população brasileira.",
            "Não deve ser tratado como fonte estatística oficial.",
        ),
        realism_profile="Coerência estrutural alta, com realismo limitado pelas regras sintéticas.",
        privacy_considerations=(
            "Não consulta bases oficiais.",
            "Documentos válidos estruturalmente não comprovam inexistência real.",
            "Métricas de privacidade continuam sendo indicadores de risco, não garantia de anonimização.",
        ),
        compliance_considerations=(
            "Apoia minimização operacional em ambientes de teste.",
            "Gera manifestos e rastreabilidade técnica.",
            "Não substitui avaliação jurídica ou processo institucional de conformidade.",
        ),
        requires_training=False,
        requires_saved_artifact=False,
        capacity_notes=(
            "Executado com sucesso com 6.400.000 registros equivalentes no ambiente avaliado. "
            "A busca foi encerrada por decisão metodológica; o limite máximo absoluto não foi determinado."
        ),
        benchmark_notes=(
            "No benchmark do vocabulário 2, cobriu as 37 ocupações nas três seeds. "
            "Possui vantagem estrutural porque a referência sintética usa as mesmas regras."
        ),
    ),
    ModelCatalogEntry(
        name="ctgan",
        label="CTGAN — Modelo Tabular Avançado",
        category="Modelo Neural Tabular",
        status="Avançado",
        recommended=False,
        experimental=False,
        complexity_level="Alta",
        short_description="Modelo generativo tabular capaz de aprender atributos numéricos, categóricos e discretos.",
        detailed_description=(
            "Modelo generativo especializado em dados tabulares, capaz de aprender conjuntamente "
            "atributos numéricos, categóricos e discretos."
        ),
        simple_summary=(
            "Aprende padrões a partir de dados tabulares de treinamento e gera novas combinações sintéticas, "
            "que depois passam por normalização, pós-processamento e validação."
        ),
        technical_summary=(
            "Usa a biblioteca `ctgan`, treinamento adversarial tabular, declaração explícita de colunas "
            "discretas e amostragem a partir de um artefato previamente treinado."
        ),
        objective="Avaliar uma alternativa neural tabular mais flexível que a GAN simples.",
        generated_data="Amostra bruta de colunas-base, normalizada e validada pelo pipeline antes da exportação.",
        recommended_use_cases=(
            "experimentos tabulares",
            "comparação com baseline programático",
            "pesquisa metodológica",
            "uso de artefatos treinados e avaliados pela equipe responsável",
        ),
        benefits=(
            "Cobriu as 37 ocupações no benchmark do vocabulário 2.",
            "Aprende relações categóricas e numéricas a partir da calibração.",
            "Foi mais robusta que a GAN simples nos experimentos atuais.",
            "Permite avaliar trade-offs entre flexibilidade estatística e custo computacional.",
        ),
        limitations=(
            "Exige treinamento prévio.",
            "Possui maior custo computacional.",
            "É sensível a hiperparâmetros e versão de backend.",
            "Pode gerar combinações brutas inválidas.",
            "O resultado final depende parcialmente do pós-processamento.",
            "Pode apresentar risco de memorização quando treinada com dados sensíveis.",
        ),
        realism_profile="Boa cobertura categórica observada, com dependência relevante do treinamento e do pós-processamento.",
        privacy_considerations=(
            "Requer avaliação de memorização quando treinada com dados sensíveis.",
            "Menor distância estatística não significa maior privacidade.",
            "Artefatos serializados devem ser produzidos ou aprovados pela aplicação.",
        ),
        compliance_considerations=(
            "Manifestos registram versão, seed, schema, vocabulário e ambiente.",
            "Uso institucional exige governança sobre base de treinamento, finalidade e aprovação do artefato.",
            "Não constitui certificação de conformidade.",
        ),
        requires_training=True,
        requires_saved_artifact=True,
        capacity_notes=(
            "Maior treinamento concluído observado: 4.800.000 registros. "
            "Primeira falha por recursos: 6.400.000 registros. O limite absoluto não foi determinado."
        ),
        benchmark_notes=(
            "No benchmark do vocabulário 2, cobriu 100% das 37 ocupações e teve cerca de 91,3% "
            "de validade bruta entre escolaridade e ocupação, chegando a 100% no resultado final."
        ),
    ),
    ModelCatalogEntry(
        name="simple_gan",
        label="GAN Simples — Experimental",
        category="Baseline Neural Acadêmico",
        status="Experimental",
        recommended=False,
        experimental=True,
        complexity_level="Média",
        short_description="GAN tabular densa simples, mantida principalmente para comparação metodológica.",
        detailed_description=(
            "Baseline neural acadêmico baseado em uma GAN densa. Foi mantido para comparação "
            "metodológica, mas apresentou limitações de qualidade e capacidade nos experimentos realizados."
        ),
        simple_summary=(
            "É uma rede neural adversarial simples usada para comparar o projeto com um baseline neural, "
            "não como opção principal de geração."
        ),
        technical_summary=(
            "Usa codificação tabular, gerador denso, discriminador denso e treinamento adversarial em Keras. "
            "Tem dificuldade maior com muitas categorias e dependências semânticas."
        ),
        objective="Servir como baseline acadêmico para comparar ganhos e limitações de modelos neurais simples.",
        generated_data="Amostra bruta de colunas-base que passa por normalização, pós-processamento e validação.",
        recommended_use_cases=(
            "estudos acadêmicos sobre GANs",
            "comparação de arquiteturas",
            "análise de colapso categórico",
            "experimentos metodológicos controlados",
        ),
        benefits=(
            "Implementação simples e didática.",
            "Permite observar limitações de uma GAN densa convencional.",
            "Útil para comparar contra `programmatic` e `ctgan`.",
        ),
        limitations=(
            "Modelo experimental — não recomendado como opção principal.",
            "Exige artefato previamente treinado.",
            "Apresentou colapso categórico no vocabulário 2.",
            "Teve baixa cobertura de ocupações.",
            "Mostrou instabilidade entre seeds.",
            "Falha técnica observada em treinamento de 200.000 registros.",
        ),
        realism_profile="Adequado como baseline, mas com baixa robustez categórica observada.",
        privacy_considerations=(
            "Também requer avaliação de memorização quando treinada com dados sensíveis.",
            "Baixa diversidade pode distorcer métricas de risco.",
            "Não deve ser promovida automaticamente como artefato padrão.",
        ),
        compliance_considerations=(
            "Útil para documentação metodológica de comparação.",
            "Não é recomendada como opção institucional padrão.",
            "Não substitui validação, quality gates ou aprovação formal.",
        ),
        requires_training=True,
        requires_saved_artifact=True,
        capacity_notes=(
            "Maior sucesso observado: 100.000 registros de treinamento. "
            "Primeira falha técnica observada: 200.000 registros."
        ),
        benchmark_notes=(
            "No benchmark do vocabulário 2, apresentou baixa cobertura de ocupações e forte concentração "
            "em poucas categorias, permanecendo como baseline experimental."
        ),
    ),
)


def model_catalog() -> tuple[ModelCatalogEntry, ...]:
    """Retorna o catálogo de modelos em ordem de apresentação."""
    return MODEL_CATALOG


def model_catalog_by_name() -> dict[str, ModelCatalogEntry]:
    """Retorna o catálogo de modelos indexado pelo identificador técnico."""
    return {entry.name: entry for entry in MODEL_CATALOG}
