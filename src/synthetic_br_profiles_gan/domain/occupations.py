"""Catálogo sintético de ocupações e regras contextuais associadas."""

from __future__ import annotations

from dataclasses import dataclass


EDUCATION_FUNDAMENTAL = "Fundamental"
EDUCATION_HIGH_SCHOOL = "Ensino Médio"
EDUCATION_INCOMPLETE_HIGHER = "Superior Incompleto"
EDUCATION_HIGHER = "Superior Completo"
EDUCATION_POSTGRAD = "Pós-graduação"

ALL_EDUCATION_LEVELS = (
    EDUCATION_FUNDAMENTAL,
    EDUCATION_HIGH_SCHOOL,
    EDUCATION_INCOMPLETE_HIGHER,
    EDUCATION_HIGHER,
    EDUCATION_POSTGRAD,
)

FROM_HIGH_SCHOOL = (
    EDUCATION_HIGH_SCHOOL,
    EDUCATION_INCOMPLETE_HIGHER,
    EDUCATION_HIGHER,
    EDUCATION_POSTGRAD,
)

FROM_INCOMPLETE_HIGHER = (
    EDUCATION_INCOMPLETE_HIGHER,
    EDUCATION_HIGHER,
    EDUCATION_POSTGRAD,
)

HIGHER_AND_POSTGRAD = (EDUCATION_HIGHER, EDUCATION_POSTGRAD)


@dataclass(frozen=True)
class OccupationProfile:
    """Perfil de uma ocupação sintética usada na calibração e validação."""

    name: str
    group: str
    allowed_education: tuple[str, ...]
    minimum_age: int
    maximum_age: int | None
    income_multiplier: float
    sampling_weight: float
    description: str
    income_variability: float = 1.0


OCCUPATION_CATALOG: tuple[OccupationProfile, ...] = (
    OccupationProfile(
        "Estudante",
        "Educação e início de carreira",
        FROM_INCOMPLETE_HIGHER,
        18,
        None,
        0.45,
        0.72,
        "Pessoa em atividade educacional, geralmente com renda própria menor ou parcial.",
    ),
    OccupationProfile(
        "Estagiário",
        "Educação e início de carreira",
        FROM_INCOMPLETE_HIGHER,
        18,
        35,
        0.55,
        0.62,
        "Ocupação de entrada associada a formação em andamento e menor experiência profissional.",
    ),
    OccupationProfile(
        "Serviços Gerais",
        "Serviços operacionais",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        0.72,
        1.05,
        "Atividade operacional ampla em serviços de apoio.",
    ),
    OccupationProfile(
        "Atendente",
        "Comércio e atendimento",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        0.78,
        1.10,
        "Atendimento ao público em comércio, serviços ou operações presenciais.",
    ),
    OccupationProfile(
        "Operador de Caixa",
        "Comércio e atendimento",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        0.80,
        0.90,
        "Atividade de operação de caixa em comércio ou serviços.",
    ),
    OccupationProfile(
        "Recepcionista",
        "Comércio e atendimento",
        FROM_HIGH_SCHOOL,
        18,
        None,
        0.86,
        0.78,
        "Atendimento, organização de agendas e apoio administrativo inicial.",
    ),
    OccupationProfile(
        "Vendedor",
        "Comércio e atendimento",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        0.95,
        1.05,
        "Atividade comercial em vendas presenciais, telefônicas ou digitais.",
    ),
    OccupationProfile(
        "Auxiliar Administrativo",
        "Administração",
        FROM_HIGH_SCHOOL,
        18,
        None,
        0.90,
        0.86,
        "Apoio administrativo, organização de documentos e rotinas de escritório.",
    ),
    OccupationProfile(
        "Motorista",
        "Transporte e ofícios",
        ALL_EDUCATION_LEVELS,
        21,
        None,
        1.00,
        0.80,
        "Atividade de transporte de pessoas, produtos ou cargas.",
    ),
    OccupationProfile(
        "Entregador",
        "Transporte e ofícios",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        0.90,
        0.74,
        "Atividade de entrega urbana ou regional de mercadorias.",
    ),
    OccupationProfile(
        "Agricultor",
        "Transporte e ofícios",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        0.95,
        0.62,
        "Atividade rural de produção agrícola familiar ou comercial.",
    ),
    OccupationProfile(
        "Pedreiro",
        "Transporte e ofícios",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        1.05,
        0.62,
        "Ofício associado à construção civil e manutenção predial.",
    ),
    OccupationProfile(
        "Eletricista",
        "Técnico e manutenção",
        FROM_HIGH_SCHOOL,
        18,
        None,
        1.12,
        0.50,
        "Ofício técnico de instalação, manutenção e reparo elétrico.",
    ),
    OccupationProfile(
        "Mecânico",
        "Técnico e manutenção",
        FROM_HIGH_SCHOOL,
        18,
        None,
        1.12,
        0.50,
        "Ofício técnico de manutenção e reparo mecânico.",
    ),
    OccupationProfile(
        "Técnico",
        "Técnico e manutenção",
        FROM_HIGH_SCHOOL,
        18,
        None,
        1.05,
        0.72,
        "Categoria técnica sintética preservada para compatibilidade com o baseline legado.",
    ),
    OccupationProfile(
        "Técnico de Informática",
        "Técnico e manutenção",
        FROM_HIGH_SCHOOL,
        18,
        None,
        1.15,
        0.62,
        "Atividade técnica de suporte, manutenção e configuração de informática.",
    ),
    OccupationProfile(
        "Técnico de Enfermagem",
        "Saúde",
        FROM_HIGH_SCHOOL,
        18,
        None,
        1.12,
        0.52,
        "Atividade técnica em cuidados de saúde e apoio à equipe de enfermagem.",
    ),
    OccupationProfile(
        "Professor",
        "Educação e saúde",
        HIGHER_AND_POSTGRAD,
        22,
        None,
        1.25,
        0.55,
        "Atividade de docência em contextos educacionais variados.",
    ),
    OccupationProfile(
        "Enfermeiro",
        "Educação e saúde",
        HIGHER_AND_POSTGRAD,
        22,
        None,
        1.45,
        0.42,
        "Profissional de saúde com formação superior em enfermagem.",
    ),
    OccupationProfile(
        "Assistente Social",
        "Educação e saúde",
        HIGHER_AND_POSTGRAD,
        22,
        None,
        1.20,
        0.32,
        "Profissional de apoio social, orientação e acompanhamento de pessoas e famílias.",
    ),
    OccupationProfile(
        "Designer",
        "Tecnologia e criação",
        FROM_INCOMPLETE_HIGHER,
        18,
        None,
        1.20,
        0.45,
        "Atividade criativa ligada a design gráfico, produto, interfaces ou comunicação visual.",
    ),
    OccupationProfile(
        "Desenvolvedor de Software",
        "Tecnologia e criação",
        FROM_INCOMPLETE_HIGHER,
        18,
        None,
        1.65,
        0.48,
        "Atividade de desenvolvimento, manutenção e evolução de sistemas de software.",
    ),
    OccupationProfile(
        "Analista",
        "Administração e tecnologia",
        FROM_INCOMPLETE_HIGHER,
        20,
        None,
        1.35,
        0.60,
        "Categoria analítica sintética preservada para compatibilidade com o baseline legado.",
    ),
    OccupationProfile(
        "Analista de Dados",
        "Administração e tecnologia",
        FROM_INCOMPLETE_HIGHER,
        20,
        None,
        1.70,
        0.42,
        "Atividade de análise, tratamento e interpretação de dados.",
    ),
    OccupationProfile(
        "Analista Administrativo",
        "Administração e tecnologia",
        FROM_INCOMPLETE_HIGHER,
        20,
        None,
        1.25,
        0.48,
        "Atividade de análise e organização de processos administrativos.",
    ),
    OccupationProfile(
        "Contador",
        "Profissões regulamentadas",
        HIGHER_AND_POSTGRAD,
        22,
        None,
        1.45,
        0.32,
        "Profissional de contabilidade e rotinas fiscais, financeiras e patrimoniais.",
    ),
    OccupationProfile(
        "Engenheiro",
        "Profissões regulamentadas",
        HIGHER_AND_POSTGRAD,
        23,
        None,
        1.85,
        0.34,
        "Profissional de engenharia em áreas técnicas, industriais ou de infraestrutura.",
    ),
    OccupationProfile(
        "Arquiteto",
        "Profissões regulamentadas",
        HIGHER_AND_POSTGRAD,
        23,
        None,
        1.65,
        0.24,
        "Profissional de arquitetura, urbanismo, projetos e planejamento de espaços.",
    ),
    OccupationProfile(
        "Advogado",
        "Profissões regulamentadas",
        HIGHER_AND_POSTGRAD,
        23,
        None,
        1.65,
        0.32,
        "Profissional do direito em atuação consultiva, contenciosa ou institucional.",
    ),
    OccupationProfile(
        "Dentista",
        "Profissões regulamentadas",
        HIGHER_AND_POSTGRAD,
        24,
        None,
        2.10,
        0.20,
        "Profissional de odontologia e cuidados de saúde bucal.",
    ),
    OccupationProfile(
        "Médico",
        "Profissões regulamentadas",
        HIGHER_AND_POSTGRAD,
        25,
        None,
        2.70,
        0.22,
        "Profissional de medicina em atuação clínica, hospitalar ou especializada.",
    ),
    OccupationProfile(
        "Coordenador",
        "Gestão",
        HIGHER_AND_POSTGRAD,
        23,
        None,
        1.65,
        0.36,
        "Atividade de coordenação de equipes, processos ou áreas de trabalho.",
    ),
    OccupationProfile(
        "Gerente",
        "Gestão",
        HIGHER_AND_POSTGRAD,
        25,
        None,
        2.10,
        0.28,
        "Atividade gerencial com responsabilidade por equipes, orçamento ou resultados.",
    ),
    OccupationProfile(
        "Diretor",
        "Gestão",
        (EDUCATION_POSTGRAD,),
        30,
        None,
        2.80,
        0.10,
        "Atividade executiva sintética associada a maior responsabilidade organizacional.",
    ),
    OccupationProfile(
        "Autônomo",
        "Empreendedorismo",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        1.05,
        0.86,
        "Trabalho independente em atividades variadas, com maior variabilidade de renda.",
        income_variability=1.22,
    ),
    OccupationProfile(
        "Microempreendedor",
        "Empreendedorismo",
        ALL_EDUCATION_LEVELS,
        18,
        None,
        1.20,
        0.74,
        "Atividade econômica própria de pequeno porte, com maior variabilidade de renda.",
        income_variability=1.28,
    ),
    OccupationProfile(
        "Aposentado",
        "Aposentadoria",
        ALL_EDUCATION_LEVELS,
        50,
        None,
        0.85,
        0.35,
        "Pessoa sintética fora da atividade laboral principal, com renda de aposentadoria ou fontes similares.",
    ),
)


def occupation_categories() -> list[str]:
    """Retorna os nomes canônicos das ocupações sintéticas."""
    return [profile.name for profile in OCCUPATION_CATALOG]


def occupation_by_name() -> dict[str, OccupationProfile]:
    """Retorna o catálogo indexado pelo nome canônico da ocupação."""
    return {profile.name: profile for profile in OCCUPATION_CATALOG}


def get_occupation_profile(name: str) -> OccupationProfile | None:
    """Busca uma ocupação pelo nome canônico."""
    return occupation_by_name().get(str(name))


def is_occupation_compatible(occupation: str, education: str, age: int) -> bool:
    """Verifica compatibilidade estrutural entre ocupação, escolaridade e idade."""
    profile = get_occupation_profile(occupation)
    if profile is None:
        return False
    if str(education) not in profile.allowed_education:
        return False
    numeric_age = int(age)
    if numeric_age < profile.minimum_age:
        return False
    if profile.maximum_age is not None and numeric_age > profile.maximum_age:
        return False
    return True


def income_multiplier_for_occupation(occupation: str) -> float:
    """Retorna o multiplicador de renda associado a uma ocupação."""
    profile = get_occupation_profile(occupation)
    return 1.0 if profile is None else float(profile.income_multiplier)


def income_variability_for_occupation(occupation: str) -> float:
    """Retorna o fator de variabilidade de renda associado a uma ocupação."""
    profile = get_occupation_profile(occupation)
    return 1.0 if profile is None else float(profile.income_variability)


def eligible_occupation_profiles(education: str, age: int) -> tuple[OccupationProfile, ...]:
    """Lista ocupações elegíveis para uma escolaridade e idade."""
    return tuple(
        profile
        for profile in OCCUPATION_CATALOG
        if is_occupation_compatible(profile.name, education, int(age))
    )


def occupation_sampling_weights(education: str, age: int) -> dict[str, float]:
    """Calcula pesos contextuais de sorteio para ocupações elegíveis."""
    weights: dict[str, float] = {}
    for profile in eligible_occupation_profiles(education, int(age)):
        weight = float(profile.sampling_weight)
        if profile.name == "Estudante":
            if age <= 24:
                weight *= 2.4
            elif age <= 30:
                weight *= 1.4
            elif age >= 45:
                weight *= 0.25
        elif profile.name == "Estagiário":
            if age <= 24:
                weight *= 2.2
            elif age <= 30:
                weight *= 1.35
            else:
                weight *= 0.45
        elif profile.name == "Aposentado":
            if age >= 65:
                weight *= 7.0
            elif age >= 60:
                weight *= 4.0
            else:
                weight *= 0.35
        elif profile.name in {"Coordenador", "Gerente"}:
            if age >= 35:
                weight *= 1.45
            elif age < 28:
                weight *= 0.45
        elif profile.name == "Diretor":
            if age >= 45:
                weight *= 2.3
            elif age < 35:
                weight *= 0.30
        if education == EDUCATION_POSTGRAD and profile.name in {
            "Professor",
            "Analista de Dados",
            "Engenheiro",
            "Médico",
            "Dentista",
            "Advogado",
            "Coordenador",
            "Gerente",
            "Diretor",
        }:
            weight *= 1.75
        weights[profile.name] = max(weight, 0.0)
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {name: weight / total for name, weight in weights.items()}
