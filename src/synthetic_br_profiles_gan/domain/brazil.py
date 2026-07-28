"""Pequenas tabelas explícitas de referência geográfica brasileira e DDD.

As tabelas são intencionalmente locais e estáticas. Elas não são consultas a
cadastros oficiais e não devem ser tratadas como prova de que registros gerados
correspondem a pessoas, telefones, endereços ou documentos reais.
"""

from __future__ import annotations

REGIONS = ("Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul")

STATE_REGION: dict[str, str] = {
    "AC": "Norte",
    "AL": "Nordeste",
    "AP": "Norte",
    "AM": "Norte",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "DF": "Centro-Oeste",
    "ES": "Sudeste",
    "GO": "Centro-Oeste",
    "MA": "Nordeste",
    "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "MG": "Sudeste",
    "PA": "Norte",
    "PB": "Nordeste",
    "PR": "Sul",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RJ": "Sudeste",
    "RN": "Nordeste",
    "RS": "Sul",
    "RO": "Norte",
    "RR": "Norte",
    "SC": "Sul",
    "SP": "Sudeste",
    "SE": "Nordeste",
    "TO": "Norte",
}

REGION_STATES: dict[str, tuple[str, ...]] = {
    region: tuple(state for state, state_region in STATE_REGION.items() if state_region == region)
    for region in REGIONS
}

STATE_MUNICIPALITIES: dict[str, tuple[str, ...]] = {
    "AC": ("Rio Branco", "Cruzeiro do Sul", "Sena Madureira"),
    "AL": ("Maceió", "Arapiraca", "Palmeira dos Índios"),
    "AP": ("Macapá", "Santana", "Laranjal do Jari"),
    "AM": ("Manaus", "Parintins", "Itacoatiara"),
    "BA": ("Salvador", "Feira de Santana", "Vitória da Conquista"),
    "CE": ("Fortaleza", "Juazeiro do Norte", "Sobral"),
    "DF": ("Brasília", "Ceilândia", "Taguatinga"),
    "ES": ("Vitória", "Vila Velha", "Serra"),
    "GO": ("Goiânia", "Anápolis", "Aparecida de Goiânia"),
    "MA": ("São Luís", "Imperatriz", "Caxias"),
    "MT": ("Cuiabá", "Várzea Grande", "Rondonópolis"),
    "MS": ("Campo Grande", "Dourados", "Três Lagoas"),
    "MG": ("Belo Horizonte", "Uberlândia", "Contagem"),
    "PA": ("Belém", "Ananindeua", "Santarém"),
    "PB": ("João Pessoa", "Campina Grande", "Patos"),
    "PR": ("Curitiba", "Londrina", "Maringá"),
    "PE": ("Recife", "Olinda", "Caruaru"),
    "PI": ("Teresina", "Parnaíba", "Picos"),
    "RJ": ("Rio de Janeiro", "Niterói", "Petrópolis"),
    "RN": ("Natal", "Mossoró", "Parnamirim"),
    "RS": ("Porto Alegre", "Caxias do Sul", "Pelotas"),
    "RO": ("Porto Velho", "Ji-Paraná", "Ariquemes"),
    "RR": ("Boa Vista", "Rorainópolis", "Caracaraí"),
    "SC": ("Florianópolis", "Joinville", "Blumenau"),
    "SP": ("São Paulo", "Campinas", "Santos"),
    "SE": ("Aracaju", "Nossa Senhora do Socorro", "Lagarto"),
    "TO": ("Palmas", "Araguaína", "Gurupi"),
}

STATE_DDDS: dict[str, tuple[int, ...]] = {
    "AC": (68,),
    "AL": (82,),
    "AP": (96,),
    "AM": (92, 97),
    "BA": (71, 73, 74, 75, 77),
    "CE": (85, 88),
    "DF": (61,),
    "ES": (27, 28),
    "GO": (62, 64),
    "MA": (98, 99),
    "MT": (65, 66),
    "MS": (67,),
    "MG": (31, 32, 33, 34, 35, 37, 38),
    "PA": (91, 93, 94),
    "PB": (83,),
    "PR": (41, 42, 43, 44, 45, 46),
    "PE": (81, 87),
    "PI": (86, 89),
    "RJ": (21, 22, 24),
    "RN": (84,),
    "RS": (51, 53, 54, 55),
    "RO": (69,),
    "RR": (95,),
    "SC": (47, 48, 49),
    "SP": (11, 12, 13, 14, 15, 16, 17, 18, 19),
    "SE": (79,),
    "TO": (63,),
}


def states_for_region(region: str) -> tuple[str, ...]:
    """Retorna os estados conhecidos de uma região."""
    return REGION_STATES.get(str(region), ())


def region_for_state(state: str) -> str | None:
    """Retorna a região associada à sigla de um estado."""
    return STATE_REGION.get(str(state))


def municipalities_for_state(state: str) -> tuple[str, ...]:
    """Retorna os municípios configurados para um estado."""
    return STATE_MUNICIPALITIES.get(str(state), ())


def ddds_for_state(state: str) -> tuple[int, ...]:
    """Retorna códigos DDD compatíveis com um estado."""
    return STATE_DDDS.get(str(state), ())


def all_ddds() -> tuple[int, ...]:
    """Retorna todos os códigos DDD conhecidos na tabela local de referência."""
    return tuple(sorted({ddd for ddds in STATE_DDDS.values() for ddd in ddds}))
