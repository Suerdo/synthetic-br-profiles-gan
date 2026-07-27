"""Small, explicit Brazilian geography and DDD reference tables.

The tables are intentionally local and static. They are not official registry
queries and must not be treated as proof that generated records correspond to
real people, phones, addresses, or documents.
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
    "AL": ("Maceio", "Arapiraca", "Palmeira dos Indios"),
    "AP": ("Macapa", "Santana", "Laranjal do Jari"),
    "AM": ("Manaus", "Parintins", "Itacoatiara"),
    "BA": ("Salvador", "Feira de Santana", "Vitoria da Conquista"),
    "CE": ("Fortaleza", "Juazeiro do Norte", "Sobral"),
    "DF": ("Brasilia", "Ceilandia", "Taguatinga"),
    "ES": ("Vitoria", "Vila Velha", "Serra"),
    "GO": ("Goiania", "Anapolis", "Aparecida de Goiania"),
    "MA": ("Sao Luis", "Imperatriz", "Caxias"),
    "MT": ("Cuiaba", "Varzea Grande", "Rondonopolis"),
    "MS": ("Campo Grande", "Dourados", "Tres Lagoas"),
    "MG": ("Belo Horizonte", "Uberlandia", "Contagem"),
    "PA": ("Belem", "Ananindeua", "Santarem"),
    "PB": ("Joao Pessoa", "Campina Grande", "Patos"),
    "PR": ("Curitiba", "Londrina", "Maringa"),
    "PE": ("Recife", "Olinda", "Caruaru"),
    "PI": ("Teresina", "Parnaiba", "Picos"),
    "RJ": ("Rio de Janeiro", "Niteroi", "Petropolis"),
    "RN": ("Natal", "Mossoro", "Parnamirim"),
    "RS": ("Porto Alegre", "Caxias do Sul", "Pelotas"),
    "RO": ("Porto Velho", "Ji-Parana", "Ariquemes"),
    "RR": ("Boa Vista", "Rorainopolis", "Caracarai"),
    "SC": ("Florianopolis", "Joinville", "Blumenau"),
    "SP": ("Sao Paulo", "Campinas", "Santos"),
    "SE": ("Aracaju", "Nossa Senhora do Socorro", "Lagarto"),
    "TO": ("Palmas", "Araguaina", "Gurupi"),
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
    """Return the known states for a region."""
    return REGION_STATES.get(str(region), ())


def region_for_state(state: str) -> str | None:
    """Return the region associated with a state abbreviation."""
    return STATE_REGION.get(str(state))


def municipalities_for_state(state: str) -> tuple[str, ...]:
    """Return the configured municipalities for a state."""
    return STATE_MUNICIPALITIES.get(str(state), ())


def ddds_for_state(state: str) -> tuple[int, ...]:
    """Return DDD codes compatible with a state."""
    return STATE_DDDS.get(str(state), ())


def all_ddds() -> tuple[int, ...]:
    """Return all known DDD codes in the local reference table."""
    return tuple(sorted({ddd for ddds in STATE_DDDS.values() for ddd in ddds}))
