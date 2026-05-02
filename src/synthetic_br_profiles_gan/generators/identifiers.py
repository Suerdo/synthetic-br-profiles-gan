"""Geradores de identificadores ficticios com formato brasileiro.

Os valores produzidos sao exclusivamente sinteticos e devem ser usados apenas
em ambientes controlados de teste, pesquisa, homologacao e experimentacao.
"""

from __future__ import annotations

import random

VALID_DDDS = (
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    24,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    37,
    38,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    51,
    53,
    54,
    55,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    71,
    73,
    74,
    75,
    77,
    79,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
)


def _random_source(rng: random.Random | None = None):
    return rng if rng is not None else random


def gerar_cpf(rng: random.Random | None = None) -> str:
    """Gera um CPF ficticio com mascara e digitos verificadores validos."""
    random_source = _random_source(rng)

    def calc_digito(digs: list[str]) -> str:
        soma = sum(int(d) * w for d, w in zip(digs, range(len(digs) + 1, 1, -1)))
        resto = 11 - (soma % 11)
        return "0" if resto > 9 else str(resto)

    numero = [str(random_source.randint(0, 9)) for _ in range(9)]
    numero.append(calc_digito(numero))
    numero.append(calc_digito(numero))
    return (
        f"{''.join(numero[:3])}."
        f"{''.join(numero[3:6])}."
        f"{''.join(numero[6:9])}-"
        f"{''.join(numero[9:])}"
    )


def gerar_cnh(rng: random.Random | None = None) -> str:
    """Gera um numero ficticio de CNH com 11 digitos."""
    random_source = _random_source(rng)
    numero = [random_source.randint(0, 9) for _ in range(9)]

    soma = sum((9 - i) * numero[i] for i in range(9))
    d1 = soma % 11
    d1 = 0 if d1 >= 10 else d1

    soma = sum((i + 1) * numero[i] for i in range(9))
    d2 = soma % 11
    d2 = 0 if d2 >= 10 else d2

    return "".join(map(str, numero)) + str(d1) + str(d2)


def gerar_rg(rng: random.Random | None = None) -> str:
    """Gera um RG ficticio no formato 00.000.000-0."""
    random_source = _random_source(rng)
    numero = [str(random_source.randint(0, 9)) for _ in range(8)]
    digito = random_source.randint(0, 9)
    return f"{''.join(numero[:2])}.{''.join(numero[2:5])}.{''.join(numero[5:8])}-{digito}"


def gerar_titulo_eleitor(rng: random.Random | None = None) -> str:
    """Gera titulo de eleitor ficticio com estrutura numerica plausivel."""
    random_source = _random_source(rng)

    def calc_dv(num: str, uf: str) -> str:
        d1 = sum(int(num[i]) * (9 - i) for i in range(8)) % 11
        d1 = 0 if d1 == 10 else d1
        d2 = sum(int(num[i]) * (8 - i) for i in range(8)) + d1 * 9 + int(uf) * 10
        d2 = d2 % 11
        d2 = 0 if d2 == 10 else d2
        return str(d1) + str(d2)

    numero = "".join(str(random_source.randint(0, 9)) for _ in range(8))
    uf = f"{random_source.randint(1, 28):02d}"
    dv = calc_dv(numero, uf)
    return f"{numero[:4]} {numero[4:]} {uf} {dv}"


def gerar_telefone(rng: random.Random | None = None) -> str:
    """Gera telefone celular ficticio no formato brasileiro."""
    random_source = _random_source(rng)
    ddd = random_source.choice(VALID_DDDS)
    prefixo = random_source.randint(90000, 99999)
    sufixo = random_source.randint(1000, 9999)
    return f"({ddd}) {prefixo}-{sufixo}"

