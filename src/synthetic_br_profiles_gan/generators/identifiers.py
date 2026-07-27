"""Generators for fictitious Brazilian identifiers and phone numbers.

The generated values are local synthetic artifacts. The module never queries
public or private registries and must not be used to claim that a document or
phone number belongs, or does not belong, to a real person.
"""

from __future__ import annotations

import random

from synthetic_br_profiles_gan.domain.brazil import STATE_DDDS, all_ddds

VALID_DDDS = all_ddds()


def _random_source(rng: random.Random | None = None) -> random.Random:
    return rng if rng is not None else random


def gerar_cpf(rng: random.Random | None = None) -> str:
    """Generate a masked CPF with mathematically valid check digits."""
    random_source = _random_source(rng)

    def calc_digit(digits: list[str]) -> str:
        total = sum(int(digit) * weight for digit, weight in zip(digits, range(len(digits) + 1, 1, -1)))
        remainder = 11 - (total % 11)
        return "0" if remainder > 9 else str(remainder)

    base = [str(random_source.randint(0, 9)) for _ in range(9)]
    base.append(calc_digit(base))
    base.append(calc_digit(base))
    return f"{''.join(base[:3])}.{''.join(base[3:6])}.{''.join(base[6:9])}-{''.join(base[9:])}"


def gerar_cnh(rng: random.Random | None = None) -> str:
    """Generate an 11-digit CNH-like number with local check digits."""
    random_source = _random_source(rng)
    number = [random_source.randint(0, 9) for _ in range(9)]

    total = sum((9 - index) * number[index] for index in range(9))
    digit_1 = total % 11
    digit_1 = 0 if digit_1 >= 10 else digit_1

    total = sum((index + 1) * number[index] for index in range(9))
    digit_2 = total % 11
    digit_2 = 0 if digit_2 >= 10 else digit_2

    return "".join(str(value) for value in number) + str(digit_1) + str(digit_2)


def gerar_rg(rng: random.Random | None = None) -> str:
    """Generate a fictitious RG in the project format 00.000.000-0."""
    random_source = _random_source(rng)
    number = [str(random_source.randint(0, 9)) for _ in range(8)]
    digit = random_source.randint(0, 9)
    return f"{''.join(number[:2])}.{''.join(number[2:5])}.{''.join(number[5:8])}-{digit}"


def _titulo_eleitor_digits(number: str, uf_code: str) -> str:
    first = sum(int(number[index]) * (9 - index) for index in range(8)) % 11
    first = 0 if first == 10 else first
    second = sum(int(number[index]) * (8 - index) for index in range(8)) + first * 9 + int(uf_code) * 10
    second = second % 11
    second = 0 if second == 10 else second
    return str(first) + str(second)


def gerar_titulo_eleitor(rng: random.Random | None = None, uf_codigo: int | None = None) -> str:
    """Generate a fictitious voter title with the local check-digit rule."""
    random_source = _random_source(rng)
    number = "".join(str(random_source.randint(0, 9)) for _ in range(8))
    uf = f"{uf_codigo if uf_codigo is not None else random_source.randint(1, 28):02d}"
    digits = _titulo_eleitor_digits(number, uf)
    return f"{number[:4]} {number[4:]} {uf} {digits}"


def gerar_telefone(
    rng: random.Random | None = None,
    estado: str | None = None,
    ddd: int | None = None,
) -> str:
    """Generate a fictitious Brazilian mobile phone compatible with a state or DDD."""
    random_source = _random_source(rng)
    if ddd is None:
        ddd_pool = STATE_DDDS.get(str(estado), VALID_DDDS) if estado else VALID_DDDS
        ddd = random_source.choice(tuple(ddd_pool))
    prefix = random_source.randint(90000, 99999)
    suffix = random_source.randint(1000, 9999)
    return f"({int(ddd):02d}) {prefix}-{suffix}"
