"""Validacoes estruturais para documentos e campos sinteticos brasileiros."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

CPF_FORMAT_RE = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
RG_FORMAT_RE = re.compile(r"^\d{2}\.\d{3}\.\d{3}-\d$")
PHONE_FORMAT_RE = re.compile(r"^\(\d{2}\) \d{5}-\d{4}$")
CNH_FORMAT_RE = re.compile(r"^\d{11}$")
TITULO_ELEITOR_FORMAT_RE = re.compile(r"^\d{4} \d{4} \d{2} \d{2}$")


def somente_digitos(valor: object) -> str:
    """Remove caracteres nao numericos de um valor."""
    return re.sub(r"\D", "", str(valor))


def validar_formato_cpf(valor: object) -> bool:
    """Valida apenas a mascara 000.000.000-00."""
    return bool(CPF_FORMAT_RE.match(str(valor)))


def validar_cpf(valor: object) -> bool:
    """Valida formato, tamanho e digitos verificadores do CPF."""
    cpf = somente_digitos(valor)
    if len(cpf) != 11 or cpf == cpf[0] * 11:
        return False

    def calc_digito(digs: str) -> str:
        soma = sum(int(d) * w for d, w in zip(digs, range(len(digs) + 1, 1, -1)))
        resto = 11 - (soma % 11)
        return "0" if resto > 9 else str(resto)

    primeiro = calc_digito(cpf[:9])
    segundo = calc_digito(cpf[:9] + primeiro)
    return cpf[-2:] == primeiro + segundo


def validar_formato_rg(valor: object) -> bool:
    """Valida o formato 00.000.000-0 usado pelo gerador."""
    return bool(RG_FORMAT_RE.match(str(valor)))


def validar_telefone(valor: object) -> bool:
    """Valida telefone no formato (00) 90000-0000."""
    return bool(PHONE_FORMAT_RE.match(str(valor)))


def extrair_ddd(valor: object) -> int | None:
    """Extrai o DDD de um telefone no formato do projeto."""
    match = PHONE_FORMAT_RE.match(str(valor))
    if not match:
        return None
    return int(str(valor)[1:3])


def validar_cnh(valor: object) -> bool:
    """Valida a CNH conforme a regra local usada pelo gerador."""
    cnh = somente_digitos(valor)
    if not CNH_FORMAT_RE.match(cnh):
        return False
    number = [int(digit) for digit in cnh[:9]]
    total = sum((9 - index) * number[index] for index in range(9))
    digit_1 = total % 11
    digit_1 = 0 if digit_1 >= 10 else digit_1

    total = sum((index + 1) * number[index] for index in range(9))
    digit_2 = total % 11
    digit_2 = 0 if digit_2 >= 10 else digit_2
    return cnh[-2:] == f"{digit_1}{digit_2}"


def validar_titulo_eleitor(valor: object) -> bool:
    """Valida o titulo de eleitor conforme a regra local usada pelo gerador."""
    text = str(valor)
    if not TITULO_ELEITOR_FORMAT_RE.match(text):
        return False
    digits = somente_digitos(text)
    number = digits[:8]
    uf_code = digits[8:10]
    digit_1 = sum(int(number[index]) * (9 - index) for index in range(8)) % 11
    digit_1 = 0 if digit_1 == 10 else digit_1
    digit_2 = sum(int(number[index]) * (8 - index) for index in range(8)) + digit_1 * 9 + int(uf_code) * 10
    digit_2 = digit_2 % 11
    digit_2 = 0 if digit_2 == 10 else digit_2
    return digits[-2:] == f"{digit_1}{digit_2}"


def checar_unicidade(df: pd.DataFrame, colunas: Iterable[str]) -> dict[str, int]:
    """Conta duplicidades internas nas colunas informadas."""
    duplicidades: dict[str, int] = {}
    for coluna in colunas:
        if coluna in df.columns:
            duplicidades[coluna] = int(df[coluna].duplicated().sum())
    return duplicidades


def _contar_invalidos(df: pd.DataFrame, coluna: str, validator) -> int:
    if coluna not in df.columns:
        return len(df)
    return int((~df[coluna].astype(str).apply(validator)).sum())


def avaliar_regras_final(df: pd.DataFrame) -> dict:
    """Avalia formato, digitos de CPF e duplicidades do dataset final."""
    n = len(df)
    contagens = {
        "cpf_formato_invalido": _contar_invalidos(df, "CPF", validar_formato_cpf),
        "cpf_digito_invalido": _contar_invalidos(df, "CPF", validar_cpf),
        "cnh_digito_invalido": _contar_invalidos(df, "CNH", validar_cnh),
        "titulo_digito_invalido": _contar_invalidos(df, "Titulo_Eleitor", validar_titulo_eleitor),
        "rg_formato_invalido": _contar_invalidos(df, "RG", validar_formato_rg),
        "tel_formato_invalido": _contar_invalidos(df, "Telefone", validar_telefone),
    }

    duplicidades = checar_unicidade(
        df,
        ["CPF", "CNH", "RG", "Titulo_Eleitor", "Telefone"],
    )
    contagens.update(
        {
            "cpf_duplicado": duplicidades.get("CPF", 0),
            "cnh_duplicada": duplicidades.get("CNH", 0),
            "rg_duplicado": duplicidades.get("RG", 0),
            "titulo_duplicado": duplicidades.get("Titulo_Eleitor", 0),
            "telefone_duplicado": duplicidades.get("Telefone", 0),
        }
    )

    if "Idade" in df.columns:
        contagens["idade_fora_faixa"] = int((~df["Idade"].between(18, 65)).sum())

    total_erros = sum(contagens.values())
    taxa_erros_por_registro = 0.0 if n == 0 else total_erros / n
    return {
        "n_registros": n,
        "contagens": contagens,
        "total_erros": int(total_erros),
        "taxa_erros_por_registro": float(taxa_erros_por_registro),
        "taxa_conformidade_aproximada": float(1 - min(1, taxa_erros_por_registro)),
    }


def avaliar_regras_bruto(df: pd.DataFrame) -> dict:
    """Avalia regras numericas simples antes do pos-processamento final."""
    n = len(df)
    violacoes = {}

    if "Idade" in df.columns:
        idade = df["Idade"]
        violacoes["idade_menor_18"] = int((idade < 18).sum())
        violacoes["idade_maior_65"] = int((idade > 65).sum())
        if "Renda" in df.columns:
            violacoes["renda_alta_para_menor"] = int(((idade < 18) & (df["Renda"] > 8000)).sum())
        else:
            violacoes["renda_alta_para_menor"] = 0
    else:
        violacoes["idade_menor_18"] = 0
        violacoes["idade_maior_65"] = 0
        violacoes["renda_alta_para_menor"] = 0

    if "Renda" in df.columns:
        renda = df["Renda"]
        violacoes["renda_fora_faixa"] = int((~renda.between(1200, 25000)).sum())
    else:
        violacoes["renda_fora_faixa"] = 0

    if "Sexo" in df.columns:
        sexo = df["Sexo"]
        violacoes["sexo_fora_0_1"] = int((~sexo.between(0, 1)).sum())
    else:
        violacoes["sexo_fora_0_1"] = 0

    total = sum(violacoes.values())
    taxa_violacoes = 0.0 if n == 0 else total / n
    return {
        "n_candidatos_avaliados": n,
        "contagens": violacoes,
        "total_violacoes": int(total),
        "taxa_violacoes": float(taxa_violacoes),
        "taxa_conformidade": float(1 - min(1, taxa_violacoes)),
    }

