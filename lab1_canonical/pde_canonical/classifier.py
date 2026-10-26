# -*- coding: utf-8 -*-
"""Задание A. По коэффициентам ДУЧП 2-го порядка определить:

  * тип уравнения (гиперболический / параболический / эллиптический);
  * замену переменных, приводящую уравнение к каноническому виду.

Работает для любого числа переменных n. Основан на характеристической
квадратичной форме (лекция 1, формулы 1-4):

    sum_{i,j} a_ij u_{x_i x_j} + F = 0      ->      Q(l) = sum_{i,j} a_ij l_i l_j

Метод Лагранжа даёт Q = sum beta_i mu_i^2 при mu = C*l. Полагая P = C^{-1},
замена переменных xi = P^T x приводит уравнение к виду sum beta_i u_{xi_i xi_i} + F = 0.

Тип определяется по набору beta (слайд 14):
  * все beta_i != 0 и не все одного знака -> гиперболический;
  * есть beta_i = 0                       -> параболический;
  * все beta_i != 0 и все одного знака    -> эллиптический.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from sympy import S, Symbol, simplify, symbols, sympify

from .exact import diag, is_zero, mat_mul, symmetrise, transpose
from .lagrange import reduce_quadratic_form

HYPERBOLIC = "hyperbolic"
PARABOLIC = "parabolic"
ELLIPTIC = "elliptic"

TYPE_RU = {
    HYPERBOLIC: "гиперболический",
    PARABOLIC: "параболический",
    ELLIPTIC: "эллиптический",
}

SUBTYPE_RU = {
    "normal": "обычный",
    "definite": "знакоопределённый",
    "ultrahyperbolic": "ультрагиперболический",
    "ultraparabolic": "ультрапараболический",
}

SUBSCRIPTS = "₁₂₃₄₅₆₇₈₉"


@dataclass
class ClassificationResult:
    A: List[List]
    variables: List
    beta: List
    C: List[List]
    P: List[List]
    type: str
    subtype: str
    new_variables: List
    substitution_text: List[str]
    canonical_text: str
    canonical_latex: str
    verified: bool
    steps: List[str] = field(default_factory=list)

    @property
    def n(self):
        return len(self.variables)

    @property
    def type_ru(self):
        return TYPE_RU[self.type]

    @property
    def subtype_ru(self):
        return SUBTYPE_RU.get(self.subtype, self.subtype)


def _sign(value):
    value = simplify(sympify(value))
    if value.is_zero:
        return 0
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    raise ValueError(f"не удалось определить знак beta = {value}")


def _as_matrix(coeffs):
    """Принимает матрицу или словарь {(i, j): a_ij}."""
    if isinstance(coeffs, dict):
        if not coeffs:
            raise ValueError("коэффициенты не заданы")
        n = max(max(i, j) for i, j in coeffs) + 1
        matrix = [[S.Zero] * n for _ in range(n)]
        for (i, j), value in coeffs.items():
            matrix[i][j] = sympify(value)
        return matrix
    matrix = [[sympify(v) for v in row] for row in coeffs]
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("матрица коэффициентов должна быть квадратной")
    return matrix


def _subscript(i):
    return SUBSCRIPTS[i - 1] if 1 <= i <= len(SUBSCRIPTS) else str(i)


def _classify_signature(beta):
    signs = [_sign(b) for b in beta]
    zeros = signs.count(0)
    positives = signs.count(1)
    negatives = signs.count(-1)

    if zeros == 0:
        if positives == 0 or negatives == 0:
            return ELLIPTIC, "definite"
        minority = min(positives, negatives)
        return HYPERBOLIC, "normal" if minority == 1 else "ultrahyperbolic"

    if positives > 0 and negatives > 0:
        return PARABOLIC, "ultraparabolic"
    return PARABOLIC, "normal"


def _canonical_text(beta, latex=False):
    terms = []
    for i, value in enumerate(beta, start=1):
        sign = _sign(value)
        if sign == 0:
            continue
        if latex:
            body = rf"u_{{\xi_{i}\xi_{i}}}"
        else:
            sub = _subscript(i)
            body = f"u_{{ξ{sub}ξ{sub}}}"

        magnitude = simplify(abs(sympify(value)))
        if magnitude != 1:
            body = f"{magnitude}·{body}" if not latex else rf"{magnitude} {body}"

        if not terms:
            terms.append(body if sign > 0 else f"-{body}")
        else:
            terms.append(f"{'+' if sign > 0 else '-'} {body}")

    tail = r" + \tilde F = 0" if latex else " + F̃ = 0"
    return " ".join(terms) + tail


def classify(coeffs, variables: Optional[Sequence] = None, normalize: bool = True):
    """Определяет тип ДУЧП и замену переменных xi = P^T x."""
    a = symmetrise(_as_matrix(coeffs))
    n = len(a)

    if all(is_zero(a[i][j]) for i in range(n) for j in range(n)):
        raise ValueError("старшая часть уравнения пуста: это не ДУЧП 2-го порядка")

    if variables is None:
        variables = list(symbols(f"x1:{n + 1}", real=True))
    else:
        variables = [sympify(v) for v in variables]
        if len(variables) != n:
            raise ValueError(
                f"переменных {len(variables)}, а порядок матрицы {n} — не совпадает"
            )

    reduction = reduce_quadratic_form(a, normalize=normalize)
    eq_type, subtype = _classify_signature(reduction.beta)

    # xi = P^T x
    pt = transpose(reduction.P)
    new_variables = [
        simplify(sum(pt[i][j] * variables[j] for j in range(n))) for i in range(n)
    ]
    substitution_text = [
        f"ξ{_subscript(i + 1)} = {new_variables[i]}" for i in range(n)
    ]

    # Критерий правильности: P^T A P = diag(beta).
    product = mat_mul(mat_mul(pt, a), reduction.P)
    target = diag(reduction.beta)
    verified = all(
        is_zero(product[i][j] - target[i][j]) for i in range(n) for j in range(n)
    )

    return ClassificationResult(
        A=a,
        variables=variables,
        beta=reduction.beta,
        C=reduction.C,
        P=reduction.P,
        type=eq_type,
        subtype=subtype,
        new_variables=new_variables,
        substitution_text=substitution_text,
        canonical_text=_canonical_text(reduction.beta),
        canonical_latex=_canonical_text(reduction.beta, latex=True),
        verified=verified,
        steps=reduction.steps,
    )
