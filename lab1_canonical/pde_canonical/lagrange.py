# -*- coding: utf-8 -*-
"""Метод Лагранжа: приведение квадратичной формы к сумме квадратов.

Реализует схему из «Примеры к лекции 1.pdf», п.2:

    Q(l) = sum a_ij l_i l_j   --выделение полных квадратов-->   sum beta_i mu_i^2

Метод возвращает матрицу C замены mu = C*l, матрицу P = C^{-1} и набор beta.
По лекции (слайд 8) замена переменных в самом уравнении тогда есть xi = P^T x,
и уравнение приобретает вид sum beta_i u_{xi_i xi_i} + F = 0.

Критерий правильности: P^T A P = diag(beta).
"""

from dataclasses import dataclass, field
from typing import List

from sympy import Abs, Rational, S, simplify, sqrt, sympify

from .exact import identity, inverse, is_zero, mat_mul, symmetrise, transpose


@dataclass
class LagrangeResult:
    beta: List
    C: List[List]
    P: List[List]
    steps: List[str] = field(default_factory=list)

    @property
    def n(self):
        return len(self.beta)


class IndeterminateSignError(ValueError):
    """Знак ведущего коэффициента не определён — нормировка невозможна."""


def _sign_of(value):
    value = simplify(sympify(value))
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    if value.is_zero:
        return 0
    raise IndeterminateSignError(
        f"не удалось определить знак выражения {value}; "
        "для символьных коэффициентов используйте метод характеристик"
    )


def _pair_substitution(n, i, j):
    """Матрица S замены l_i = (v_i+v_j)/2, l_j = (v_i-v_j)/2 (остальные без изменений).

    Нужна, когда в форме нет ни одного квадрата: без неё выделять нечего.
    Ровно этот приём применён в Примере 1 лекции.
    """
    s = identity(n)
    half = Rational(1, 2)
    s[i][i], s[i][j] = half, half
    s[j][i], s[j][j] = half, -half
    return s


def reduce_quadratic_form(a, normalize=True):
    """Приводит форму с матрицей `a` к каноническому виду методом Лагранжа.

    normalize=True  -> beta_i из {+1, -1, 0}, множители уходят внутрь mu_i
                       (как в лекции: mu_3 = sqrt(2)*l_3).
    normalize=False -> beta_i остаются рациональными, C без радикалов.
    """
    n = len(a)
    b = symmetrise(a)
    w = identity(n)  # текущие переменные w = W * lambda
    mu_rows = [None] * n
    beta = [None] * n
    remaining = list(range(n))
    steps: List[str] = []

    while remaining:
        k = next((i for i in remaining if not is_zero(b[i][i])), None)

        if k is None:
            pair = None
            for pos_i, i in enumerate(remaining):
                for j in remaining[pos_i + 1:]:
                    if not is_zero(b[i][j]):
                        pair = (i, j)
                        break
                if pair:
                    break

            if pair is None:
                # Оставшаяся часть формы тождественно нулевая.
                for i in remaining:
                    beta[i] = S.Zero
                    mu_rows[i] = list(w[i])
                    steps.append(f"mu_{i + 1}: квадратов больше нет, beta_{i + 1} = 0")
                remaining = []
                continue

            i, j = pair
            s = _pair_substitution(n, i, j)
            b = mat_mul(mat_mul(transpose(s), b), s)
            w = mat_mul(inverse(s), w)
            steps.append(
                f"в форме нет квадратов: замена "
                f"l_{i + 1} = (v_{i + 1}+v_{j + 1})/2, l_{j + 1} = (v_{i + 1}-v_{j + 1})/2"
            )
            continue

        head = b[k][k]

        # mu_k = scale * (w_k + sum_{j != k} (b_kj / b_kk) * w_j)
        row_w = [S.Zero] * n
        row_w[k] = S.One
        for j in remaining:
            if j != k:
                row_w[j] = simplify(b[k][j] / head)

        if normalize:
            scale = sqrt(Abs(head))
            beta[k] = S(_sign_of(head))
        else:
            scale = S.One
            beta[k] = head

        row_w = [simplify(scale * v) for v in row_w]
        mu_rows[k] = [simplify(v) for v in mat_mul([row_w], w)[0]]
        steps.append(
            f"выделяем квадрат по переменной {k + 1}: "
            f"коэффициент {head}, beta_{k + 1} = {beta[k]}"
        )

        # Дополнение Шура: b_ij <- b_ij - b_ki*b_kj/b_kk на оставшихся индексах.
        updated = [list(row) for row in b]
        for i in remaining:
            for j in remaining:
                if i == k or j == k:
                    updated[i][j] = S.Zero
                else:
                    updated[i][j] = simplify(b[i][j] - b[k][i] * b[k][j] / head)
        b = updated
        remaining.remove(k)

    c = mu_rows
    return LagrangeResult(beta=beta, C=c, P=inverse(c), steps=steps)
