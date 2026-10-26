# -*- coding: utf-8 -*-
"""Точная матричная арифметика.

Намеренно не используем numpy.linalg: обращение матрицы делается методом
Гаусса-Жордана над точными числами sympy, поэтому радикалы вида sqrt(2)
сохраняются, а не превращаются в 1.4142...
"""

from sympy import S, simplify, sympify


class SingularMatrixError(ValueError):
    """Матрица вырождена — обратной не существует."""


def is_zero(value):
    """Надёжная проверка «выражение равно нулю» для точных чисел sympy."""
    return simplify(sympify(value)) == 0


def identity(n):
    return [[S.One if i == j else S.Zero for j in range(n)] for i in range(n)]


def transpose(a):
    if not a:
        return []
    return [[sympify(a[i][j]) for i in range(len(a))] for j in range(len(a[0]))]


def mat_mul(a, b):
    if not a or not b:
        return []
    inner = len(b)
    if len(a[0]) != inner:
        raise ValueError(f"несогласованные размеры: {len(a[0])} и {inner}")
    cols = len(b[0])
    out = []
    for i in range(len(a)):
        row = []
        for j in range(cols):
            acc = S.Zero
            for k in range(inner):
                acc += sympify(a[i][k]) * sympify(b[k][j])
            row.append(simplify(acc))
        out.append(row)
    return out


def mat_vec(a, v):
    return [row[0] for row in mat_mul(a, [[x] for x in v])]


def scale_row(row, factor):
    return [simplify(sympify(v) * factor) for v in row]


def inverse(a):
    """Обращение методом Гаусса-Жордана с выбором ненулевого ведущего элемента."""
    n = len(a)
    for row in a:
        if len(row) != n:
            raise ValueError("обращать можно только квадратную матрицу")

    # Расширенная матрица [A | E].
    work = [
        [sympify(a[i][j]) for j in range(n)] + [S.One if i == j else S.Zero for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        pivot = next((r for r in range(col, n) if not is_zero(work[r][col])), None)
        if pivot is None:
            raise SingularMatrixError(f"матрица вырождена: нулевой столбец {col + 1}")
        work[col], work[pivot] = work[pivot], work[col]

        head = work[col][col]
        work[col] = [simplify(v / head) for v in work[col]]

        for r in range(n):
            if r == col or is_zero(work[r][col]):
                continue
            factor = work[r][col]
            work[r] = [
                simplify(work[r][k] - factor * work[col][k]) for k in range(2 * n)
            ]

    return [[work[i][j + n] for j in range(n)] for i in range(n)]


def diag(values):
    n = len(values)
    return [[sympify(values[i]) if i == j else S.Zero for j in range(n)] for i in range(n)]


def symmetrise(a):
    """Приводит матрицу квадратичной формы к симметричному виду."""
    n = len(a)
    out = [[sympify(a[i][j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            avg = simplify((out[i][j] + out[j][i]) / 2)
            out[i][j] = avg
            out[j][i] = avg
    return out
