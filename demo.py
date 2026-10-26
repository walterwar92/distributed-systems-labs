# -*- coding: utf-8 -*-
"""Демонстрация: воспроизводим все примеры из лекции 1 и разбора к ней.

Запуск:  python demo.py
"""
import sys

from sympy import Rational, symbols

from pde_canonical.characteristics import classify_2d
from pde_canonical.classifier import classify
from pde_canonical.regions import analyse_regions

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

x, y, z = symbols("x y z", real=True)

RULE = "=" * 78
THIN = "-" * 78


def head(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def show_nvar(title, equation, coeffs, variables):
    print(f"\n{THIN}\n{title}\n  {equation}\n{THIN}")
    res = classify(coeffs, variables=variables)
    print(f"  тип:              {res.type_ru} ({res.subtype_ru})")
    print(f"  beta:             {[str(b) for b in res.beta]}")
    print(f"  матрица C (mu = C*lambda):")
    for row in res.C:
        print(f"      {[str(v) for v in row]}")
    print(f"  матрица P = C^(-1):")
    for row in res.P:
        print(f"      {[str(v) for v in row]}")
    print("  замена переменных (xi = P^T x):")
    for line in res.substitution_text:
        print(f"      {line}")
    print(f"  канонический вид: {res.canonical_text}")
    print(f"  проверка P^T A P = diag(beta):  {'ОК' if res.verified else 'ОШИБКА'}")


def show_2d(title, equation, a11, a12, a22, region=None):
    print(f"\n{THIN}\n{title}\n  {equation}\n{THIN}")
    res = classify_2d(a11, a12, a22, region=region)
    print(f"  D = a12^2 - a11*a22 = {res.D}")
    print(f"  тип:              {res.type_ru}")
    print("  характеристики:")
    for line in res.characteristics:
        print(f"      {line}")
    print(f"  замена:           xi = {res.xi},   eta = {res.eta}")
    print(f"  якобиан:          {res.jacobian}")
    print(
        f"  новые коэффициенты: A11 = {res.new_a11}, "
        f"A12 = {res.new_a12}, A22 = {res.new_a22}"
    )
    print(f"  канонический вид: {res.canonical_text}")


def main():
    head("ЗАДАНИЕ A. n переменных — метод Лагранжа (Примеры к лекции 1, п.2)")

    show_nvar(
        "Пример 1",
        "u_xy - 2u_xz + u_yz + u_x + (1/2)u_y = 0",
        [
            [0, Rational(1, 2), -1],
            [Rational(1, 2), 0, Rational(1, 2)],
            [-1, Rational(1, 2), 0],
        ],
        [x, y, z],
    )

    show_nvar(
        "Пример 2",
        "u_xx + 2u_xy + 5u_yy - 32u = 0",
        [[1, 1], [1, 5]],
        [x, y],
    )

    head("ЗАДАНИЕ A. Две переменные — метод характеристик (п.3) и классика")

    show_2d(
        "Пример 3 (параболический)",
        "u_xx - 2u_xy + u_yy + 9u_x + 9u_y - 9u = 0",
        1, -1, 1,
    )
    show_2d("Волновое уравнение", "u_tt - u_xx = 0", 1, 0, -1)
    show_2d("Уравнение теплопроводности", "u_t = u_xx", 1, 0, 0)
    show_2d("Уравнение Лапласа", "u_xx + u_yy = 0", 1, 0, 1)

    head("ЗАДАНИЕ C. Переменные коэффициенты — уравнение Трикоми")

    analysis = analyse_regions(y, 0, 1)
    print(f"\n  Уравнение:  y*u_xx + u_yy = 0")
    print(f"  D(x, y) = {analysis.D}")
    print(f"  уравнение меняет тип: {'да' if analysis.is_mixed else 'нет'}\n")

    for region in analysis.regions:
        print(f"  Область {region.description!r}: {region.type_ru}")
        if region.result is not None:
            res = region.result
            print(f"      замена:  xi = {res.xi},   eta = {res.eta}")
            print(
                f"      новые коэффициенты: A11 = {res.new_a11}, "
                f"A12 = {res.new_a12}, A22 = {res.new_a22}"
            )
            print(f"      канонический вид: {res.canonical_text}")
        if region.note:
            print(f"      примечание: {region.note}")
        print()

    head("ЗАДАНИЕ C. Ещё примеры смены типа")

    for title, a11, a12, a22 in [
        ("u_xx + x*u_yy = 0", 1, 0, x),
        ("u_xx + x*y*u_yy = 0", 1, 0, x * y),
        ("(1 - x^2)u_xx + u_yy = 0", 1 - x**2, 0, 1),
    ]:
        analysis = analyse_regions(a11, a12, a22)
        print(f"\n  {title}:  D = {analysis.D}")
        for region in analysis.regions:
            print(f"      {region.description}  ->  {region.type_ru}")


if __name__ == "__main__":
    main()
