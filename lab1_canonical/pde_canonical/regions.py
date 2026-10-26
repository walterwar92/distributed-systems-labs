# -*- coding: utf-8 -*-
"""Задание C. ДУЧП с переменными коэффициентами: где какой тип сохраняется.

Для уравнения a11(x,y)u_xx + 2a12(x,y)u_xy + a22(x,y)u_yy + F = 0 дискриминант
D(x, y) = a12^2 - a11*a22 сам является функцией точки. Плоскость разбивается на
области постоянства знака D, и в каждой уравнение имеет свой тип:

    D > 0 — гиперболический, D = 0 — параболический, D < 0 — эллиптический.

Классический пример (лекция 1, слайды 15-17) — уравнение Трикоми y*u_xx + u_yy = 0
с D = -y: гиперболическое при y < 0, параболическое на y = 0, эллиптическое при y > 0.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sympy import (
    Rational,
    S,
    lambdify,
    nsimplify,
    simplify,
    solve,
    symbols,
    sympify,
)

from .characteristics import ELLIPTIC, HYPERBOLIC, PARABOLIC, TYPE_RU, classify_2d

X, Y = symbols("x y", real=True)

SIGN_TO_TYPE = {1: HYPERBOLIC, 0: PARABOLIC, -1: ELLIPTIC}


@dataclass
class Region:
    description: str
    type: str
    sample_point: Dict
    result: object = None
    note: str = ""

    @property
    def type_ru(self):
        return TYPE_RU[self.type]


@dataclass
class RegionAnalysis:
    a11: object
    a12: object
    a22: object
    D: object
    regions: List[Region] = field(default_factory=list)
    boundary: List = field(default_factory=list)

    @property
    def is_mixed(self):
        return len({r.type for r in self.regions}) > 1


def _sign_at(expr, point):
    value = simplify(expr.subs(point))
    if value.is_zero:
        return 0
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    raise ValueError(f"не удалось определить знак D в точке {point}")


def _try_reduce(a11, a12, a22, region, note_prefix=""):
    """Приведение в области; при неудаче возвращает (None, пояснение)."""
    try:
        return classify_2d(a11, a12, a22, region=region), ""
    except (ValueError, TypeError, NotImplementedError) as exc:
        return None, f"{note_prefix}приведение не выполнено автоматически: {exc}"


def _real_roots(expr, var):
    try:
        roots = solve(expr, var)
    except (NotImplementedError, TypeError):
        return []
    out = []
    for root in roots:
        root = simplify(root)
        if root.is_real and root not in out:
            out.append(root)
    return sorted(out, key=lambda r: float(r))


def _grid_values(lo, hi, n):
    lo, hi = nsimplify(lo), nsimplify(hi)
    if n == 1:
        return [simplify((lo + hi) / 2)]
    step = Rational(1, n - 1)
    return [simplify(lo + (hi - lo) * step * i) for i in range(n)]


def region_map(a11, a12, a22, xlim=(-2, 2), ylim=(-2, 2), n=121, x=X, y=Y):
    """Карта типов: возвращает (xs, ys, grid), grid[j][i] = знак D в точке.

    Знаки: +1 гиперболический, 0 параболический, -1 эллиптический.
    """
    disc = simplify(sympify(a12) ** 2 - sympify(a11) * sympify(a22))
    xs = _grid_values(xlim[0], xlim[1], n)
    ys = _grid_values(ylim[0], ylim[1], n)

    try:
        fast = lambdify((x, y), disc, "math")
    except Exception:  # noqa: BLE001 - любая неудача компиляции -> точный путь
        fast = None

    grid = []
    for yv in ys:
        row = []
        for xv in xs:
            value = None
            if fast is not None:
                try:
                    value = fast(float(xv), float(yv))
                except Exception:  # noqa: BLE001
                    value = None
            if value is None or abs(value) < 1e-12:
                # Около нуля решаем точно, чтобы не спутать границу с шумом.
                row.append(_sign_at(disc, {x: xv, y: yv}))
            else:
                row.append(1 if value > 0 else -1)
        grid.append(row)
    return xs, ys, grid


def _analyse_univariate(a11, a12, a22, disc, var, other):
    """D зависит от одной переменной: границы — это её корни."""
    roots = _real_roots(disc, var)
    regions: List[Region] = []

    if not roots:
        point = {var: S.Zero, other: S.Zero}
        sign = _sign_at(disc, point)
        result, note = _try_reduce(a11, a12, a22, {var: 1})
        regions.append(
            Region(
                description=f"вся плоскость (D = {disc} знака не меняет)",
                type=SIGN_TO_TYPE[sign],
                sample_point=point,
                result=result,
                note=note,
            )
        )
        return regions, roots

    bounds = [None] + list(roots) + [None]
    for index in range(len(bounds) - 1):
        lo, hi = bounds[index], bounds[index + 1]

        if lo is None and hi is not None:
            sample, text = hi - 1, f"{var} < {hi}"
        elif lo is not None and hi is None:
            sample, text = lo + 1, f"{var} > {lo}"
        else:
            sample, text = simplify((lo + hi) / 2), f"{lo} < {var} < {hi}"

        point = {var: sample, other: S.Zero}
        sign = _sign_at(disc, point)

        # Знак самой переменной постоянен на интервале только если он не
        # содержит нуля; иначе подсказку region передать нельзя.
        if (lo is None or lo >= 0) and sample > 0:
            hint = {var: 1}
        elif (hi is None or hi <= 0) and sample < 0:
            hint = {var: -1}
        else:
            hint = None

        if hint is None:
            result, note = None, "интервал содержит 0: укажите область вручную"
        else:
            result, note = _try_reduce(a11, a12, a22, hint)

        regions.append(
            Region(
                description=text,
                type=SIGN_TO_TYPE[sign],
                sample_point=point,
                result=result,
                note=note,
            )
        )

        # Сама граница var = hi.
        if hi is not None:
            edge_point = {var: hi, other: S.Zero}
            b11 = simplify(a11.subs(var, hi))
            b12 = simplify(a12.subs(var, hi))
            b22 = simplify(a22.subs(var, hi))
            if all(simplify(c) == 0 for c in (b11, b12, b22)):
                edge_result, edge_note = None, "на границе старшая часть вырождается"
            else:
                edge_result, edge_note = _try_reduce(b11, b12, b22, None, "на границе ")
            regions.append(
                Region(
                    description=f"{var} = {hi}",
                    type=SIGN_TO_TYPE[_sign_at(disc, edge_point)],
                    sample_point=edge_point,
                    result=edge_result,
                    note=edge_note,
                )
            )

    return regions, roots


def _analyse_by_sampling(a11, a12, a22, disc, x, y, window):
    """Общий случай: D зависит от обеих переменных — сканируем сетку."""
    xs, ys, grid = region_map(a11, a12, a22, window[:2], window[2:], n=41, x=x, y=y)

    best: Dict[int, Dict] = {}
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            sign = grid[j][i]
            candidate = {x: xv, y: yv}
            # Предпочитаем точки, где обе координаты ненулевые: только для них
            # можно передать подсказку о знаках переменных.
            score = (xv != 0) + (yv != 0)
            if sign not in best or score > best[sign][1]:
                best[sign] = (candidate, score)

    regions = []
    for sign in (1, 0, -1):
        if sign not in best:
            continue
        point, _ = best[sign]
        hint = {}
        for sym in (x, y):
            if not disc.has(sym):
                continue
            value = point[sym]
            if value == 0:
                hint = None
                break
            hint[sym] = 1 if value > 0 else -1

        if hint is None or sign == 0:
            result, note = None, "граница/особая точка: приведение задайте вручную"
        else:
            result, note = _try_reduce(a11, a12, a22, hint)

        readable = {1: f"D = {disc} > 0", 0: f"D = {disc} = 0", -1: f"D = {disc} < 0"}
        regions.append(
            Region(
                description=readable[sign],
                type=SIGN_TO_TYPE[sign],
                sample_point=point,
                result=result,
                note=note,
            )
        )
    return regions


def analyse_regions(a11, a12, a22, x=X, y=Y, window=(-2, 2, -2, 2)):
    """Разбивает плоскость на области, в каждой из которых тип уравнения постоянен."""
    a11, a12, a22 = sympify(a11), sympify(a12), sympify(a22)
    disc = simplify(a12**2 - a11 * a22)
    free = [sym for sym in (x, y) if disc.has(sym)]

    if not free:
        sign = _sign_at(disc, {})
        result, note = _try_reduce(a11, a12, a22, None)
        return RegionAnalysis(
            a11=a11,
            a12=a12,
            a22=a22,
            D=disc,
            regions=[
                Region(
                    description="вся плоскость (коэффициенты постоянны)",
                    type=SIGN_TO_TYPE[sign],
                    sample_point={x: S.Zero, y: S.Zero},
                    result=result,
                    note=note,
                )
            ],
            boundary=[],
        )

    if len(free) == 1:
        var = free[0]
        other = y if var == x else x
        regions, roots = _analyse_univariate(a11, a12, a22, disc, var, other)
        return RegionAnalysis(
            a11=a11, a12=a12, a22=a22, D=disc, regions=regions, boundary=roots
        )

    regions = _analyse_by_sampling(a11, a12, a22, disc, x, y, window)
    return RegionAnalysis(
        a11=a11,
        a12=a12,
        a22=a22,
        D=disc,
        regions=regions,
        boundary=_real_roots(disc, y) if disc.has(y) else [],
    )
