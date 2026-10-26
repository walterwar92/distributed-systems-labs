# -*- coding: utf-8 -*-
"""Метод характеристик для ДУЧП 2-го порядка от двух переменных.

Исходное уравнение (лекция 1, формула 5'):

    a11*u_xx + 2*a12*u_xy + a22*u_yy + F(x, y, u, u_x, u_y) = 0

Характеристическое уравнение (7):  a11*dy^2 - 2*a12*dx*dy + a22*dx^2 = 0,
его дискриминант (8):              D = a12^2 - a11*a22.

    D > 0 -> гиперболический,  D = 0 -> параболический,  D < 0 -> эллиптический.

Коэффициенты могут зависеть от x и y. Тогда знак D сам по себе неизвестен, и
нужно указать область: region={y: -1} читается как «в области y < 0».
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sympy import (
    Dummy,
    I,
    Integral,
    S,
    Symbol,
    diff,
    expand,
    integrate,
    log,
    simplify,
    sqrt,
    symbols,
    sympify,
)

X, Y = symbols("x y", real=True)

HYPERBOLIC = "hyperbolic"
PARABOLIC = "parabolic"
ELLIPTIC = "elliptic"

TYPE_RU = {
    HYPERBOLIC: "гиперболический",
    PARABOLIC: "параболический",
    ELLIPTIC: "эллиптический",
}

CANONICAL_LATEX = {
    HYPERBOLIC: r"u_{\xi\eta} = \Phi(\xi, \eta, u, u_\xi, u_\eta)",
    PARABOLIC: r"u_{\eta\eta} = \Phi(\xi, \eta, u, u_\xi, u_\eta)",
    ELLIPTIC: r"u_{\xi\xi} + u_{\eta\eta} = \Phi(\xi, \eta, u, u_\xi, u_\eta)",
}

CANONICAL_TEXT = {
    HYPERBOLIC: "u_ξη = Φ(ξ, η, u, u_ξ, u_η)",
    PARABOLIC: "u_ηη = Φ(ξ, η, u, u_ξ, u_η)",
    ELLIPTIC: "u_ξξ + u_ηη = Φ(ξ, η, u, u_ξ, u_η)",
}


@dataclass
class Result2D:
    a11: object
    a12: object
    a22: object
    D: object
    type: str
    xi: object
    eta: object
    jacobian: object
    new_a11: object
    new_a12: object
    new_a22: object
    characteristics: List[str] = field(default_factory=list)
    canonical_latex: str = ""
    canonical_text: str = ""
    steps: List[str] = field(default_factory=list)

    @property
    def type_ru(self):
        return TYPE_RU[self.type]


def _is_zero(value):
    return simplify(sympify(value)) == 0


def _sign_of(value):
    """Знак выражения; символы уже заменены положительными фиктивными."""
    value = simplify(sympify(value))
    if value.is_zero:
        return 0
    if value.is_positive:
        return 1
    if value.is_negative:
        return -1
    raise ValueError(
        f"не удалось определить знак дискриминанта D = {value}. "
        "Укажите область, например region={y: -1} для y < 0"
    )


def _build_frame(region, x, y):
    """Переход к рабочим переменным, в которых знаки коэффициентов известны.

    Если задано y < 0, подставляем y = -p с положительным p. Замена меняет знак
    дифференциала (dy = -dp), поэтому в характеристическом уравнении
    a11*dy^2 - 2*a12*dx*dy + a22*dx^2 = 0 у коэффициента a12 меняется знак.
    Замена только одной из переменных переворачивает произведение dx*dy один раз.
    """
    region = region or {}
    to_work, back = {}, {}
    xw, yw = x, y
    flipped = 0

    for sym, sign in region.items():
        dummy = Dummy(f"{getattr(sym, 'name', 'p')}_pos", positive=True)
        to_work[sym] = dummy if sign >= 0 else -dummy
        back[dummy] = sym if sign >= 0 else -sym
        if sym == x:
            xw = dummy
        elif sym == y:
            yw = dummy
        if sign < 0:
            flipped += 1

    return to_work, back, xw, yw, flipped % 2 == 1


def _separate(r, u, v):
    """Разложение r(u, v) = F(u) * G(v), если оно существует."""
    try:
        if simplify(diff(log(r), u, v)) != 0:
            return None, None
        u0, v0 = S(2), S(3)
        anchor = simplify(r.subs({u: u0, v: v0}))
        if anchor == 0:
            return None, None
        f_u = simplify(r.subs(v, v0))
        g_v = simplify(r.subs(u, u0) / anchor)
        if simplify(f_u * g_v - r) != 0:
            return None, None
        return f_u, g_v
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None


def _first_integral(kind, r, x, y):
    """Первый интеграл phi(x, y) = C характеристического ОДУ."""
    r = simplify(sympify(r))

    if kind == "dydx":
        if r == 0:
            return y
        if not r.has(y):
            return simplify(y - integrate(r, x))
        if not r.has(x):
            return simplify(integrate(1 / r, y) - x)
        f_x, g_y = _separate(r, x, y)
        if f_x is None:
            return None
        return simplify(integrate(1 / g_y, y) - integrate(f_x, x))

    if r == 0:
        return x
    if not r.has(x):
        return simplify(x - integrate(r, y))
    if not r.has(y):
        return simplify(integrate(1 / r, x) - y)
    f_y, g_x = _separate(r, y, x)
    if f_y is None:
        return None
    return simplify(integrate(1 / g_x, x) - integrate(f_y, y))


def _normalise_sign(phi, x, y):
    """phi и -phi задают одну характеристику; выбираем «положительную» запись."""
    if phi is None:
        return None
    for var in (y, x):
        slope = simplify(diff(phi, var))
        if slope.is_number:
            if slope.is_negative:
                return simplify(-phi)
            if slope.is_positive:
                return phi
    # Ни одна производная не решила спор (например, phi = -(2/3)y^{3/2}):
    # тогда просто убираем ведущий минус, чтобы запись совпадала с лекцией.
    if phi.could_extract_minus_sign():
        return simplify(-phi)
    return phi


def _roots(a11, a12, a22, disc):
    """Корни характеристического уравнения с учётом вырождения a11 = 0."""
    root = sqrt(disc)
    if not _is_zero(a11):
        return [
            ("dydx", simplify((a12 + root) / a11)),
            ("dydx", simplify((a12 - root) / a11)),
        ]
    if not _is_zero(a22):
        # Тот же квадратный трёхчлен относительно dx/dy: a22*s^2 - 2*a12*s + a11 = 0.
        return [
            ("dxdy", simplify((a12 + root) / a22)),
            ("dxdy", simplify((a12 - root) / a22)),
        ]
    # a11 = a22 = 0, остаётся -2*a12*dx*dy = 0.
    return [("dydx", S.Zero), ("dxdy", S.Zero)]


def _pick_second_variable(xi, x, y):
    """Свободный выбор eta при D = 0: лишь бы якобиан был ненулевым."""
    for candidate in (x, y, x + y, x - y):
        jac = simplify(diff(xi, x) * diff(candidate, y) - diff(xi, y) * diff(candidate, x))
        if jac != 0:
            return candidate, jac
    raise ValueError(f"не удалось подобрать eta, линейно независимую с xi = {xi}")


def classify_2d(a11, a12, a22, region: Optional[Dict] = None, x=X, y=Y):
    """Определяет тип уравнения и замену переменных, приводящую к каноническому виду."""
    a11, a12, a22 = sympify(a11), sympify(a12), sympify(a22)
    disc = simplify(a12**2 - a11 * a22)

    if _is_zero(a11) and _is_zero(a12) and _is_zero(a22):
        raise ValueError("старшая часть уравнения пуста: это не ДУЧП 2-го порядка")

    to_work, back, xw, yw, flip_mixed = _build_frame(region, x, y)
    a11w = simplify(a11.subs(to_work)) if to_work else a11
    a12w = simplify(a12.subs(to_work)) if to_work else a12
    a22w = simplify(a22.subs(to_work)) if to_work else a22
    if flip_mixed:
        a12w = simplify(-a12w)
    discw = simplify(a12w**2 - a11w * a22w)

    sign = _sign_of(discw)
    steps = [f"D = a12^2 - a11*a22 = {disc}"]

    if sign > 0:
        eq_type = HYPERBOLIC
        (k1, r1), (k2, r2) = _roots(a11w, a12w, a22w, discw)
        phi = _normalise_sign(_first_integral(k1, r1, xw, yw), xw, yw)
        psi = _normalise_sign(_first_integral(k2, r2, xw, yw), xw, yw)
        if phi is None or psi is None:
            raise ValueError("не удалось проинтегрировать характеристическое уравнение")
        xi, eta = phi, psi
        steps.append("D > 0: две характеристики, xi и eta — обе характеристики")

    elif sign == 0:
        eq_type = PARABOLIC
        k1, r1 = _roots(a11w, a12w, a22w, discw)[0]
        phi = _normalise_sign(_first_integral(k1, r1, xw, yw), xw, yw)
        if phi is None:
            raise ValueError("не удалось проинтегрировать характеристическое уравнение")
        xi, eta = phi, None
        steps.append("D = 0: одна характеристика, eta выбирается свободно")

    else:
        eq_type = ELLIPTIC
        k1, r1 = _roots(a11w, a12w, a22w, discw)[0]
        full = _first_integral(k1, r1, xw, yw)
        if full is None:
            raise ValueError("не удалось проинтегрировать характеристическое уравнение")
        real_part, imag_part = simplify(expand(full)).as_real_imag()
        xi = _normalise_sign(simplify(real_part), xw, yw)
        eta = _normalise_sign(simplify(imag_part), xw, yw)
        steps.append("D < 0: корни комплексно сопряжены, xi = Re(phi), eta = Im(phi)")

    # Возвращаемся к исходным переменным.
    if back:
        xi = simplify(xi.subs(back))
        if eta is not None:
            eta = simplify(eta.subs(back))

    if eta is None:
        eta, _ = _pick_second_variable(xi, x, y)

    xi_x, xi_y = simplify(diff(xi, x)), simplify(diff(xi, y))
    eta_x, eta_y = simplify(diff(eta, x)), simplify(diff(eta, y))
    jacobian = simplify(xi_x * eta_y - xi_y * eta_x)

    if jacobian == 0:
        raise ValueError(f"вырожденная замена: якобиан равен нулю (xi={xi}, eta={eta})")

    new_a11 = simplify(a11 * xi_x**2 + 2 * a12 * xi_x * xi_y + a22 * xi_y**2)
    new_a12 = simplify(
        a11 * xi_x * eta_x + a12 * (xi_x * eta_y + xi_y * eta_x) + a22 * xi_y * eta_y
    )
    new_a22 = simplify(a11 * eta_x**2 + 2 * a12 * eta_x * eta_y + a22 * eta_y**2)

    characteristics = [f"{xi} = C1"]
    if eq_type == HYPERBOLIC:
        characteristics.append(f"{eta} = C2")

    return Result2D(
        a11=a11,
        a12=a12,
        a22=a22,
        D=disc,
        type=eq_type,
        xi=xi,
        eta=eta,
        jacobian=jacobian,
        new_a11=new_a11,
        new_a12=new_a12,
        new_a22=new_a22,
        characteristics=characteristics,
        canonical_latex=CANONICAL_LATEX[eq_type],
        canonical_text=CANONICAL_TEXT[eq_type],
        steps=steps,
    )
