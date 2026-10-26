"""Задание А: задача Коши для уравнения теплопроводности на прямой.

    u_t = a^2 u_xx + f(x, t),   -inf < x < inf,  t > 0;      u(x, 0) = phi(x).

Решение (формула Пуассона + интеграл Дюамеля для источника):

    u(x,t) = int K(x, xi, t) phi(xi) dxi + int_0^t dtau int K(x, xi, t - tau) f(xi, tau) dxi,
    K(x, xi, t) = exp(-(x - xi)^2 / (4 a^2 t)) / (2a sqrt(pi t)).

Два способа счёта (параметр method):

* "poisson" -- формула Пуассона после замены xi = x + 2a sqrt(t) z:
      int K phi dxi = 1/sqrt(pi) * int phi(x + 2a sqrt(t) z) exp(-z^2) dz,
  интеграл по z в [-Z, Z] (Z = 6, exp(-36) ~ 2e-16) считается адаптивной формулой Симпсона
  (шаг сам мельчает возле разрывов phi);
* "green" -- прямая свёртка с функцией Грина на неподвижной сетке xi_j = (j + 1/2) h
  (середины ячеек, формула средних прямоугольников):
      u(x, t) ~ sum_j K(x, xi_j, t) phi(xi_j) h  /  sum_j K(x, xi_j, t) h
  (знаменатель -- дискретный аналог int K dxi = 1, он сохраняет точность при малых t;
  суммирование в окне |x - xi| <= ZG * 2a sqrt(t), значения phi(xi_j) считаются один раз).
"""
import math

from .kernels import SQRT_PI, green_line
from .numint import Simpson, adaptive_simpson, linspace

Z = 6.0                                     # обрезка интеграла Пуассона: |z| <= Z
ZG = 5.0                                    # окно метода Грина: |x - xi| <= ZG * 2a sqrt(t), exp(-25) ~ 1e-11


def is_zero(func):
    """True, если пользователь ввёл тождественный нуль: такое слагаемое можно не считать."""
    return func is None or getattr(func, "text", "").strip() in ("0", "0.0")


class HeatLine:
    """Решатель задания А. Все данные задаются один раз, u(x, t) -- для любых x и t."""

    def __init__(self, a, phi, f=None, method="poisson", tol=1e-6, ntau=21, nzf=41, h=0.04):
        if a <= 0:
            raise ValueError("коэффициент a должен быть положительным")
        if method not in ("poisson", "green"):
            raise ValueError("method: poisson или green")
        self.a, self.phi, self.method, self.h, self.tol = a, phi, method, h, tol
        self.f = None if is_zero(f) else f
        self.table = {}                     # phi(xi_j) на сетке метода Грина: считается один раз
        self.rule_f = Simpson(nzf)          # для внутреннего интеграла источника
        self.rule_tau = Simpson(ntau)       # для интеграла по времени tau

    # --- интеграл от K(x, xi, s) g(xi) по xi ------------------------------
    def convolve(self, g, x, s, rule=None, table=None):
        """int K(x, xi, s) g(xi) dxi; при s = 0 это просто g(x).

        table -- словарь j -> g((j + 1/2) h) для метода Грина (кэш значений на сетке)."""
        if s <= 0.0:
            return g(x)
        if self.method == "poisson":
            r = 2.0 * self.a * math.sqrt(s)
            integrand = lambda z: g(x + r * z) * math.exp(-z * z)
            if rule is None:                # начальное условие: адаптивный Симпсон (точен и у разрывов)
                return adaptive_simpson(integrand, -Z, Z, self.tol) / SQRT_PI
            return rule(integrand, -Z, Z) / SQRT_PI
        # свёртка на неподвижной сетке xi_j = (j + 1/2) h -- середины ячеек [jh, (j+1)h]
        # (формула средних прямоугольников) в окне |x - xi| <= ZG * 2a sqrt(s)
        width = ZG * 2.0 * self.a * math.sqrt(s)
        j0, j1 = math.floor((x - width) / self.h), math.ceil((x + width) / self.h)
        num = den = 0.0
        for j in range(j0, j1 + 1):
            xi = (j + 0.5) * self.h
            k = green_line(x, xi, s, self.a)
            if table is None:
                value = g(xi)
            else:
                value = table.get(j)
                if value is None:
                    value = table[j] = g(xi)
            num += k * value
            den += k
        return num / den if den > 0.0 else g(x)

    def initial_part(self, x, t):
        """Вклад начального условия: формула Пуассона."""
        return self.convolve(self.phi, x, t, table=self.table)

    def source_part(self, x, t):
        """Вклад источника f: интеграл Дюамеля по tau в [0, t]."""
        if self.f is None or t <= 0.0:
            return 0.0
        f = self.f
        inner = lambda tau: self.convolve(lambda xi: f(xi, tau), x, t - tau, self.rule_f)
        return self.rule_tau(inner, 0.0, t)

    def u(self, x, t):
        if t <= 0.0:
            return self.phi(x)
        return self.initial_part(x, t) + self.source_part(x, t)

    def row(self, xs, t):
        return [self.u(x, t) for x in xs]


def compute_frames(a, phi, f, T, xmin, xmax, xi0, frames=81, method="poisson", points=None):
    """Решение и функция Грина на всех кадрах анимации: times, x, U[k][i], G[k][i]."""
    solver = HeatLine(a, phi, f, method)
    if points is None:
        points = 241 if solver.f is None else 121       # с источником -- двойной интеграл, сетка грубее
    x = linspace(xmin, xmax, points)
    times = linspace(0.0, T, frames)
    U = [solver.row(x, t) for t in times]
    G = [[green_line(v, xi0, t, a) for v in x] for t in times]
    return times, x, U, G
