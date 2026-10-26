"""Задание Б: уравнение теплопроводности на полупрямой через функцию Грина.

    u_t = a^2 u_xx + f(x, t),   x > 0,  t > 0;      u(x, 0) = phi(x),  x >= 0.

Вариант группы -- граничное условие II рода, правая часть -- произведение двух функций:

    u_x(0, t) = mu(t) * nu(t)  =: g(t).

Решение через функцию Грина G2 = K(x, xi, t) + K(x, -xi, t):

    u(x,t) = int_0^inf G2(x, xi, t) phi(xi) dxi
             - a^2 int_0^t G2(x, 0, t - tau) g(tau) dtau
             + int_0^t dtau int_0^inf G2(x, xi, t - tau) f(xi, tau) dxi.

Граничное слагаемое: G2(x, 0, s) = exp(-x^2/(4a^2 s)) / (a sqrt(pi s)) имеет особенность
1/sqrt(s) при tau -> t; замена t - tau = sigma^2 её убирает:

    -a^2 int_0^t G2(x,0,t-tau) g(tau) dtau = -(2a/sqrt(pi)) int_0^sqrt(t) g(t - sigma^2) exp(-x^2/(4a^2 sigma^2)) dsigma.

Дополнительно (kind = 1) -- условие I рода u(0, t) = mu(t) * nu(t), функция Грина
G1 = K(x, xi, t) - K(x, -xi, t); граничное слагаемое после замены w = x / (2a sqrt(t - tau)):

    a^2 int_0^t dG1/dxi(x, 0, t-tau) g(tau) dtau = (2/sqrt(pi)) int_{x/(2a sqrt t)}^inf g(t - x^2/(4a^2 w^2)) exp(-w^2) dw.
"""
import math

from .kernels import SQRT_PI, green_half
from .line import Z, is_zero
from .numint import Simpson, adaptive_simpson, linspace


class HeatHalfLine:
    """Решатель задания Б. kind = 2 -- вариант группы (II рода), kind = 1 -- I рода."""

    def __init__(self, a, phi, mu, nu, kind=2, f=None, tol=1e-6, ntau=15, nxif=31):
        if a <= 0:
            raise ValueError("коэффициент a должен быть положительным")
        if kind not in (1, 2):
            raise ValueError("род граничного условия: 1 или 2")
        self.a, self.phi, self.kind = a, phi, kind
        self.g = lambda t: mu(t) * nu(t)            # правая часть граничного условия
        self.g_zero = is_zero(mu) or is_zero(nu)
        self.f = None if is_zero(f) else f
        self.tol = tol                              # допуск адаптивного Симпсона (phi и граница)
        self.rule_tau = Simpson(ntau)               # интеграл по tau (источник)
        self.rule_f = Simpson(nxif)                 # внутренний интеграл источника

    def green(self, x, xi, t):
        return green_half(x, xi, t, self.a, self.kind)

    # --- int_0^inf G(x, xi, s) h(xi) dxi ---------------------------------
    def convolve(self, h, x, s, rule=None):
        if s <= 0.0:
            if self.kind == 1 and x <= 0.0:
                return 0.0
            return h(x)
        width = Z * 2.0 * self.a * math.sqrt(s)
        lo, hi = max(0.0, x - width), x + width
        # вне [lo, hi] обе части G (прямая и отражённая) меньше 1e-16
        integrand = lambda xi: self.green(x, xi, s) * h(xi)
        if rule is None:
            return adaptive_simpson(integrand, lo, hi, self.tol)
        return rule(integrand, lo, hi)

    def initial_part(self, x, t):
        return self.convolve(self.phi, x, t)

    def boundary_part(self, x, t):
        if self.g_zero or t <= 0.0:
            return 0.0
        a, g = self.a, self.g
        if self.kind == 2:
            c = x * x / (4.0 * a * a)

            def integrand(sigma):
                if sigma == 0.0:
                    return g(t) if x == 0.0 else 0.0
                q = c / (sigma * sigma)
                return g(t - sigma * sigma) * math.exp(-q) if q < 700.0 else 0.0

            return -2.0 * a / SQRT_PI * adaptive_simpson(integrand, 0.0, math.sqrt(t), self.tol)
        # I рода
        w0 = x / (2.0 * a * math.sqrt(t))
        if w0 >= Z:
            return 0.0
        c = x * x / (4.0 * a * a)
        integrand = lambda w: g(t - c / (w * w)) * math.exp(-w * w) if w > 0.0 else g(t) * 1.0
        return 2.0 / SQRT_PI * adaptive_simpson(integrand, w0, Z, self.tol)

    def source_part(self, x, t):
        if self.f is None or t <= 0.0:
            return 0.0
        f = self.f
        inner = lambda tau: self.convolve(lambda xi: f(xi, tau), x, t - tau, self.rule_f)
        return self.rule_tau(inner, 0.0, t)

    def u(self, x, t):
        if x < 0.0:
            raise ValueError("решение задаётся при x >= 0")
        if t <= 0.0:
            return self.phi(x)
        return self.initial_part(x, t) + self.boundary_part(x, t) + self.source_part(x, t)

    def row(self, xs, t):
        return [self.u(x, t) for x in xs]

    def flux(self, t):
        """Значение граничного условия g(t) = mu(t) * nu(t)."""
        return self.g(t)


def compute_frames(a, phi, mu, nu, kind, f, T, xmax, xi0, frames=81, points=None):
    """Решение, функция Грина и граничная функция на всех кадрах:
    times, x, U[k][i], G[k][i], B[k] = mu*nu в момент times[k]."""
    solver = HeatHalfLine(a, phi, mu, nu, kind, f)
    if points is None:
        points = 201 if solver.f is None else 101
    x = linspace(0.0, xmax, points)
    times = linspace(0.0, T, frames)
    U = [solver.row(x, t) for t in times]
    G = [[solver.green(v, xi0, t) for v in x] for t in times]
    B = [solver.flux(t) for t in times]
    return times, x, U, G, B
