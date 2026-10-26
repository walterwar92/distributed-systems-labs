"""Задание Б. Задача Коши на прямой через функцию Грина.

    u_tt = a^2 u_xx + f(x, t),  -inf < x < inf, t > 0;   u(x, 0) = phi(x),  u_t(x, 0) = psi(x).

Функция Грина (лекция 4, формула (4)):
    G(x, xi, t) = 1/(2a) при |x - xi| <= a t,   0 при |x - xi| > a t.

Решение (формула (3)):
    u = int G_t(x, xi, t) phi(xi) dxi + int G(x, xi, t) psi(xi) dxi
        + int_0^t int G(x, xi, t - tau) f(xi, tau) dxi dtau.
"""
from .numint import linspace, simpson_weights


def green(x, xi, t, a):
    """G(x, xi, t) -- отклик в точке x на единичный импульс в точке xi в момент t = 0."""
    if t > 0 and abs(x - xi) <= a * t + 1e-12:
        return 1.0 / (2 * a)
    return 0.0


def green_integral(x, t, a, func, s, w):
    """int G(x, xi, t) func(xi) dxi по формуле Симпсона.

    G = 0 вне [x - a t, x + a t], поэтому узлы берутся только на этом отрезке:
    xi_j = x + a t s_j, где s_j -- узлы на [-1, 1], w_j -- веса.
    """
    radius = a * t
    total = 0.0
    for j in range(len(s)):
        xi = x + radius * s[j]
        total += w[j] * green(x, xi, t, a) * func(xi)
    return radius * total


def solve_cauchy(x, t, a, phi, psi=None, f=None, n=201, nt=101):
    """Список значений u(x_i, t) для списка точек x в момент времени t >= 0.

    phi(x), psi(x), f(x, t) -- функции чисел; psi=None и f=None означают нулевые функции.
    n, nt -- числа узлов формулы Симпсона по xi и по tau (нечётные).
    """
    if a <= 0 or t < 0:
        raise ValueError("нужно a > 0 и t >= 0")
    s, w = linspace(-1.0, 1.0, n), simpson_weights(n)
    if f is not None:
        tau = linspace(0.0, t, nt)
        w_tau = simpson_weights(nt)

    result = []
    for xp in x:
        # 1) слагаемое с phi. G_t = [delta(xi - (x - a t)) + delta(xi - (x + a t))] / 2,
        #    поэтому интеграл берётся точно (фильтрующее свойство дельта-функции)
        u = 0.5 * (phi(xp - a * t) + phi(xp + a * t))
        if t > 0:
            # 2) слагаемое с psi
            if psi is not None:
                u += green_integral(xp, t, a, psi, s, w)
            # 3) слагаемое с f: внешний интеграл по tau, внутренний -- с функцией G(x, xi, t - tau)
            if f is not None:
                outer = 0.0
                for i in range(nt):
                    moment = tau[i]
                    outer += w_tau[i] * green_integral(xp, t - moment, a, lambda xi: f(xi, moment), s, w)
                u += outer * t / 2
        result.append(u)
    return result
