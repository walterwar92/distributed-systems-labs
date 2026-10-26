"""Задание А. Волновое уравнение на полупрямой, метод характеристик.

    u_tt = a^2 u_xx,  x > 0, t > 0;   u(x, 0) = phi(x),  u_t(x, 0) = psi(x);
    граничное условие при x = 0:
        kind = 1:  u(0, t) = g(t)                  (I рода)
        kind = 2:  u_x(0, t) = g(t)                (II рода)
        kind = 3:  u_x(0, t) - h*u(0, t) = g(t)    (III рода)

Общее решение: u = F(x - a t) + G(x + a t)  (F постоянна на характеристиках
x - a t = C, G -- на характеристиках x + a t = C).

Из начальных условий при z >= 0:
    G(z) = phi(z)/2 + Psi(z)/(2a),   F(z) = phi(z)/2 - Psi(z)/(2a),   Psi(z) = int_0^z psi.
Характеристика x - a t = -y < 0 выходит с границы, и F(-y) находится из граничного условия:
    I рода:       F(-y) = g(y/a) - G(y);
    II, III рода: F(-y) = G(y) - W(y),  где  W' + h W = 2 h G(y) + g(y/a),  W(0) = 0
                  (для II рода h = 0, то есть W(y) = int_0^y g(q/a) dq).
"""
import math

from .numint import cumtrapz, interp, linspace


class HalfLine:
    """Решение задачи при 0 <= x + a t <= zmax: таблицы строятся один раз, u(x, t) считается быстро."""

    def __init__(self, a, phi, psi, kind, g, h=0.0, zmax=1.0, n=20000):
        """phi, psi, g -- функции одного числа; n -- число шагов сетки по аргументу бегущих волн."""
        if kind not in (1, 2, 3):
            raise ValueError("kind должен быть 1, 2 или 3")
        if a <= 0:
            raise ValueError("нужно a > 0")
        self.a, self.phi, self.kind, self.g = a, phi, kind, g
        self.zmax = max(zmax, 1e-12)
        self.dz = self.zmax / n
        z = linspace(0.0, self.zmax, n + 1)

        # Psi(z_k) = int_0^{z_k} psi -- формула трапеций
        self.Psi = cumtrapz([psi(v) for v in z], self.dz)

        if kind != 1:
            if kind == 2:
                h = 0.0
            # W' + h W = r(y): формула трапеций с точным множителем exp(-h dz) на шаге
            r = [(2 * h * self.G(v) if h else 0.0) + g(v / a) for v in z]
            q = math.exp(-h * self.dz)
            self.W = [0.0] * (n + 1)
            for k in range(n):
                self.W[k + 1] = q * (self.W[k] + 0.5 * self.dz * r[k]) + 0.5 * self.dz * r[k + 1]

    def G(self, s):
        """Волна, бегущая влево; s >= 0."""
        return self.phi(s) / 2 + interp(s, self.dz, self.Psi) / (2 * self.a)

    def F(self, s):
        """Волна, бегущая вправо; s любого знака."""
        if s >= 0:                       # характеристика выходит с оси t = 0
            return self.phi(s) / 2 - interp(s, self.dz, self.Psi) / (2 * self.a)
        y = -s                           # характеристика выходит с границы x = 0
        if self.kind == 1:
            return self.g(y / self.a) - self.G(y)
        return self.G(y) - interp(y, self.dz, self.W)

    def u(self, x, t):
        """Значение решения в точке x >= 0 в момент t >= 0."""
        if x < 0 or t < 0:
            raise ValueError("решение определено только при x >= 0, t >= 0")
        if x + self.a * t > self.zmax * (1 + 1e-9):
            raise ValueError("точка вне области, для которой построены таблицы (x + a t > zmax)")
        return self.F(x - self.a * t) + self.G(x + self.a * t)


def solve_halfline(x, t, a, phi, psi, kind, g, h=0.0, n=20000):
    """Список значений u(x_i, t) для списка точек x в момент времени t."""
    if any(v < 0 for v in x) or t < 0:
        raise ValueError("решение определено только при x >= 0, t >= 0")
    solver = HalfLine(a, phi, psi, kind, g, h, zmax=max(x) + a * t, n=n)
    return [solver.u(v, t) for v in x]
