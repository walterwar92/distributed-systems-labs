"""Задание А: сравнение с точными решениями."""
import math
import unittest

from wavelab.halfline import HalfLine, solve_halfline
from wavelab.numint import linspace

A, H = 2.0, 0.7
X = linspace(0.0, 12.0, 241)


def zero(_):
    return 0.0


def max_diff(u, v):
    return max(abs(p - q) for p, q in zip(u, v))


# точное решение u = sin(x - a t) + cos(2 (x + a t)); из него берутся все данные задачи
def exact(x, t):
    return math.sin(x - A * t) + math.cos(2 * (x + A * t))


def phi(x):
    return math.sin(x) + math.cos(2 * x)


def psi(x):
    return -A * math.cos(x) - 2 * A * math.sin(2 * x)


def mu(t):                      # u(0, t)
    return math.sin(-A * t) + math.cos(2 * A * t)


def nu(t):                      # u_x(0, t)
    return math.cos(A * t) - 2 * math.sin(2 * A * t)


def omega(t):                   # u_x(0, t) - h u(0, t)
    return nu(t) - H * mu(t)


class ExactSolution(unittest.TestCase):
    def check(self, kind, g, h=0.0):
        for t in (0.0, 0.5, 1.7, 4.0):
            u = solve_halfline(X, t, A, phi, psi, kind, g, h)
            self.assertLess(max_diff(u, [exact(x, t) for x in X]), 1e-5, f"kind={kind}, t={t}")

    def test_dirichlet(self):
        self.check(1, mu)

    def test_neumann(self):
        self.check(2, nu)

    def test_robin(self):
        self.check(3, omega, H)

    def test_one_table_many_times(self):
        # класс HalfLine: таблицы строятся один раз, решение берётся в разные моменты
        solver = HalfLine(A, phi, psi, 3, omega, H, zmax=12.0 + A * 4.0)
        for t in (0.0, 1.0, 4.0):
            self.assertLess(max_diff([solver.u(x, t) for x in X], [exact(x, t) for x in X]), 1e-5)


class LectureFormulas(unittest.TestCase):
    """Нулевые начальные данные: формулы (4в), (4г), (4д) лекции 3."""
    t = 3.0

    def check(self, kind, g, reference, tol, h=0.0):
        u = solve_halfline(X, self.t, A, zero, zero, kind, g, h)
        for x, value in zip(X, u):
            if x < A * self.t:                       # возмущение дошло
                self.assertLess(abs(value - reference(self.t - x / A)), tol)
            else:                                    # ещё не дошло
                self.assertEqual(value, 0.0)

    def test_4v(self):
        self.check(1, math.sin, math.sin, 1e-9)

    def test_4g(self):
        # -a * int_0^y cos(p) dp = -a sin(y)
        self.check(2, math.cos, lambda y: -A * math.sin(y), 1e-6)

    def test_4d(self):
        # omega = 1: -a e^{-h a y} int_0^y e^{h a s} ds = -(1 - e^{-h a y}) / h
        self.check(3, lambda t: 1.0, lambda y: -(1 - math.exp(-H * A * y)) / H, 1e-6, H)


class Reflection(unittest.TestCase):
    """Горб, ушедший влево, возвращается: от закреплённого конца перевёрнутым, от свободного -- нет."""

    def test_fixed_and_free_end(self):
        hump = lambda x: math.exp(-(x - 5.0) ** 2)
        t = 5.0                               # 2 * 5 / a: левая полуволна вернулась в x = 5
        fixed = solve_halfline(X, t, A, hump, zero, 1, zero)
        free = solve_halfline(X, t, A, hump, zero, 2, zero)
        i = X.index(5.0)
        self.assertAlmostEqual(fixed[i], -0.5, places=6)
        self.assertAlmostEqual(free[i], 0.5, places=6)
        self.assertAlmostEqual(fixed[0], 0.0, places=9)


class Variant(unittest.TestCase):
    """Вариант: u(0, t) = mu(t) + g(t). Решение -- сумма решений трёх более простых задач."""

    def test_superposition(self):
        m, g, t = math.sin, lambda s: 0.3 * s, 4.0
        full = solve_halfline(X, t, A, phi, psi, 1, lambda s: m(s) + g(s))
        part1 = solve_halfline(X, t, A, phi, psi, 1, zero)      # начальные данные, u(0,t) = 0
        part2 = solve_halfline(X, t, A, zero, zero, 1, m)       # только mu
        part3 = solve_halfline(X, t, A, zero, zero, 1, g)       # только g
        self.assertLess(max_diff(full, [p + q + r for p, q, r in zip(part1, part2, part3)]), 1e-12)
        self.assertAlmostEqual(full[0], m(t) + g(t), places=12)  # граничное условие выполнено


class BadInput(unittest.TestCase):
    def test_errors(self):
        with self.assertRaises(ValueError):
            solve_halfline(X, 1.0, A, phi, psi, 4, mu)
        with self.assertRaises(ValueError):
            solve_halfline([-1.0, 1.0], 1.0, A, phi, psi, 1, mu)
        with self.assertRaises(ValueError):
            solve_halfline(X, 1.0, 0.0, phi, psi, 1, mu)
        with self.assertRaises(ValueError):
            HalfLine(A, phi, psi, 1, mu, zmax=5.0).u(4.0, 1.0)     # x + a t = 6 > zmax


if __name__ == "__main__":
    unittest.main()
