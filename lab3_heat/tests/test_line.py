"""Задание А: сравнение с точными решениями задачи Коши на прямой."""
import math
import unittest

from heatlab.kernels import green_line
from heatlab.line import HeatLine
from heatlab.numint import Simpson, adaptive_simpson
from heatlab.userfunc import make_func

METHODS = ("poisson", "green")


def gauss_exact(x, t, a):
    """phi = exp(-x^2)  ->  u = exp(-x^2/(1+4a^2 t)) / sqrt(1+4a^2 t)."""
    d = 1.0 + 4.0 * a * a * t
    return math.exp(-x * x / d) / math.sqrt(d)


class TestKernel(unittest.TestCase):
    def test_kernel_integral_is_one(self):
        for t in (0.01, 0.5, 3.0):
            total = adaptive_simpson(lambda xi: green_line(0.3, xi, t, 0.8), -30, 30, 1e-12)
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_kernel_solves_heat_equation(self):
        a, x, xi, t, h = 0.9, 0.7, -0.2, 0.6, 1e-4
        k = lambda x_, t_: green_line(x_, xi, t_, a)
        k_t = (k(x, t + h) - k(x, t - h)) / (2 * h)
        k_xx = (k(x + h, t) - 2 * k(x, t) + k(x - h, t)) / (h * h)
        self.assertAlmostEqual(k_t, a * a * k_xx, places=5)

    def test_simpson_exact_for_cubic(self):
        self.assertAlmostEqual(Simpson(5)(lambda s: s ** 3 + s * s, 0.0, 2.0), 4 + 8 / 3, places=12)


class TestLine(unittest.TestCase):
    def check(self, phi, f, exact, a, points, tol):
        for method in METHODS:
            solver = HeatLine(a, make_func(phi, "x"), make_func(f, "x t"), method)
            for x, t in points:
                with self.subTest(method=method, x=x, t=t):
                    delta = tol[method] if isinstance(tol, dict) else tol
                    self.assertAlmostEqual(solver.u(x, t), exact(x, t), delta=delta)

    def test_gauss(self):
        self.check("exp(-x**2)", "0", lambda x, t: gauss_exact(x, t, 0.7), 0.7,
                   [(0.0, 0.3), (0.4, 1.3), (-2.0, 2.5), (1.0, 0.005)], 1e-6)

    def test_sine_decays(self):
        a, k = 1.2, 2.0
        self.check("sin(2*x)", "0", lambda x, t: math.exp(-a * a * k * k * t) * math.sin(k * x), a,
                   [(0.3, 0.1), (1.1, 0.5)], 1e-6)

    def test_step(self):
        # разрыв в точке, кратной h: у метода Грина (середины ячеек) ошибка O(h^2),
        # у адаптивного Симпсона в формуле Пуассона -- на уровне допуска
        a = 0.7
        exact = lambda x, t: 0.5 * (1 + math.erf(x / (2 * a * math.sqrt(t))))
        self.check("theta(x)", "0", exact, a, [(0.3, 0.5), (-0.5, 1.0), (0.0, 2.0)],
                   {"poisson": 1e-5, "green": 1e-3})

    def test_constant_source(self):
        self.check("0", "1", lambda x, t: t, 1.0, [(0.0, 0.5), (2.0, 1.5)], 1e-9)

    def test_source_and_initial_superpose(self):
        a = 0.7
        exact = lambda x, t: math.exp(-4 * a * a * t) * math.sin(2 * x) + t * t / 2
        self.check("sin(2*x)", "t", exact, a, [(0.3, 0.8), (-1.0, 1.4)], 1e-6)

    def test_initial_condition_at_zero(self):
        solver = HeatLine(1.0, make_func("x**2", "x"))
        self.assertEqual(solver.u(1.5, 0.0), 2.25)

    def test_quadratic(self):
        # phi = x^2  ->  u = x^2 + 2 a^2 t
        a = 0.5
        self.check("x**2", "0", lambda x, t: x * x + 2 * a * a * t, a, [(1.0, 1.0), (-2.0, 0.3)], 1e-6)

    def test_mass_conserved(self):
        solver = HeatLine(1.0, make_func("theta(1-abs(x))", "x"))
        rule = Simpson(401)
        mass = rule(lambda x: solver.u(x, 0.7), -12.0, 12.0)
        self.assertAlmostEqual(mass, 2.0, places=4)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            HeatLine(-1.0, make_func("x", "x"))
        with self.assertRaises(ValueError):
            HeatLine(1.0, make_func("x", "x"), method="fourier")


if __name__ == "__main__":
    unittest.main()
