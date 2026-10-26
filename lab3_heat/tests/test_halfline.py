"""Задание Б: полупрямая, сравнение с точными решениями."""
import math
import unittest

from heatlab.halfline import HeatHalfLine
from heatlab.kernels import green_half
from heatlab.userfunc import make_func


def solver(a, phi, mu, nu, kind=2, f="0"):
    return HeatHalfLine(a, make_func(phi, "x"), make_func(mu, "t"), make_func(nu, "t"), kind,
                        make_func(f, "x t"))


def unit_flux(x, t, a):
    """u_x(0,t) = 1, phi = 0:  u = -2[a sqrt(t/pi) exp(-z^2) - x/2 erfc z], z = x/(2a sqrt t)."""
    if t <= 0:
        return 0.0
    z = x / (2 * a * math.sqrt(t))
    return -2 * (a * math.sqrt(t / math.pi) * math.exp(-z * z) - x / 2 * math.erfc(z))


class TestGreenHalf(unittest.TestCase):
    def test_boundary_conditions_of_green(self):
        a, xi, t, h = 0.8, 1.3, 0.7, 1e-5
        self.assertAlmostEqual(green_half(0.0, xi, t, a, 1), 0.0, places=14)
        g2 = lambda x: green_half(x, xi, t, a, 2)
        self.assertAlmostEqual((g2(h) - g2(-h)) / (2 * h), 0.0, places=8)


class TestNeumannVariant(unittest.TestCase):
    """Вариант группы: u_x(0,t) = mu(t)*nu(t)."""

    def test_constant_flux_product(self):
        a, q = 0.7, 1.5                     # mu = -1, nu = 1.5  ->  u_x(0,t) = -1.5
        s = solver(a, "0", "-1", "1.5")
        for x in (0.0, 0.5, 2.0):
            self.assertAlmostEqual(s.u(x, 1.2), -q * unit_flux(x, 1.2, a), delta=1e-7)

    def test_time_dependent_flux_duhamel(self):
        # u_x(0,t) = t*2: по Дюамелю u = int_0^t g'(tau) U(x, t - tau) dtau, U -- единичный поток
        a, t = 0.7, 1.0
        s = solver(a, "0", "t", "2")
        for x in (0.0, 0.3, 1.5):
            n = 20000
            ref = sum(2.0 * unit_flux(x, t - (i + 0.5) * t / n, a) for i in range(n)) * t / n
            self.assertAlmostEqual(s.u(x, t), ref, delta=1e-6)

    def test_flux_is_satisfied(self):
        # численная производная u_x в точке x = 0 совпадает с mu*nu (вдали от t = 0)
        s = solver(1.0, "exp(-(x-2)**2)", "sin(t)", "2")
        t, h = 1.5, 0.02
        ux = (-3 * s.u(0.0, t) + 4 * s.u(h, t) - s.u(2 * h, t)) / (2 * h)
        self.assertAlmostEqual(ux, math.sin(t) * 2, delta=2e-2)

    def test_insulated_cosine(self):
        a = 0.7
        s = solver(a, "cos(2*x)", "0", "1")
        self.assertAlmostEqual(s.u(0.7, 0.6), math.exp(-4 * a * a * 0.6) * math.cos(1.4), delta=1e-7)

    def test_insulated_conserves_heat(self):
        from heatlab.numint import Simpson
        s = solver(1.0, "exp(-(x-1)**2)", "0", "5")
        mass0 = math.sqrt(math.pi) / 2 * (1 + math.erf(1.0))
        self.assertAlmostEqual(Simpson(301)(lambda x: s.u(x, 0.8), 0.0, 15.0), mass0, places=4)

    def test_source(self):
        s = solver(1.0, "0", "0", "1", f="1")
        self.assertAlmostEqual(s.u(0.7, 0.6), 0.6, delta=1e-9)
        self.assertAlmostEqual(s.u(0.0, 0.6), 0.6, delta=1e-9)


class TestDirichlet(unittest.TestCase):
    def test_constant_temperature(self):
        a = 0.7
        s = solver(a, "0", "1", "1", kind=1)
        for x in (0.0, 0.5, 2.0):
            self.assertAlmostEqual(s.u(x, 1.2), math.erfc(x / (2 * a * math.sqrt(1.2))), delta=1e-7)

    def test_sine(self):
        a = 0.7
        s = solver(a, "sin(2*x)", "0", "1", kind=1)
        self.assertAlmostEqual(s.u(0.7, 0.6), math.exp(-4 * a * a * 0.6) * math.sin(1.4), delta=1e-7)

    def test_boundary_value(self):
        s = solver(1.0, "0", "cos(t)", "3", kind=1)
        self.assertAlmostEqual(s.u(0.0, 1.0), 3 * math.cos(1.0), delta=1e-6)


class TestInput(unittest.TestCase):
    def test_errors(self):
        with self.assertRaises(ValueError):
            solver(1.0, "0", "1", "1", kind=3)
        with self.assertRaises(ValueError):
            solver(1.0, "0", "1", "1").u(-1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
