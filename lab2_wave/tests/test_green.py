"""Задание Б: свойства функции Грина и сравнение с точными решениями."""
import math
import unittest

from wavelab.green import green, solve_cauchy
from wavelab.numint import cumtrapz, interp, linspace, simpson_weights
from wavelab.userfunc import make_func

A = 1.5
X = linspace(-6.0, 6.0, 25)


def max_diff(u, v):
    return max(abs(p - q) for p, q in zip(u, v))


def bell(x):
    return math.exp(-x * x)


class GreenFunction(unittest.TestCase):
    def test_values(self):
        self.assertEqual(green(0.0, 0.0, 1.0, A), 1 / (2 * A))
        self.assertEqual(green(A, 0.0, 1.0, A), 1 / (2 * A))        # на фронте |x - xi| = a t
        self.assertEqual(green(A + 0.01, 0.0, 1.0, A), 0.0)         # вне конуса
        self.assertEqual(green(0.0, 0.0, 0.0, A), 0.0)              # G(x, xi, 0) = 0
        self.assertEqual(green(0.0, 0.0, -1.0, A), 0.0)             # G = 0 при t < 0

    def test_symmetry(self):
        for x in X:
            for xi in X:
                self.assertEqual(green(x, xi, 2.0, A), green(xi, x, 2.0, A))

    def test_area(self):
        # int G dx = (1/2a) * 2 a t = t
        x = linspace(-20.0, 20.0, 40001)
        for t in (0.5, 2.0, 7.0):
            area = sum(green(v, 0.3, t, A) for v in x) * (x[1] - x[0])
            self.assertAlmostEqual(area, t, places=2)


class ExactSolutions(unittest.TestCase):
    def test_dalembert(self):
        for t in (0.0, 0.4, 2.0):
            u = solve_cauchy(X, t, A, bell, math.sin)
            ref = [0.5 * (bell(x - A * t) + bell(x + A * t))
                   + (math.cos(x - A * t) - math.cos(x + A * t)) / (2 * A) for x in X]
            self.assertLess(max_diff(u, ref), 1e-7)

    def test_source_sin(self):
        # u_tt = a^2 u_xx + sin x, нулевые начальные данные: u = sin x (1 - cos a t) / a^2
        for t in (0.3, 2.0):
            u = solve_cauchy(X, t, A, lambda x: 0.0, None, lambda x, t: math.sin(x))
            ref = [math.sin(x) * (1 - math.cos(A * t)) / A ** 2 for x in X]
            self.assertLess(max_diff(u, ref), 1e-7)

    def test_source_xt(self):
        # f = x t: u = x t^3 / 6
        u = solve_cauchy(X, 2.0, A, lambda x: 0.0, None, lambda x, t: x * t, n=21, nt=21)
        self.assertLess(max_diff(u, [x * 8 / 6 for x in X]), 1e-9)

    def test_lecture_example(self):
        # лекция 3, слайды 5-7: phi = l - |x| на [-l, l], psi = 0, t = 10
        l, a, t = 1.0, 1.0, 10.0
        phi = lambda x: max(l - abs(x), 0.0)
        x = linspace(-12.0, 12.0, 481)
        u = solve_cauchy(x, t, a, phi)
        ref = [0.5 * (phi(v - a * t) + phi(v + a * t)) for v in x]
        self.assertLess(max_diff(u, ref), 1e-12)
        self.assertAlmostEqual(u[x.index(10.0)], 0.5)

    def test_phi_term_is_time_derivative(self):
        # int G_t phi dxi = d/dt int G phi dxi: проверяем численной производной по t
        t, dt = 1.2, 1e-4
        v = lambda tt: solve_cauchy(X, tt, A, lambda x: 0.0, bell)     # int G phi dxi
        numeric = [(p - q) / (2 * dt) for p, q in zip(v(t + dt), v(t - dt))]
        self.assertLess(max_diff(numeric, solve_cauchy(X, t, A, bell)), 1e-6)


class Helpers(unittest.TestCase):
    def test_linspace(self):
        self.assertEqual(linspace(0.0, 1.0, 5), [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_simpson(self):
        s, w = linspace(-1.0, 1.0, 11), simpson_weights(11)
        self.assertAlmostEqual(sum(wj * sj ** 3 for wj, sj in zip(w, s)), 0.0)
        self.assertAlmostEqual(sum(wj * sj ** 2 for wj, sj in zip(w, s)), 2 / 3)
        with self.assertRaises(ValueError):
            simpson_weights(10)

    def test_cumtrapz_and_interp(self):
        z = linspace(0.0, 2.0, 2001)
        table = cumtrapz([math.cos(v) for v in z], z[1])
        self.assertLess(max_diff(table, [math.sin(v) for v in z]), 1e-6)
        self.assertAlmostEqual(interp(1.23456, z[1], table), math.sin(1.23456), places=6)
        self.assertEqual(interp(5.0, z[1], table), table[-1])          # правее таблицы

    def test_make_func(self):
        f = make_func("exp(-x^2) * theta(x)")
        self.assertEqual([f(-1.0), f(0.0)], [0.0, 1.0])
        self.assertAlmostEqual(f(1.0), math.exp(-1))
        self.assertEqual(make_func("0")(3.0), 0)
        self.assertEqual(make_func("x * t", "x t")(2.0, 3.0), 6.0)
        self.assertEqual(make_func("maximum(1-abs(x),0)")(0.25), 0.75)
        for bad in ("__import__('os')", "open('file')", "exp(-(x-5)**2", "0.3t", "sn(x)", "x*t"):
            with self.assertRaises(ValueError, msg=bad):
                make_func(bad)


if __name__ == "__main__":
    unittest.main()
