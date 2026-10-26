# -*- coding: utf-8 -*-
"""Тесты метода характеристик для случая двух переменных.

Эталоны: «Лекция 1» (слайды 11-13, 15-17) и «Примеры к лекции 1.pdf», п.3.

Замечание о знаках. Характеристика phi(x,y) = C определена с точностью до
ненулевого множителя: phi и -phi задают одно и то же семейство. Порядок xi/eta
для эллиптического типа также несущественен. Поэтому такие проверки ниже
сравнивают пары «с точностью до знака и порядка», а не буквально.
"""
import unittest

from sympy import Rational, simplify, symbols

from pde_canonical.characteristics import classify_2d

x, y = symbols("x y", real=True)


def same_up_to_sign(a, b):
    return simplify(a - b) == 0 or simplify(a + b) == 0


class PairAssertions(unittest.TestCase):
    def assertPair(self, got, want):
        g1, g2 = got
        w1, w2 = want
        direct = same_up_to_sign(g1, w1) and same_up_to_sign(g2, w2)
        swapped = same_up_to_sign(g1, w2) and same_up_to_sign(g2, w1)
        self.assertTrue(
            direct or swapped,
            f"получено {got}, ожидалось {want} (с точностью до знака и порядка)",
        )


class TestDiscriminantAndType(unittest.TestCase):
    def test_wave_equation_is_hyperbolic(self):
        res = classify_2d(1, 0, -1)
        self.assertEqual(simplify(res.D), 1)
        self.assertEqual(res.type, "hyperbolic")

    def test_laplace_equation_is_elliptic(self):
        res = classify_2d(1, 0, 1)
        self.assertEqual(simplify(res.D), -1)
        self.assertEqual(res.type, "elliptic")

    def test_heat_equation_is_parabolic(self):
        # u_t = u_xx -> старшая часть только u_xx, т.е. a11=1, a12=a22=0
        res = classify_2d(1, 0, 0)
        self.assertEqual(simplify(res.D), 0)
        self.assertEqual(res.type, "parabolic")

    def test_discriminant_uses_a12_squared_minus_a11_a22(self):
        self.assertEqual(simplify(classify_2d(1, 1, 5).D), -4)

    def test_symbolic_discriminant_without_region_is_rejected(self):
        # Знак -y неизвестен -> классифицировать нельзя, нужна область.
        with self.assertRaises(ValueError):
            classify_2d(y, 0, 1)


class TestLectureParabolicExample(unittest.TestCase):
    """u_xx - 2u_xy + u_yy + 9u_x + 9u_y - 9u = 0 -> xi = x + y, eta = x."""

    def setUp(self):
        self.res = classify_2d(1, -1, 1)

    def test_type_is_parabolic(self):
        self.assertEqual(self.res.type, "parabolic")

    def test_xi_is_the_single_characteristic(self):
        self.assertTrue(same_up_to_sign(self.res.xi, x + y), f"xi = {self.res.xi}")

    def test_eta_is_linearly_independent_of_xi(self):
        self.assertNotEqual(simplify(self.res.jacobian), 0)

    def test_only_u_eta_eta_survives(self):
        self.assertEqual(simplify(self.res.new_a11), 0)
        self.assertEqual(simplify(self.res.new_a12), 0)
        self.assertNotEqual(simplify(self.res.new_a22), 0)


class TestHyperbolicSubstitution(PairAssertions):
    """Для D > 0 обе новые переменные — характеристики, остаётся только u_xi_eta."""

    def test_wave_equation_characteristics(self):
        res = classify_2d(1, 0, -1)
        self.assertPair((res.xi, res.eta), (y - x, y + x))

    def test_mixed_derivative_is_the_only_survivor(self):
        res = classify_2d(1, 0, -1)
        self.assertEqual(simplify(res.new_a11), 0)
        self.assertEqual(simplify(res.new_a22), 0)
        self.assertNotEqual(simplify(res.new_a12), 0)

    def test_canonical_form_is_the_first_hyperbolic_one(self):
        res = classify_2d(1, 0, -1)
        self.assertIn("u_{\\xi\\eta}", res.canonical_latex)


class TestEllipticSubstitution(PairAssertions):
    def test_laplace_substitution_is_essentially_identity(self):
        res = classify_2d(1, 0, 1)
        self.assertPair((res.xi, res.eta), (y, x))

    def test_new_coefficients_are_equal_and_mixed_one_vanishes(self):
        res = classify_2d(1, 1, 5)
        self.assertEqual(simplify(res.new_a12), 0)
        self.assertEqual(simplify(res.new_a11 - res.new_a22), 0)


class TestDegenerateLeadingCoefficient(unittest.TestCase):
    """a11 = 0 ломает формулу dy/dx = (a12 ± sqrt(D))/a11 — нужен отдельный разбор."""

    def test_u_xy_is_already_canonical(self):
        res = classify_2d(0, Rational(1, 2), 0)
        self.assertEqual(res.type, "hyperbolic")
        self.assertEqual(simplify(res.new_a11), 0)
        self.assertEqual(simplify(res.new_a22), 0)
        self.assertNotEqual(simplify(res.jacobian), 0)

    def test_u_yy_only_is_parabolic_with_valid_substitution(self):
        res = classify_2d(0, 0, 1)
        self.assertEqual(res.type, "parabolic")
        self.assertNotEqual(simplify(res.jacobian), 0)
        self.assertEqual(simplify(res.new_a11), 0)
        self.assertEqual(simplify(res.new_a12), 0)

    def test_a22_zero_parabolic_is_handled(self):
        res = classify_2d(1, 0, 0)
        self.assertEqual(res.type, "parabolic")
        self.assertNotEqual(simplify(res.jacobian), 0)
        self.assertEqual(simplify(res.new_a11), 0)
        self.assertEqual(simplify(res.new_a12), 0)


class TestTricomiPointwise(unittest.TestCase):
    """Уравнение Трикоми y*u_xx + u_yy = 0 в отдельных точках."""

    def test_below_the_axis_it_is_hyperbolic(self):
        self.assertEqual(classify_2d(-1, 0, 1).type, "hyperbolic")

    def test_above_the_axis_it_is_elliptic(self):
        self.assertEqual(classify_2d(1, 0, 1).type, "elliptic")

    def test_on_the_axis_it_is_parabolic(self):
        self.assertEqual(classify_2d(0, 0, 1).type, "parabolic")


class TestTricomiWithVariableCoefficients(PairAssertions):
    """Трикоми по областям: region={y: -1} означает «в области y < 0»."""

    def test_hyperbolic_branch_characteristics(self):
        res = classify_2d(y, 0, 1, region={y: -1})
        self.assertEqual(res.type, "hyperbolic")
        want = (
            x + Rational(2, 3) * (-y) ** Rational(3, 2),
            x - Rational(2, 3) * (-y) ** Rational(3, 2),
        )
        self.assertPair((res.xi, res.eta), want)

    def test_hyperbolic_branch_really_kills_the_square_terms(self):
        res = classify_2d(y, 0, 1, region={y: -1})
        self.assertEqual(simplify(res.new_a11), 0)
        self.assertEqual(simplify(res.new_a22), 0)
        self.assertNotEqual(simplify(res.new_a12), 0)

    def test_elliptic_branch_substitution(self):
        res = classify_2d(y, 0, 1, region={y: 1})
        self.assertEqual(res.type, "elliptic")
        self.assertPair((res.xi, res.eta), (x, Rational(2, 3) * y ** Rational(3, 2)))

    def test_elliptic_branch_gives_equal_square_coefficients(self):
        res = classify_2d(y, 0, 1, region={y: 1})
        self.assertEqual(simplify(res.new_a12), 0)
        self.assertEqual(simplify(res.new_a11 - res.new_a22), 0)


class TestSignNormalisation(unittest.TestCase):
    """Запись выбирается детерминированно, без ведущего минуса.

    phi и -phi задают одну характеристику, но в отчёте и на защите запись
    должна совпадать с лекцией: eta = (2/3)y^{3/2}, а не -(2/3)y^{3/2}.
    """

    def test_elliptic_eta_has_no_leading_minus(self):
        res = classify_2d(y, 0, 1, region={y: 1})
        self.assertEqual(
            simplify(res.eta - Rational(2, 3) * y ** Rational(3, 2)),
            0,
            f"eta = {res.eta}",
        )

    def test_normalisation_does_not_break_the_reduction(self):
        res = classify_2d(y, 0, 1, region={y: 1})
        self.assertEqual(simplify(res.new_a12), 0)
        self.assertEqual(simplify(res.new_a11 - res.new_a22), 0)
        self.assertNotEqual(simplify(res.jacobian), 0)


if __name__ == "__main__":
    unittest.main()
