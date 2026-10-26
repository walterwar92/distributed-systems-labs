# -*- coding: utf-8 -*-
"""Задание A: по коэффициентам ДУЧП определить тип и замену переменных.

Эталоны — «Примеры к лекции 1.pdf», п.2 (примеры 1 и 2) и слайд 14 лекции 1.
"""
import unittest

from sympy import Rational, simplify, sqrt, symbols

from pde_canonical.characteristics import classify_2d
from pde_canonical.classifier import classify

x, y, z = symbols("x y z", real=True)

# Пример 1: u_xy - 2u_xz + u_yz + u_x + (1/2)u_y = 0
EXAMPLE_1 = [
    [0, Rational(1, 2), -1],
    [Rational(1, 2), 0, Rational(1, 2)],
    [-1, Rational(1, 2), 0],
]

# Пример 2: u_xx + 2u_xy + 5u_yy - 32u = 0
EXAMPLE_2 = [[1, 1], [1, 5]]


class TestLectureExample1(unittest.TestCase):
    """Гиперболический; xi = x+y, eta = x-y, omega = -(sqrt2/2)x + sqrt2*y + (sqrt2/2)z."""

    def setUp(self):
        self.res = classify(EXAMPLE_1, variables=[x, y, z])

    def test_type_is_hyperbolic(self):
        self.assertEqual(self.res.type, "hyperbolic")

    def test_beta_matches_lecture(self):
        self.assertEqual([simplify(b) for b in self.res.beta], [1, -1, 1])

    def test_substitution_matches_lecture(self):
        want = [
            x + y,
            x - y,
            -sqrt(2) / 2 * x + sqrt(2) * y + sqrt(2) / 2 * z,
        ]
        for got, expected in zip(self.res.new_variables, want):
            self.assertEqual(simplify(got - expected), 0, f"получено {got}")

    def test_reduction_is_verified(self):
        self.assertTrue(self.res.verified)


class TestLectureExample2(unittest.TestCase):
    """Эллиптический; xi = x, eta = -x/2 + y/2."""

    def setUp(self):
        self.res = classify(EXAMPLE_2, variables=[x, y])

    def test_type_is_elliptic(self):
        self.assertEqual(self.res.type, "elliptic")

    def test_substitution_matches_lecture(self):
        want = [x, -x / 2 + y / 2]
        for got, expected in zip(self.res.new_variables, want):
            self.assertEqual(simplify(got - expected), 0, f"получено {got}")

    def test_reduction_is_verified(self):
        self.assertTrue(self.res.verified)


class TestClassicalEquations(unittest.TestCase):
    def test_wave_equation_is_hyperbolic(self):
        # u_tt - u_xx = 0
        self.assertEqual(classify([[1, 0], [0, -1]]).type, "hyperbolic")

    def test_three_dimensional_wave_equation_is_hyperbolic(self):
        a = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        self.assertEqual(classify(a).type, "hyperbolic")

    def test_heat_equation_is_parabolic(self):
        # u_t = u_xx -> старшая часть только u_xx
        self.assertEqual(classify([[1, 0], [0, 0]]).type, "parabolic")

    def test_laplace_equation_is_elliptic(self):
        self.assertEqual(classify([[1, 0], [0, 1]]).type, "elliptic")

    def test_three_dimensional_laplace_is_elliptic(self):
        self.assertEqual(classify([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).type, "elliptic")

    def test_negative_definite_form_is_still_elliptic(self):
        self.assertEqual(classify([[-1, 0], [0, -1]]).type, "elliptic")


class TestDegenerateAndExoticSignatures(unittest.TestCase):
    def test_two_plus_two_signature_is_flagged_ultrahyperbolic(self):
        a = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        res = classify(a)
        self.assertEqual(res.type, "hyperbolic")
        self.assertEqual(res.subtype, "ultrahyperbolic")

    def test_normal_hyperbolic_is_not_flagged_ultrahyperbolic(self):
        res = classify([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.assertEqual(res.subtype, "normal")

    def test_zero_beta_with_mixed_signs_is_ultraparabolic(self):
        a = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]
        res = classify(a)
        self.assertEqual(res.type, "parabolic")
        self.assertEqual(res.subtype, "ultraparabolic")

    def test_empty_principal_part_is_rejected(self):
        with self.assertRaises(ValueError):
            classify([[0, 0], [0, 0]])


class TestInputForms(unittest.TestCase):
    def test_accepts_dictionary_of_coefficients(self):
        # {(i, j): a_ij}; недостающие считаются нулями, матрица симметризуется
        res = classify({(0, 0): 1, (1, 1): -1}, variables=[x, y])
        self.assertEqual(res.type, "hyperbolic")

    def test_asymmetric_input_is_symmetrised(self):
        # u_xy с коэффициентом 2 можно задать как a12 = 2, a21 = 0
        res = classify([[0, 2], [0, 0]])
        self.assertEqual(res.type, "hyperbolic")
        self.assertTrue(res.verified)

    def test_default_variable_names_are_generated(self):
        res = classify([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(len(res.new_variables), 3)


class TestCanonicalFormRendering(unittest.TestCase):
    def test_canonical_text_shows_signs_of_beta(self):
        text = classify([[1, 0], [0, -1]]).canonical_text
        self.assertIn("u_", text)
        self.assertIn("-", text)

    def test_substitution_is_rendered_as_equations(self):
        lines = classify(EXAMPLE_2, variables=[x, y]).substitution_text
        self.assertEqual(len(lines), 2)
        self.assertTrue(all("=" in line for line in lines))


class TestAgreementBetweenTheTwoMethods(unittest.TestCase):
    """Для n = 2 метод Лагранжа и метод характеристик обязаны давать один тип."""

    CASES = [
        (1, 0, -1),
        (1, 0, 1),
        (1, 0, 0),
        (1, 1, 5),
        (1, -1, 1),
        (0, Rational(1, 2), 0),
        (2, 3, 1),
        (-1, 0, 1),
        (3, -2, 4),
    ]

    def test_both_methods_agree_on_the_type(self):
        for a11, a12, a22 in self.CASES:
            with self.subTest(coeffs=(a11, a12, a22)):
                lagrange_type = classify([[a11, a12], [a12, a22]]).type
                characteristic_type = classify_2d(a11, a12, a22).type
                self.assertEqual(lagrange_type, characteristic_type)


if __name__ == "__main__":
    unittest.main()
