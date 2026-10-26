# -*- coding: utf-8 -*-
"""Тесты метода Лагранжа (выделение полных квадратов).

Эталонные значения взяты из «Примеры к лекции 1.pdf», п.2.
"""
import unittest

from sympy import Rational, sqrt, simplify

from pde_canonical.exact import mat_mul, transpose
from pde_canonical.lagrange import reduce_quadratic_form

# Пример 1: u_xy - 2u_xz + u_yz + u_x + (1/2)u_y = 0
# Q = l1*l2 - 2*l1*l3 + l2*l3  =>  a_ij (симметричная матрица)
EXAMPLE_1_A = [
    [0, Rational(1, 2), -1],
    [Rational(1, 2), 0, Rational(1, 2)],
    [-1, Rational(1, 2), 0],
]

# Пример 2: u_xx + 2u_xy + 5u_yy - 32u = 0
EXAMPLE_2_A = [[1, 1], [1, 5]]


def assert_mat(test, got, want):
    test.assertEqual(len(got), len(want))
    for i, (rg, rw) in enumerate(zip(got, want)):
        for j, (g, w) in enumerate(zip(rg, rw)):
            test.assertEqual(
                simplify(g - w), 0, f"[{i}][{j}]: получено {g}, ожидалось {w}"
            )


def diag(values):
    n = len(values)
    return [[values[i] if i == j else 0 for j in range(n)] for i in range(n)]


class TestLectureExample2(unittest.TestCase):
    """u_xx + 2u_xy + 5u_yy - 32u = 0 -> эллиптический, beta = (1, 1)."""

    def setUp(self):
        self.res = reduce_quadratic_form(EXAMPLE_2_A)

    def test_beta_is_all_ones(self):
        self.assertEqual([simplify(b) for b in self.res.beta], [1, 1])

    def test_C_matches_lecture(self):
        # mu1 = l1 + l2,  mu2 = 2*l2
        assert_mat(self, self.res.C, [[1, 1], [0, 2]])

    def test_P_is_inverse_of_C(self):
        assert_mat(self, self.res.P, [[1, Rational(-1, 2)], [0, Rational(1, 2)]])


class TestLectureExample1(unittest.TestCase):
    """u_xy - 2u_xz + u_yz -> гиперболический, beta = (1, -1, 1)."""

    def setUp(self):
        self.res = reduce_quadratic_form(EXAMPLE_1_A)

    def test_beta_matches_lecture(self):
        self.assertEqual([simplify(b) for b in self.res.beta], [1, -1, 1])

    def test_C_matches_lecture(self):
        # mu1 = (l1+l2-l3)/2, mu2 = (l1-l2+3l3)/2, mu3 = sqrt(2)*l3
        want = [
            [Rational(1, 2), Rational(1, 2), Rational(-1, 2)],
            [Rational(1, 2), Rational(-1, 2), Rational(3, 2)],
            [0, 0, sqrt(2)],
        ]
        assert_mat(self, self.res.C, want)

    def test_P_matches_lecture(self):
        want = [
            [1, 1, -sqrt(2) / 2],
            [1, -1, sqrt(2)],
            [0, 0, sqrt(2) / 2],
        ]
        assert_mat(self, self.res.P, want)


class TestReductionInvariant(unittest.TestCase):
    """P^T A P = diag(beta) — главный признак правильности приведения."""

    def check(self, a):
        res = reduce_quadratic_form(a)
        got = mat_mul(mat_mul(transpose(res.P), a), res.P)
        assert_mat(self, got, diag(res.beta))

    def test_invariant_for_lecture_example_1(self):
        self.check(EXAMPLE_1_A)

    def test_invariant_for_lecture_example_2(self):
        self.check(EXAMPLE_2_A)

    def test_invariant_for_wave_equation(self):
        self.check([[1, 0], [0, -1]])

    def test_invariant_for_heat_equation(self):
        # u_t = u_xx: старшая часть содержит только u_xx
        self.check([[1, 0], [0, 0]])

    def test_invariant_for_form_without_any_square_terms(self):
        self.check([[0, 1], [1, 0]])

    def test_invariant_for_4d_wave_operator(self):
        self.check([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])


class TestDegenerateForms(unittest.TestCase):
    def test_heat_equation_has_one_zero_beta(self):
        res = reduce_quadratic_form([[1, 0], [0, 0]])
        self.assertEqual([simplify(b) for b in res.beta], [1, 0])

    def test_zero_form_gives_all_zero_beta_and_identity_C(self):
        res = reduce_quadratic_form([[0, 0], [0, 0]])
        self.assertEqual([simplify(b) for b in res.beta], [0, 0])
        assert_mat(self, res.C, [[1, 0], [0, 1]])


class TestWithoutNormalisation(unittest.TestCase):
    """normalize=False оставляет beta рациональными, без радикалов."""

    def test_beta_keeps_rational_values(self):
        res = reduce_quadratic_form(EXAMPLE_2_A, normalize=False)
        self.assertEqual([simplify(b) for b in res.beta], [1, 4])
        assert_mat(self, res.C, [[1, 1], [0, 1]])

    def test_invariant_still_holds(self):
        res = reduce_quadratic_form(EXAMPLE_2_A, normalize=False)
        got = mat_mul(mat_mul(transpose(res.P), EXAMPLE_2_A), res.P)
        assert_mat(self, got, diag(res.beta))


class TestTrace(unittest.TestCase):
    def test_reduction_records_human_readable_steps(self):
        res = reduce_quadratic_form(EXAMPLE_1_A)
        self.assertTrue(res.steps, "ожидался непустой протокол выкладок")
        self.assertTrue(all(isinstance(s, str) for s in res.steps))


if __name__ == "__main__":
    unittest.main()
