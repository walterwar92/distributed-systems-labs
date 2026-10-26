# -*- coding: utf-8 -*-
"""Тесты точной матричной арифметики (без numpy.linalg)."""
import unittest

from sympy import Rational, sqrt, simplify

from pde_canonical.exact import (
    SingularMatrixError,
    identity,
    inverse,
    mat_mul,
    mat_vec,
    transpose,
)


def assert_mat(test, got, want):
    test.assertEqual(len(got), len(want), f"разное число строк: {got}")
    for i, (rg, rw) in enumerate(zip(got, want)):
        test.assertEqual(len(rg), len(rw), f"строка {i}: разная длина")
        for j, (g, w) in enumerate(zip(rg, rw)):
            test.assertEqual(
                simplify(g - w), 0, f"элемент [{i}][{j}]: получено {g}, ожидалось {w}"
            )


class TestBasicOps(unittest.TestCase):
    def test_identity_has_ones_on_diagonal_and_zeros_elsewhere(self):
        assert_mat(self, identity(3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_mat_mul_multiplies_two_square_matrices(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        assert_mat(self, mat_mul(a, b), [[19, 22], [43, 50]])

    def test_mat_mul_handles_rectangular_shapes(self):
        a = [[1, 2, 3]]
        b = [[1], [0], [-1]]
        assert_mat(self, mat_mul(a, b), [[-2]])

    def test_transpose_swaps_rows_and_columns(self):
        assert_mat(self, transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])

    def test_mat_vec_applies_matrix_to_column_vector(self):
        self.assertEqual(mat_vec([[1, 2], [3, 4]], [1, -1]), [-1, -1])


class TestInverse(unittest.TestCase):
    def test_inverse_of_lecture_example_2_matrix(self):
        # Пример 2 (Примеры к лекции 1): C = [[1,1],[0,2]] -> P = C^{-1}
        got = inverse([[1, 1], [0, 2]])
        assert_mat(self, got, [[1, Rational(-1, 2)], [0, Rational(1, 2)]])

    def test_inverse_of_lecture_example_1_matrix_with_radicals(self):
        # Пример 1: C = [[1/2,1/2,-1/2],[1/2,-1/2,3/2],[0,0,sqrt(2)]]
        c = [
            [Rational(1, 2), Rational(1, 2), Rational(-1, 2)],
            [Rational(1, 2), Rational(-1, 2), Rational(3, 2)],
            [0, 0, sqrt(2)],
        ]
        want = [
            [1, 1, -sqrt(2) / 2],
            [1, -1, sqrt(2)],
            [0, 0, sqrt(2) / 2],
        ]
        assert_mat(self, inverse(c), want)

    def test_inverse_times_original_is_identity(self):
        a = [[2, Rational(1, 3), 0], [0, 1, 5], [1, 0, -1]]
        assert_mat(self, mat_mul(inverse(a), a), identity(3))

    def test_inverse_raises_on_singular_matrix(self):
        with self.assertRaises(SingularMatrixError):
            inverse([[1, 2], [2, 4]])

    def test_inverse_pivots_past_a_zero_in_the_top_left(self):
        # Требует перестановки строк: без неё наивный Гаусс-Жордан делит на ноль.
        a = [[0, 1], [1, 0]]
        assert_mat(self, inverse(a), [[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
