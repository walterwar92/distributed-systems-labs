# -*- coding: utf-8 -*-
"""Задание C: уравнения с переменными коэффициентами и разбиение на области.

Главный эталон — уравнение Трикоми y*u_xx + u_yy = 0 (лекция 1, слайды 15-17):
    D = -y;  y < 0 — гиперболический, y = 0 — параболический, y > 0 — эллиптический.
"""
import unittest

from sympy import Rational, simplify, symbols

from pde_canonical.regions import analyse_regions, region_map

x, y = symbols("x y", real=True)


def same_up_to_sign(a, b):
    return simplify(a - b) == 0 or simplify(a + b) == 0


class TestTricomi(unittest.TestCase):
    def setUp(self):
        self.analysis = analyse_regions(y, 0, 1)
        self.by_type = {r.type: r for r in self.analysis.regions}

    def test_discriminant_is_minus_y(self):
        self.assertEqual(simplify(self.analysis.D + y), 0)

    def test_all_three_types_are_present(self):
        self.assertEqual(
            set(self.by_type), {"hyperbolic", "parabolic", "elliptic"}
        )

    def test_hyperbolic_region_lies_below_the_axis(self):
        self.assertLess(self.by_type["hyperbolic"].sample_point[y], 0)

    def test_elliptic_region_lies_above_the_axis(self):
        self.assertGreater(self.by_type["elliptic"].sample_point[y], 0)

    def test_parabolic_region_is_the_axis_itself(self):
        self.assertEqual(self.by_type["parabolic"].sample_point[y], 0)

    def test_every_region_carries_a_human_readable_description(self):
        for region in self.analysis.regions:
            self.assertTrue(region.description.strip())

    def test_hyperbolic_region_reproduces_lecture_characteristics(self):
        res = self.by_type["hyperbolic"].result
        want = (
            x + Rational(2, 3) * (-y) ** Rational(3, 2),
            x - Rational(2, 3) * (-y) ** Rational(3, 2),
        )
        got = (res.xi, res.eta)
        direct = same_up_to_sign(got[0], want[0]) and same_up_to_sign(got[1], want[1])
        swapped = same_up_to_sign(got[0], want[1]) and same_up_to_sign(got[1], want[0])
        self.assertTrue(direct or swapped, f"получено {got}")

    def test_elliptic_region_reproduces_lecture_substitution(self):
        res = self.by_type["elliptic"].result
        want = (x, Rational(2, 3) * y ** Rational(3, 2))
        got = (res.xi, res.eta)
        direct = same_up_to_sign(got[0], want[0]) and same_up_to_sign(got[1], want[1])
        swapped = same_up_to_sign(got[0], want[1]) and same_up_to_sign(got[1], want[0])
        self.assertTrue(direct or swapped, f"получено {got}")

    def test_hyperbolic_reduction_kills_both_square_terms(self):
        res = self.by_type["hyperbolic"].result
        self.assertEqual(simplify(res.new_a11), 0)
        self.assertEqual(simplify(res.new_a22), 0)

    def test_elliptic_reduction_equalises_square_terms(self):
        res = self.by_type["elliptic"].result
        self.assertEqual(simplify(res.new_a12), 0)
        self.assertEqual(simplify(res.new_a11 - res.new_a22), 0)


class TestConstantCoefficients(unittest.TestCase):
    def test_single_region_covers_the_whole_plane(self):
        analysis = analyse_regions(1, 0, -1)
        self.assertEqual(len(analysis.regions), 1)
        self.assertEqual(analysis.regions[0].type, "hyperbolic")

    def test_laplace_has_one_elliptic_region(self):
        analysis = analyse_regions(1, 0, 1)
        self.assertEqual([r.type for r in analysis.regions], ["elliptic"])


class TestBoundaryInTheOtherVariable(unittest.TestCase):
    """u_xx + x*u_yy = 0 -> D = -x, границы идут по вертикали."""

    def setUp(self):
        self.by_type = {r.type: r for r in analyse_regions(1, 0, x).regions}

    def test_discriminant_splits_along_x(self):
        self.assertEqual(set(self.by_type), {"hyperbolic", "parabolic", "elliptic"})

    def test_hyperbolic_region_is_to_the_left(self):
        self.assertLess(self.by_type["hyperbolic"].sample_point[x], 0)

    def test_elliptic_region_is_to_the_right(self):
        self.assertGreater(self.by_type["elliptic"].sample_point[x], 0)


class TestTwoVariableDiscriminant(unittest.TestCase):
    """u_xx + x*y*u_yy = 0 -> D = -x*y, тип меняется по квадрантам."""

    def setUp(self):
        self.analysis = analyse_regions(1, 0, x * y)

    def test_discriminant_is_minus_xy(self):
        self.assertEqual(simplify(self.analysis.D + x * y), 0)

    def test_both_open_types_are_detected(self):
        types = {r.type for r in self.analysis.regions}
        self.assertIn("hyperbolic", types)
        self.assertIn("elliptic", types)

    def test_sample_points_are_consistent_with_their_type(self):
        for region in self.analysis.regions:
            value = self.analysis.D.subs(region.sample_point)
            expected = {"hyperbolic": 1, "parabolic": 0, "elliptic": -1}[region.type]
            with self.subTest(region=region.description):
                self.assertEqual(
                    0 if value == 0 else (1 if value > 0 else -1), expected
                )


class TestRegionMap(unittest.TestCase):
    def test_map_matches_pointwise_classification_for_tricomi(self):
        xs, ys, grid = region_map(y, 0, 1, xlim=(-1, 1), ylim=(-1, 1), n=21)
        self.assertEqual(len(ys), 21)
        self.assertEqual(len(grid), 21)
        self.assertEqual(len(grid[0]), 21)
        for j, yv in enumerate(ys):
            for i in range(len(xs)):
                expected = 0 if yv == 0 else (1 if -yv > 0 else -1)
                self.assertEqual(grid[j][i], expected)

    def test_map_of_constant_coefficients_is_uniform(self):
        _, _, grid = region_map(1, 0, -1, n=9)
        flat = {value for row in grid for value in row}
        self.assertEqual(flat, {1})


if __name__ == "__main__":
    unittest.main()
