"""Тесты pde.py на примерах из лекций (1.pdf, 2.pdf) и методички «Примеры к лекции 1.pdf».

Запуск:  python -X utf8 tools/test_pde.py
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde  # noqa: E402


def same(result, expected, point=None):
    """Символьное (или, если задана точка, численное) равенство результата и строки-эталона."""
    result = sp.sympify(result)
    names = {s.name: s for s in result.free_symbols}
    diff = result - sp.sympify(expected, locals=names)
    if sp.simplify(diff) == 0:
        return True
    if point is None:
        return False
    names = {s.name: s for s in diff.free_symbols}
    subs = {names[k]: v for k, v in point.items() if k in names}
    for i, s in enumerate(s for s in diff.free_symbols if s not in subs):
        subs[s] = sp.Rational(3 + i, 7)
    return abs(sp.N(diff.subs(subs))) < 1e-9


EX1_PARABOLIC = "u_xx - 2*u_xy + u_yy + 9*u_x + 9*u_y - 9*u"
EX1_ELLIPTIC = "u_xx + 2*u_xy + 5*u_yy - 32*u"
EX1_3D = "u_xy - 2*u_xz + u_yz + u_x + u_y/2"


class ParseTest(unittest.TestCase):
    def test_coefficients_of_constant_equation(self):
        p = pde.parse(EX1_PARABOLIC, "x y")
        self.assertEqual(p.A, sp.Matrix([[1, -1], [-1, 1]]))
        self.assertEqual(list(p.b), [9, 9])
        self.assertEqual(p.c, -9)
        self.assertEqual(p.f, 0)

    def test_mixed_derivative_in_any_order_and_free_term(self):
        p = pde.parse("u_yx + x*y", "x y")
        half = sp.Rational(1, 2)
        self.assertEqual(p.A, sp.Matrix([[0, half], [half, 0]]))
        x, y = p.vars
        self.assertEqual(p.f, x * y)

    def test_right_hand_side_parameters_and_functions(self):
        p = pde.parse("u_tt = a**2*u_xx + f(x, t)", "x t")
        x, t = p.vars
        a = sp.Symbol("a", positive=True)
        self.assertEqual(p.A, sp.Matrix([[-a**2, 0], [0, 1]]))
        self.assertEqual(p.f, -sp.Function("f")(x, t))


class ClassifyTest(unittest.TestCase):
    def test_parabolic_example_from_ex1(self):
        r = pde.classify(EX1_PARABOLIC, "x y")
        self.assertEqual(r["D"], 0)
        self.assertEqual(r["type"], "параболический")

    def test_elliptic_example_from_ex1(self):
        self.assertEqual(pde.classify(EX1_ELLIPTIC, "x y")["type"], "эллиптический")

    def test_wave_equation_is_hyperbolic(self):
        self.assertEqual(pde.classify("u_tt = a**2*u_xx", "x t")["type"], "гиперболический")

    def test_tricomi_type_depends_on_region(self):
        r = pde.classify("y*u_xx + u_yy", "x y")
        x, y = r["vars"]
        self.assertEqual(r["D"], -y)
        self.assertIsNone(r["type"])
        self.assertTrue(r["regions"]["гиперболический"].subs(y, -1))
        self.assertFalse(r["regions"]["гиперболический"].subs(y, 1))
        self.assertTrue(r["regions"]["параболический"].subs(y, 0))
        self.assertTrue(r["regions"]["эллиптический"].subs(y, 2))

    def test_tricomi_with_sign_assumption(self):
        y = sp.Symbol("y", negative=True)
        self.assertEqual(pde.classify("y*u_xx + u_yy", ["x", y])["type"], "гиперболический")

    def test_three_variables_by_signature(self):
        r = pde.classify(EX1_3D, "x y z")
        self.assertEqual(r["inertia"], (2, 1, 0))
        self.assertEqual(r["type"], "гиперболический")

    def test_three_variables_parabolic(self):
        self.assertEqual(pde.classify("u_xx + 2*u_xy + u_yy + u_zz", "x y z")["type"], "параболический")


class ChangeVarsTest(unittest.TestCase):
    def test_linear_change_ex1_section3(self):
        eq = pde.change_vars(EX1_PARABOLIC, "x y", {"ξ": "x + y", "η": "x"})
        self.assertTrue(same(eq, "v_ηη + 18*v_ξ + 9*v_η - 9*v"))

    def test_tricomi_elliptic_region(self):
        x, y = sp.Symbol("x", real=True), sp.Symbol("y", positive=True)
        eq = pde.change_vars("y*u_xx + u_yy", [x, y], {"ξ": "x", "η": "2*y**(3/2)/3"},
                             inverse={"x": "ξ", "y": "(3*η/2)**(2/3)"}, normalize=True)
        self.assertTrue(same(eq, "v_ξξ + v_ηη + v_η/(3*η)", point={"ξ": 1, "η": 2}))

    def test_tricomi_hyperbolic_region_sign(self):
        x, y = sp.Symbol("x", real=True), sp.Symbol("y", negative=True)
        eq = pde.change_vars("y*u_xx + u_yy", [x, y],
                             {"ξ": "x + 2*(-y)**(3/2)/3", "η": "x - 2*(-y)**(3/2)/3"},
                             inverse={"x": "(ξ + η)/2", "y": "-(3*(ξ - η)/4)**(2/3)"}, normalize=True)
        self.assertTrue(same(eq, "v_ξη - (v_ξ - v_η)/(6*(ξ - η))", point={"ξ": 2, "η": 1}))

    def test_first_to_second_hyperbolic_form(self):
        eq = pde.change_vars("u_xy", "x y", {"ξ": "x + y", "η": "x - y"})
        self.assertTrue(same(eq, "v_ξξ - v_ηη"))


class Canonical2Test(unittest.TestCase):
    def test_hyperbolic_constant_coefficients(self):
        r = pde.canonical2("u_xx - 5*u_xy + 6*u_yy", "x y")
        x, y = r["vars"]
        self.assertEqual(r["type"], "гиперболический")
        self.assertEqual({sp.expand(e) for e in r["new_vars"].values()}, {y + 2 * x, y + 3 * x})
        self.assertTrue(same(r["equation"], "v_ξη"))

    def test_parabolic_ex1_section3(self):
        r = pde.canonical2(EX1_PARABOLIC, "x y")
        x, y = r["vars"]
        self.assertEqual(r["type"], "параболический")
        self.assertEqual([sp.expand(e) for e in r["new_vars"].values()], [x + y, x])
        self.assertTrue(same(r["equation"], "v_ηη + 18*v_ξ + 9*v_η - 9*v"))

    def test_elliptic_constant_coefficients(self):
        r = pde.canonical2(EX1_ELLIPTIC, "x y")
        self.assertEqual(r["type"], "эллиптический")
        self.assertTrue(same(r["equation"], "v_ξξ + v_ηη - 8*v"))

    def test_zero_coefficient_at_u_xx(self):
        r = pde.canonical2("u_xy + u_yy", "x y")
        self.assertEqual(r["type"], "гиперболический")
        self.assertTrue(same(r["equation"], "v_ξη"))

    def test_variable_coefficients(self):
        x, y = sp.Symbol("x", positive=True), sp.Symbol("y", positive=True)
        r = pde.canonical2("x**2*u_xx - y**2*u_yy", [x, y])
        self.assertEqual(r["type"], "гиперболический")
        self.assertEqual([sp.simplify(e) for e in r["new_vars"].values()], [y / x, x * y])
        self.assertTrue(same(r["equation"], "v_ξη - v_ξ/(2*η)", point={"ξ": 2, "η": 3}))


class CanonicalNTest(unittest.TestCase):
    def test_ex1_section2_example1(self):
        r = pde.canonical_n(EX1_3D, "x y z")
        A = pde.parse(EX1_3D, "x y z").A
        self.assertEqual(sorted(r["betas"]), [-1, 1, 1])
        self.assertEqual(r["type"], "гиперболический")
        self.assertEqual(sp.simplify(r["B"] * A * r["B"].T), sp.diag(*r["betas"]))
        x = sp.Matrix(r["vars"])
        by_formula = pde.change_vars(EX1_3D, "x y z", dict(zip(["ξ", "η", "ω"], r["B"] * x)))
        self.assertTrue(same(r["equation"], str(by_formula)))

    def test_ex1_section2_example2_matches_methodichka(self):
        r = pde.canonical_n(EX1_ELLIPTIC, "x y")
        x, y = r["vars"]
        self.assertEqual(r["betas"], [1, 1])
        self.assertEqual(r["B"], sp.Matrix([[1, 0], [-sp.Rational(1, 2), sp.Rational(1, 2)]]))
        self.assertTrue(same(r["equation"], "v_ξξ + v_ηη - 32*v"))


class RemoveLowerTest(unittest.TestCase):
    def test_parabolic_corrected_ex1_section4(self):
        r = pde.remove_lower("u_ηη + 18*u_ξ + 9*u_η - 9*u", "ξ η")
        xi, eta = r["vars"]
        self.assertEqual(sp.expand(r["exponent"] - (sp.Rational(13, 8) * xi - sp.Rational(9, 2) * eta)), 0)
        self.assertTrue(same(r["equation"], "w_ηη + 18*w_ξ"))

    def test_elliptic(self):
        r = pde.remove_lower("u_xx + u_yy + 2*u_x - 4*u_y + u", "x y")
        x, y = r["vars"]
        self.assertEqual(sp.expand(r["exponent"] - (-x + 2 * y)), 0)
        self.assertTrue(same(r["equation"], "w_xx + w_yy - 4*w"))

    def test_hyperbolic_first_form(self):
        r = pde.remove_lower("u_xy + 2*u_x + 3*u_y + u", "x y")
        x, y = r["vars"]
        self.assertEqual(sp.expand(r["exponent"] - (-3 * x - 2 * y)), 0)
        self.assertTrue(same(r["equation"], "w_xy - 5*w"))


class CheckTest(unittest.TestCase):
    def test_lecture_example_is_solution(self):
        self.assertEqual(pde.check("y*u_x - x*u_y", "phi(x**2 + y**2)", "x y"), 0)

    def test_general_solution_of_hyperbolic_equation(self):
        self.assertEqual(pde.check("u_xx - 5*u_xy + 6*u_yy", "F(y + 2*x) + G(y + 3*x)", "x y"), 0)

    def test_not_a_solution_gives_residual(self):
        self.assertEqual(pde.check("u_xx + u_yy", "x**2", "x y"), 2)


class DalembertTest(unittest.TestCase):
    def test_initial_displacement_only(self):
        self.assertTrue(same(pde.dalembert("sin(x)", "0", "a"), "sin(x)*cos(a*t)"))

    def test_initial_velocity_only(self):
        self.assertTrue(same(pde.dalembert("0", "cos(x)", 2), "cos(x)*sin(2*t)/2"))

    def test_solution_satisfies_cauchy_problem(self):
        u = pde.dalembert("exp(-x**2)", "x/(1 + x**2)", 3)
        x, t = sp.Symbol("x", real=True), sp.Symbol("t", real=True)
        self.assertEqual(pde.check("u_tt - 9*u_xx", u, "x t"), 0)
        self.assertEqual(sp.simplify(u.subs(t, 0) - sp.exp(-x**2)), 0)
        self.assertEqual(sp.simplify(sp.diff(u, t).subs(t, 0) - x / (1 + x**2)), 0)

    def test_inhomogeneous_equation(self):
        u = pde.dalembert("0", "0", 1, f="x")
        self.assertTrue(same(u, "x*t**2/2"))


class FourierTest(unittest.TestCase):
    def test_sine_coefficients_of_x_on_0_pi(self):
        b = pde.sine_coeff("x", "pi")
        n = next(s for s in b.free_symbols if s.name == "n")
        self.assertEqual(sp.simplify(b - 2 * (-1)**(n + 1) / n), 0)

    def test_cosine_coefficients_of_x_on_0_pi(self):
        a = pde.cosine_coeff("x", "pi")
        n = next(s for s in a.free_symbols if s.name == "n")
        self.assertEqual(sp.simplify(a - 2 * ((-1)**n - 1) / (sp.pi * n**2)), 0)
        self.assertEqual(pde.cosine_coeff("x", "pi", n=0), sp.pi)


class CliTest(unittest.TestCase):
    def test_classify_command_prints_type(self):
        out = io.StringIO()
        with redirect_stdout(out):
            pde.main(["classify", "u_xx - u_yy", "--vars", "x y"])
        self.assertIn("гиперболический", out.getvalue())

    def test_canonical_command_with_sign_option(self):
        out = io.StringIO()
        with redirect_stdout(out):
            pde.main(["canonical", "x**2*u_xx - y**2*u_yy", "--vars", "x y", "--pos", "x y"])
        self.assertIn("v_ξη", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
