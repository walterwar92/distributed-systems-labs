"""pde.py — проверка выкладок по ДУЧП 2-го порядка (sympy).

Уравнения записываются как в лекциях: u_xx, u_xy (= u_yx), u_x, u; правую часть можно отделить «=».
Переменные — имена из vars ("x y", "x t", "ξ η"), они вещественные; Symbol с допущениями тоже подходит.
Прочие имена — параметры (> 0). Имя со скобкой — произвольная функция: f(x, t), phi(x**2 + y**2).
Умножение только через «*».

Python (из папки tools):  from pde import *
    classify(eq, vars)                    тип: D для 2 переменных, инерция формы для n
    canonical2(eq, vars)                  канонический вид через характеристики (2 переменные)
    canonical_n(eq, vars)                 канонический вид через квадратичную форму (пост. коэф.)
    change_vars(eq, vars, new, inverse)   произвольная замена ξ = ξ(x, y, …)
    remove_lower(eq, vars)                u = w·exp(α·x) — убрать младшие производные
    check(eq, u, vars)                    подставить функцию в уравнение → невязка
    dalembert(phi, psi, a, f)             формула Даламбера (+ Дюамель для f)
    sine_coeff(f, l), cosine_coeff(f, l)  коэффициенты Фурье на (0, l)
    show(expr)                            уравнение строкой «старшие → младшие = 0»

CLI:  python -X utf8 tools/pde.py --help
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from itertools import combinations_with_replacement

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

HYP, PAR, ELL = "гиперболический", "параболический", "эллиптический"
NEW_NAMES = ("ξ", "η", "ω", "ζ")

_TRANSFORMS = standard_transformations + (convert_xor,)
_IDENT = re.compile(r"[^\W\d]\w*")
_GLOBALS: dict = {}
exec("from sympy import *", _GLOBALS)
# однобуквенные и «греческие» имена sympy, которые в задачах означают параметры (E — модуль Юнга, S — площадь…)
for _name in ("E", "S", "N", "O", "Q", "C", "beta", "gamma", "zeta", "li", "Li", "re", "im", "test", "var"):
    _GLOBALS.pop(_name, None)


@dataclass
class PDE:
    """Σ A_ij·u_{x_i x_j} + Σ b_i·u_{x_i} + c·u + f = 0, матрица A симметрична."""
    vars: tuple
    A: sp.Matrix
    b: sp.Matrix
    c: sp.Expr
    f: sp.Expr


def _var(name):
    return sp.Symbol(name, real=True)


def _vars(vars):
    if isinstance(vars, str):
        vars = vars.replace(",", " ").split()
    return tuple(v if isinstance(v, sp.Symbol) else _var(str(v)) for v in vars)


def _expr(value, symbols=(), extra=None):
    """Строка / число / sympy → sympy; символы из symbols подставляются по имени."""
    by_name = {s.name: s for s in symbols}
    if isinstance(value, sp.Basic):
        return value.xreplace({s: by_name[s.name] for s in value.free_symbols if s.name in by_name})
    if isinstance(value, (int, float)):
        return sp.nsimplify(value)
    text = str(value)
    local = {**by_name, **(extra or {})}
    for m in _IDENT.finditer(text):
        name = m.group()
        if name not in local and name not in _GLOBALS and not text[m.end():].lstrip().startswith("("):
            local[name] = sp.Symbol(name, positive=True)
    return parse_expr(text, local_dict=local, global_dict=_GLOBALS, transformations=_TRANSFORMS)


def _is_zero(e):
    return sp.simplify(e) == 0


def _sign(e):
    e = sp.simplify(e)
    if e.is_zero:
        return 0
    if e.is_positive:
        return 1
    if e.is_negative:
        return -1
    return None


def parse(eq, vars="x y", func="u"):
    """Разобрать линейное (квазилинейное) уравнение 2-го порядка в коэффициенты PDE."""
    X = _vars(vars)
    names = [s.name for s in X]
    text = str(eq)
    if "=" in text:
        lhs, rhs = text.split("=")
        text = f"({lhs}) - ({rhs})"
    holders = {}

    def indices(suffix):
        out, pos = [], 0
        while pos < len(suffix):
            for j in sorted(range(len(names)), key=lambda j: -len(names[j])):
                if suffix.startswith(names[j], pos):
                    out.append(j)
                    pos += len(names[j])
                    break
            else:
                raise ValueError(f"в {func}_{suffix} неизвестная переменная (переменные: {' '.join(names)})")
        return tuple(sorted(out))

    def placeholder(m):
        key = indices(m.group(1)) if m.group(1) else ()
        if len(key) > 2:
            raise ValueError("поддерживаются производные не выше второго порядка")
        name = "_U" + "".join(map(str, key))
        holders[key] = sp.Symbol(name)
        return name

    text = re.sub(rf"(?<!\w){re.escape(func)}(?:_(\w+))?(?![\w(])", placeholder, text)
    expr = _expr(text, X, {s.name: s for s in holders.values()})
    hs = list(holders.values())
    for key, s in holders.items():
        if len(key) == 2 and any(sp.diff(expr, s, h) != 0 for h in hs):
            raise ValueError("уравнение нелинейно относительно старших производных")

    def coeff(key):
        s = holders.get(key)
        return sp.simplify(sp.diff(expr, s)) if s is not None else sp.Integer(0)

    n = len(X)
    A = sp.zeros(n, n)
    for i in range(n):
        A[i, i] = coeff((i, i))
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = coeff((i, j)) / 2
    b = sp.Matrix([coeff((i,)) for i in range(n)])
    return PDE(X, A, b, coeff(()), sp.simplify(expr.subs({s: 0 for s in hs})))


def _congruence(A):
    """Метод Лагранжа в матричной форме: Tᵀ·A·T = diag(d)."""
    A = sp.Matrix(A)
    n = A.rows
    T = sp.eye(n)

    def apply(E):
        nonlocal A, T
        A = (E.T * A * E).applyfunc(sp.simplify)
        T = T * E

    for k in range(n):
        if _is_zero(A[k, k]):
            j = next((j for j in range(k + 1, n) if not _is_zero(A[j, j])), None)
            if j is not None:  # переставить переменные k и j
                E = sp.eye(n)
                E[k, k] = E[j, j] = 0
                E[k, j] = E[j, k] = 1
                apply(E)
            else:  # квадратов нет: λ_j → λ_j + λ_k даёт квадрат 2·a_kj·λ_k²
                j = next((j for j in range(k + 1, n) if not _is_zero(A[k, j])), None)
                if j is None:
                    continue
                E = sp.eye(n)
                E[j, k] = 1
                apply(E)
        E = sp.eye(n)
        for i in range(k + 1, n):
            E[k, i] = -A[k, i] / A[k, k]
        apply(E)
    return [A[i, i] for i in range(n)], T


def _type_by_inertia(pos, neg, zero):
    if pos + neg == 0:
        return None
    if zero == 0:
        return ELL if pos == 0 or neg == 0 else HYP
    return PAR if pos == 0 or neg == 0 else None


def _regions(D):
    rels = {HYP: D > 0, PAR: sp.Eq(D, 0), ELL: D < 0}
    free = list(D.free_symbols)
    if len(free) != 1:
        return rels
    s = free[0]
    out = {}
    for k, rel in rels.items():
        try:
            out[k] = sp.solveset(rel, s, sp.S.Reals).as_relational(s)
        except Exception:
            out[k] = rel
    return out


def classify(eq, vars="x y", func="u"):
    """Тип уравнения. 2 переменные: D = a12² − a11·a22 (и области, если знак D не определён);
    n переменных: инерция (+, −, 0) квадратичной формы."""
    p = parse(eq, vars, func)
    res = {"vars": p.vars, "A": p.A}
    if len(p.vars) == 2:
        D = sp.simplify(p.A[0, 1] ** 2 - p.A[0, 0] * p.A[1, 1])
        res["D"] = D
        res["type"] = None if p.A.is_zero_matrix else {1: HYP, 0: PAR, -1: ELL}.get(_sign(D))
        if res["type"] is None and not p.A.is_zero_matrix:
            res["regions"] = _regions(D)
        return res
    d, _ = _congruence(p.A)
    signs = [_sign(v) for v in d]
    res["diagonal"] = d
    if None in signs:
        res["type"] = None
    else:
        res["inertia"] = (signs.count(1), signs.count(-1), signs.count(0))
        res["type"] = _type_by_inertia(*res["inertia"])
    return res


def _inverse(X, Y, exprs, inverse):
    if inverse is not None:
        by_name = {s.name: s for s in X}
        return {by_name[getattr(k, "name", str(k))]: _expr(v, Y) for k, v in inverse.items()}
    try:
        sols = sp.solve([sp.Eq(y, e) for y, e in zip(Y, exprs)], list(X), dict=True)
    except NotImplementedError:
        return None
    return sols[0] if sols else None


def change_vars(eq, vars, new, inverse=None, func="u", new_func="v", normalize=False):
    """Замена переменных u(x) = v(ξ(x)).

    new      — {'ξ': 'x + y', 'η': 'x'} (ключи — имена или Symbol, значения — выражения от старых переменных);
    inverse  — {'x': …, 'y': …} через новые переменные; нужна, только если коэффициенты зависят от x
               (без неё обратная замена ищется через solve; если не найдена, коэффициенты остаются в x);
    normalize — разделить на коэффициент первой ненулевой старшей производной (v_ξξ, v_ξη, v_ηη, …).
    """
    p = parse(eq, vars, func)
    X, n = p.vars, len(p.vars)
    if len(new) != n:
        raise ValueError("число новых переменных должно совпадать с числом старых")
    Y = tuple(k if isinstance(k, sp.Symbol) else sp.Symbol(str(k)) for k in new)
    exprs = [_expr(v, X) for v in new.values()]
    names = [s.name for s in Y]
    V2 = {(k, m): sp.Symbol(f"{new_func}_{names[k]}{names[m]}") for k, m in combinations_with_replacement(range(n), 2)}
    V1 = [sp.Symbol(f"{new_func}_{nm}") for nm in names]
    V0 = sp.Symbol(new_func)
    J = [[sp.diff(e, xi) for xi in X] for e in exprs]

    def d1(i):
        return sum(V1[k] * J[k][i] for k in range(n))

    def d2(i, j):
        return (sum(V2[min(k, m), max(k, m)] * J[k][i] * J[m][j] for k in range(n) for m in range(n))
                + sum(V1[k] * sp.diff(exprs[k], X[i], X[j]) for k in range(n)))

    out = (sum(p.A[i, j] * d2(i, j) for i in range(n) for j in range(n))
           + sum(p.b[i] * d1(i) for i in range(n)) + p.c * V0 + p.f)
    syms = list(V2.values()) + V1 + [V0]
    coeffs = {s: sp.simplify(sp.diff(out, s)) for s in syms}
    coeffs[1] = sp.simplify(out.subs({s: 0 for s in syms}))
    if normalize:
        lead = next((coeffs[s] for s in V2.values() if not _is_zero(coeffs[s])), None)
        if lead is not None:
            coeffs = {s: sp.simplify(c / lead) for s, c in coeffs.items()}
    if any(c.free_symbols & set(X) for c in coeffs.values()):
        inv = _inverse(X, Y, exprs, inverse)
        if inv:
            coeffs = {s: sp.simplify(c.subs(inv)) for s, c in coeffs.items()}
    return sum(c * s for s, c in coeffs.items())


def _first_integral(slope, s, t):
    """Общий интеграл φ(s, t) = C уравнения dt/ds = slope(s, t)."""
    if not slope.free_symbols & {s, t}:
        return sp.expand(t - slope * s)
    T = sp.Function("T")
    sol = sp.dsolve(sp.Eq(T(s).diff(s), slope.subs(t, T(s))), T(s))
    C1 = sp.Symbol("C1")
    for eq in (sol if isinstance(sol, list) else [sol]):
        consts = sp.solve((eq.lhs - eq.rhs).subs(T(s), t), C1)
        if consts:
            return sp.simplify(consts[0])
    raise ValueError(f"не удалось проинтегрировать d{t}/d{s} = {slope}")


def canonical2(eq, vars="x y", func="u", new_func="v", names=("ξ", "η")):
    """Канонический вид для двух переменных через характеристики a11·dy² − 2a12·dx·dy + a22·dx² = 0."""
    p = parse(eq, vars, func)
    X = p.vars
    if len(X) != 2:
        raise ValueError("canonical2 — для двух переменных; для n переменных — canonical_n")
    x, y = X
    a11, a12, a22 = p.A[0, 0], p.A[0, 1], p.A[1, 1]
    D = sp.simplify(a12 ** 2 - a11 * a22)
    typ = {1: HYP, 0: PAR, -1: ELL}.get(_sign(D))
    if typ is None or p.A.is_zero_matrix:
        raise ValueError(f"D = {D}: тип зависит от области — задайте знаки переменных "
                         f"(Symbol('y', negative=True) или --neg y)")
    root = sp.sqrt(D)
    if not _is_zero(a11):  # dy/dx = (a12 ± √D)/a11
        slopes = [sp.simplify((a12 + root) / a11), sp.simplify((a12 - root) / a11)]
        chars = [_first_integral(k, x, y) for k in slopes]
    elif not _is_zero(a22):  # dx/dy = (a12 ± √D)/a22
        slopes = [sp.simplify((a12 + root) / a22), sp.simplify((a12 - root) / a22)]
        chars = [_first_integral(k, y, x) for k in slopes]
    else:  # уже вид u_xy
        slopes, chars = [], [x, y]
    if typ == HYP:
        xi, eta = chars
    elif typ == PAR:
        xi = chars[0]
        eta = x if not _is_zero(sp.diff(xi, y)) else y
    else:
        phi = sp.expand(chars[0])
        xi, eta = sp.simplify(sp.re(phi)), sp.simplify(sp.im(phi))
        if eta.could_extract_minus_sign():
            eta = -eta
    positive = all(s.is_positive for s in X)
    Y = [sp.Symbol(nm, positive=True) if positive else sp.Symbol(nm) for nm in names]
    new = {Y[0]: xi, Y[1]: eta}
    return {"vars": X, "type": typ, "D": D, "slopes": slopes, "characteristics": chars, "new_vars": new,
            "equation": change_vars(eq, X, new, func=func, new_func=new_func, normalize=True)}


def canonical_n(eq, vars, func="u", new_func="v", names=None):
    """Канонический вид через квадратичную форму: λ = Pμ, Q = Σβ_i·μ_i², замена ξ = Pᵀx (пост. коэф.)."""
    p = parse(eq, vars, func)
    X, n = p.vars, len(p.vars)
    if any(e.free_symbols & set(X) for e in p.A):
        raise ValueError("метод квадратичной формы — только для постоянных коэффициентов при старших производных")
    d, T = _congruence(p.A)
    betas, scale = [], []
    for v in d:
        sgn = _sign(v)
        if sgn is None:
            raise ValueError(f"не удалось определить знак {v}")
        betas.append(sgn)
        scale.append(1 / sp.sqrt(sp.Abs(v)) if sgn else 1)
    P = (T * sp.diag(*scale)).applyfunc(sp.radsimp)
    B = P.T
    names = list(names or (NEW_NAMES[:n] if n <= len(NEW_NAMES) else [f"ξ{i + 1}" for i in range(n)]))
    Y = [sp.Symbol(nm) for nm in names]
    new = dict(zip(Y, B * sp.Matrix(X)))
    return {"vars": X, "betas": betas, "P": P, "B": B, "new_vars": new,
            "type": _type_by_inertia(betas.count(1), betas.count(-1), betas.count(0)),
            "equation": change_vars(eq, X, new, func=func, new_func=new_func)}


def remove_lower(eq, vars, func="u", new_func="w"):
    """u = w·exp(Σα_i·x_i): новые b̄ = 2Aα + b, c̄ = αᵀAα + bᵀα + c. Убираются все первые производные,
    которые можно убрать; если какую-то нельзя (параболический случай) — обнуляется коэффициент при w."""
    p = parse(eq, vars, func)
    X, n = p.vars, len(p.vars)
    if any(e.free_symbols & set(X) for e in list(p.A) + list(p.b) + [p.c]):
        raise ValueError("нужны постоянные коэффициенты")
    al = sp.symbols(f"_a0:{n}")
    alpha = sp.Matrix(al)
    first = [sp.expand(2 * (p.A * alpha)[j] + p.b[j]) for j in range(n)]
    zero_order = sp.expand((alpha.T * p.A * alpha)[0] + (p.b.T * alpha)[0] + p.c)
    sol = sp.solve(first, al, dict=True)
    if not sol:
        keep = [first[j] for j in range(n) if not all(_is_zero(p.A[j, i]) for i in range(n))]
        sol = sp.solve(keep + [zero_order], al, dict=True)
    chosen = sol[0] if sol else {}
    values = sp.Matrix([sp.sympify(chosen.get(a, 0)).subs({b: 0 for b in al}) for a in al])
    exponent = sum(values[i] * X[i] for i in range(n))
    names = [s.name for s in X]
    new_b = 2 * p.A * values + p.b
    new_c = sp.simplify((values.T * p.A * values)[0] + (p.b.T * values)[0] + p.c)
    equation = (sum(p.A[i, j] * (1 if i == j else 2) * sp.Symbol(f"{new_func}_{names[i]}{names[j]}")
                    for i, j in combinations_with_replacement(range(n), 2))
                + sum(sp.simplify(new_b[i]) * sp.Symbol(f"{new_func}_{names[i]}") for i in range(n))
                + new_c * sp.Symbol(new_func) + sp.simplify(p.f * sp.exp(-exponent)))
    return {"vars": X, "alphas": list(values), "exponent": exponent, "equation": equation,
            "substitution": f"{func} = {new_func}·exp({exponent})"}


def check(eq, u, vars="x y", func="u"):
    """Подставить u в уравнение; 0 — значит решение."""
    p = parse(eq, vars, func)
    X, n = p.vars, len(p.vars)
    U = _expr(u, X)
    res = (sum(p.A[i, j] * sp.diff(U, X[i], X[j]) for i in range(n) for j in range(n))
           + sum(p.b[i] * sp.diff(U, X[i]) for i in range(n)) + p.c * U + p.f)
    return sp.simplify(res)


def dalembert(phi, psi, a, f=None, x="x", t="t"):
    """u_tt = a²u_xx + f, u(x,0) = φ, u_t(x,0) = ψ на всей прямой."""
    X, T = _var(x), _var(t)
    A = _expr(a, (X, T))
    s, tau = sp.Dummy("s"), sp.Dummy("tau")
    Phi, Psi = _expr(phi, (X, T)), _expr(psi, (X, T))
    u = (Phi.subs(X, X - A * T) + Phi.subs(X, X + A * T)) / 2
    u += sp.integrate(Psi.subs(X, s), (s, X - A * T, X + A * T)) / (2 * A)
    if f is not None:
        F = _expr(f, (X, T))
        inner = sp.integrate(F.subs({X: s, T: tau}), (s, X - A * (T - tau), X + A * (T - tau)))
        u += sp.integrate(inner, (tau, 0, T)) / (2 * A)
    return sp.simplify(u)


def _fourier(f, l, x, n, kernel):
    X = _var(x)
    L = _expr(l, (X,))
    N = sp.Symbol("n", integer=True, positive=True) if n is None else sp.nsimplify(n)
    return sp.simplify(2 / L * sp.integrate(_expr(f, (X,)) * kernel(N * sp.pi * X / L), (X, 0, L)))


def sine_coeff(f, l, x="x", n=None):
    """b_n = (2/l)∫₀ˡ f·sin(nπx/l)dx (n — целое > 0; можно задать число)."""
    return _fourier(f, l, x, n, sp.sin)


def cosine_coeff(f, l, x="x", n=None):
    """a_n = (2/l)∫₀ˡ f·cos(nπx/l)dx; n=0 даёт a₀ (ряд a₀/2 + Σ a_n cos)."""
    return _fourier(f, l, x, n, sp.cos)


def show(expr):
    """Уравнение строкой: слагаемые по убыванию порядка производной, «= 0» в конце."""
    expr = sp.sympify(expr)

    def order(term):
        best = 0
        for s in term.free_symbols:
            m = re.fullmatch(r"[^\W\d_]+_(\w+)", s.name)
            if m:
                best = max(best, len(m.group(1)) + 1)
            elif s.name in ("u", "v", "w", "g"):
                best = max(best, 1)
        return -best

    terms = sorted(sp.Add.make_args(expr), key=order)
    return " + ".join(sp.sstr(t) for t in terms).replace("+ -", "- ") + " = 0"


def _cli_vars(args):
    pos = set(args.pos.replace(",", " ").split())
    neg = set(args.neg.replace(",", " ").split())
    out = []
    for nm in args.vars.replace(",", " ").split():
        if nm in pos:
            out.append(sp.Symbol(nm, positive=True))
        elif nm in neg:
            out.append(sp.Symbol(nm, negative=True))
        else:
            out.append(_var(nm))
    return out


def _pairs(items):
    return dict(item.split("=", 1) for item in items) if items else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pde.py", description="Проверка выкладок по ДУЧП 2-го порядка (sympy).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def eq_command(name, help_text):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("eq", help='уравнение, например "u_xx - 2*u_xy + u_yy + 9*u_x = 0"')
        cmd.add_argument("--vars", default="x y", help='переменные (по умолчанию "x y")')
        cmd.add_argument("--pos", default="", help="переменные > 0")
        cmd.add_argument("--neg", default="", help="переменные < 0")
        cmd.add_argument("--func", default="u", help="имя неизвестной функции")
        return cmd

    eq_command("classify", "тип уравнения (D или инерция)")
    eq_command("canonical", "канонический вид").add_argument(
        "--lagrange", action="store_true", help="через квадратичную форму и для 2 переменных")
    change = eq_command("change", "замена переменных")
    change.add_argument("--new", nargs="+", required=True, help="ξ=x+y η=x")
    change.add_argument("--inverse", nargs="+", help="x=η y=ξ-η")
    change.add_argument("--normalize", action="store_true", help="разделить на коэффициент старшей производной")
    eq_command("lower", "убрать младшие производные: u = w·exp(α·x)")
    eq_command("check", "подставить функцию в уравнение").add_argument("u", help='например "phi(x**2 + y**2)"')
    dal = sub.add_parser("dalembert", help="формула Даламбера")
    dal.add_argument("phi")
    dal.add_argument("psi")
    dal.add_argument("--a", default="a")
    dal.add_argument("--f", default=None, help="правая часть f(x, t)")
    for name, word in (("sine", "синусам"), ("cosine", "косинусам")):
        cmd = sub.add_parser(name, help=f"коэффициенты ряда Фурье по {word} на (0, l)")
        cmd.add_argument("f")
        cmd.add_argument("--l", default="l")
        cmd.add_argument("--n", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "dalembert":
        print("u(x, t) =", dalembert(args.phi, args.psi, args.a, args.f))
        return
    if args.cmd in ("sine", "cosine"):
        fn = sine_coeff if args.cmd == "sine" else cosine_coeff
        print(("b_n = " if args.cmd == "sine" else "a_n = ") + str(fn(args.f, args.l, n=args.n)))
        return
    X = _cli_vars(args)
    if args.cmd == "classify":
        r = classify(args.eq, X, args.func)
        if "D" in r:
            print("D = a12² − a11·a22 =", r["D"])
        if "inertia" in r:
            print("инерция (+, −, 0):", r["inertia"])
        if r["type"]:
            print("тип:", r["type"])
        for k, v in r.get("regions", {}).items():
            print(f"{k}: {v}")
    elif args.cmd == "canonical":
        if len(X) == 2 and not args.lagrange:
            r = canonical2(args.eq, X, args.func)
            print("тип:", r["type"], "| D =", r["D"])
            print("характеристики:", "; ".join(f"{c} = C" for c in dict.fromkeys(r["characteristics"])))
        else:
            r = canonical_n(args.eq, X, args.func)
            print("тип:", r["type"], "| β =", r["betas"])
            print("P =", r["P"].tolist())
            print("B = Pᵀ =", r["B"].tolist())
        for s, e in r["new_vars"].items():
            print(f"{s} = {e}")
        print(show(r["equation"]))
    elif args.cmd == "change":
        print(show(change_vars(args.eq, X, _pairs(args.new), _pairs(args.inverse), args.func,
                               normalize=args.normalize)))
    elif args.cmd == "lower":
        r = remove_lower(args.eq, X, args.func)
        print(r["substitution"])
        print(show(r["equation"]))
    elif args.cmd == "check":
        res = check(args.eq, args.u, X, args.func)
        print("невязка:", res, "→ решение" if res == 0 else "→ НЕ решение")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
