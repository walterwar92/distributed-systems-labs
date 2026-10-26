"""Численные процедуры, написанные вручную (без numpy и scipy):
равномерная сетка и составная формула Симпсона."""


def linspace(start, stop, n):
    """n равноотстоящих точек отрезка [start, stop], включая концы."""
    if n == 1:
        return [float(start)]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def simpson_weights(n):
    """Веса составной формулы Симпсона на n равноотстоящих узлах отрезка [-1, 1].

    Интеграл по [c - r, c + r] от g равен r * sum(w[j] * g(c + r*s[j])), s = linspace(-1, 1, n).
    """
    if n < 3 or n % 2 == 0:
        raise ValueError("число узлов должно быть нечётным и не меньше 3")
    h = 2.0 / (n - 1)
    weights = []
    for j in range(n):
        if j == 0 or j == n - 1:
            c = 1.0
        elif j % 2 == 1:
            c = 4.0
        else:
            c = 2.0
        weights.append(c * h / 3.0)
    return weights


class Simpson:
    """Формула Симпсона с n узлами, веса считаются один раз: Simpson(n)(g, lo, hi)."""

    def __init__(self, n):
        self.n = n
        self.s = linspace(-1.0, 1.0, n)
        self.w = simpson_weights(n)

    def __call__(self, g, lo, hi):
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo)
        if r == 0.0:
            return 0.0
        total = 0.0
        for s, w in zip(self.s, self.w):
            total += w * g(c + r * s)
        return r * total


def trapezoid(y, dx):
    """Интеграл таблицы y с шагом dx по формуле трапеций."""
    if len(y) < 2:
        return 0.0
    return dx * (sum(y) - 0.5 * (y[0] + y[-1]))


def adaptive_simpson(g, lo, hi, tol=1e-9, panels=16, depth=40):
    """Адаптивная формула Симпсона: отрезок делится пополам, пока оценка Рунге
    |S_левая + S_правая - S| / 15 не станет меньше допуска. Возле разрывов функции
    шаг автоматически мельчает, на гладких участках остаётся крупным.
    Отрезок сначала режется на panels частей, чтобы не пропустить узкие пики."""

    def simpson(fa, fm, fb, length):
        return length * (fa + 4.0 * fm + fb) / 6.0

    def refine(a, b, fa, fm, fb, whole, eps, level):
        m = 0.5 * (a + b)
        lm, rm = 0.5 * (a + m), 0.5 * (m + b)
        flm, frm = g(lm), g(rm)
        left = simpson(fa, flm, fm, m - a)
        right = simpson(fm, frm, fb, b - m)
        diff = left + right - whole
        if level <= 0 or abs(diff) <= 15.0 * eps:
            return left + right + diff / 15.0
        return (refine(a, m, fa, flm, fm, left, 0.5 * eps, level - 1)
                + refine(m, b, fm, frm, fb, right, 0.5 * eps, level - 1))

    if hi == lo:
        return 0.0
    step = (hi - lo) / panels
    total = 0.0
    for i in range(panels):
        a = lo + i * step
        b = a + step
        fa, fm, fb = g(a), g(0.5 * (a + b)), g(b)
        total += refine(a, b, fa, fm, fb, simpson(fa, fm, fb, step), tol / panels, depth)
    return total
