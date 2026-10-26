"""Численные процедуры, написанные вручную (без numpy и scipy):
равномерная сетка, формулы трапеций и Симпсона, линейная интерполяция по таблице."""


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


def cumtrapz(y, dx):
    """Интеграл с переменным верхним пределом по формуле трапеций: out[k] = int_0^{k*dx} y."""
    out = [0.0] * len(y)
    for k in range(1, len(y)):
        out[k] = out[k - 1] + 0.5 * dx * (y[k - 1] + y[k])
    return out


def interp(s, dz, table):
    """Линейная интерполяция таблицы table[k] = T(k*dz) в точке s >= 0."""
    pos = s / dz
    k = int(pos)
    if k >= len(table) - 1:
        return table[-1]
    return table[k] + (table[k + 1] - table[k]) * (pos - k)
