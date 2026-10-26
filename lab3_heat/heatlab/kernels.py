"""Фундаментальное решение уравнения теплопроводности и функции Грина.

u_t = a^2 u_xx.  Фундаментальное решение (функция Грина на прямой):

    K(x, xi, t) = 1 / (2a sqrt(pi t)) * exp(-(x - xi)^2 / (4 a^2 t)),   t > 0.

Полупрямая x > 0 -- метод отражений:
    I рода  (u(0,t) задано):    G1 = K(x, xi, t) - K(x, -xi, t)   (нечётное продолжение);
    II рода (u_x(0,t) задано):  G2 = K(x, xi, t) + K(x, -xi, t)   (чётное продолжение).
"""
import math

SQRT_PI = math.sqrt(math.pi)


def heat_kernel(x, xi, t, a):
    """K(x, xi, t) -- температура в точке x в момент t от единичного источника в xi при t = 0."""
    if t <= 0.0:
        return 0.0                          # при t -> 0 ядро стремится к дельта-функции
    q = (x - xi) ** 2 / (4.0 * a * a * t)
    if q > 700.0:                           # exp(-700) ~ 1e-304: дальше просто нуль
        return 0.0
    return math.exp(-q) / (2.0 * a * SQRT_PI * math.sqrt(t))


def green_line(x, xi, t, a):
    """Функция Грина задачи Коши на прямой -- само фундаментальное решение."""
    return heat_kernel(x, xi, t, a)


def green_half(x, xi, t, a, kind):
    """Функция Грина полупрямой x > 0: kind = 1 -- условие I рода, kind = 2 -- II рода."""
    image = heat_kernel(x, -xi, t, a)       # источник, отражённый в точку -xi
    return heat_kernel(x, xi, t, a) + (image if kind == 2 else -image)
