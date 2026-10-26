"""Превращение строки, введённой пользователем, в функцию: "exp(-(x-3)**2)" -> phi(x)."""
import math


def theta(s):
    """Функция Хевисайда."""
    return 1.0 if s >= 0 else 0.0


def sign(s):
    return (s > 0) - (s < 0)


# имена, разрешённые в выражениях; элементарные функции -- из стандартного модуля math
_NAMES = {name: getattr(math, name) for name in (
    "sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "atan", "pi")}
_NAMES.update(abs=abs, maximum=max, minimum=min, theta=theta, sign=sign)


def make_func(text, variables="x"):
    """Функция от переменных `variables` (через пробел), заданная выражением `text`.

    Допустимы числа, + - * / ** (или ^), скобки и имена из _NAMES.
    Функция принимает и возвращает обычные числа.
    """
    if "__" in text:
        raise ValueError("недопустимое выражение")
    names = variables.split()
    try:
        func = eval(f"lambda {', '.join(names)}: ({text.replace('^', '**')})",
                    {"__builtins__": {}, **_NAMES})
    except SyntaxError:
        raise ValueError(f"не удалось разобрать «{text}»: проверьте скобки и знаки * (например, 0.3*t)")
    unknown = [name for name in func.__code__.co_names if name not in _NAMES]
    if unknown:
        raise ValueError(f"в выражении «{text}» неизвестное имя: {', '.join(unknown)}"
                         f" (переменные здесь: {', '.join(names)})")
    func.text = text
    return func
