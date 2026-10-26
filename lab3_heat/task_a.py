"""Задание А: задача Коши для u_t = a^2 u_xx + f(x,t) на прямой, анимация на [0, T].

Верхний график -- решение u(x, t), нижний -- функция Грина G(x, xi0, t) как функция x.
Способ: --method poisson (формула Пуассона) или --method green (свёртка с функцией Грина).
Запуск без аргументов -- программа спросит данные (Enter = значение в скобках). Пример:

    python task_a.py --a 1 --phi "theta(1-abs(x))" --f 0 --T 2 --xmin -6 --xmax 6 --method poisson
"""
import argparse
import sys

from heatlab.line import compute_frames
from heatlab.plots import draw_a, save_gif, ylimits
from heatlab.userfunc import make_func

# имя, вопрос, значение по умолчанию
FIELDS = [
    ("a", "коэффициент a (u_t = a^2 u_xx)", "1"),
    ("phi", "начальная температура phi(x)", "theta(1-abs(x))"),
    ("f", "источник f(x,t)", "0"),
    ("T", "длительность T", "2"),
    ("xmin", "левая граница графика", "-6"),
    ("xmax", "правая граница графика", "6"),
    ("xi0", "точка xi0 источника для графика функции Грина", "0"),
    ("method", "способ: poisson или green", "poisson"),
]


def prepare(d, frames=81):
    """Посчитать все кадры по словарю строк d (как из полей ввода)."""
    a, T = float(d["a"]), float(d["T"])
    if T <= 0:
        raise ValueError("длительность T должна быть положительной")
    xmin, xmax = float(d["xmin"]), float(d["xmax"])
    if xmax <= xmin:
        raise ValueError("нужно xmin < xmax")
    method = d["method"].strip().lower()
    phi, f = make_func(d["phi"], "x"), make_func(d["f"], "x t")
    times, x, U, G = compute_frames(a, phi, f, T, xmin, xmax, float(d["xi0"]), frames, method)
    name = "формула Пуассона" if method == "poisson" else "функция Грина"
    caption = f"u_t = {a:g}²·u_xx + f;  φ = {d['phi']};  f = {d['f']};  ({name})"
    return dict(times=times, x=x, U=U, G=G, ylim=ylimits(U), caption=caption,
                xi0=float(d["xi0"]), T=T)


def read_input():
    if len(sys.argv) == 1:                     # диалог
        print("Задача Коши для u_t = a^2 u_xx + f(x,t) на прямой.")
        print("Функции: sin, cos, exp, sqrt, abs, theta (Хевисайд), maximum, pi; степень ** или ^.")
        return {name: input(f"{question} [{default}]: ").strip() or default
                for name, question, default in FIELDS}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name, question, default in FIELDS:
        parser.add_argument(f"--{name}", default=default, help=f"{question} (по умолчанию {default})")
    parser.add_argument("--frames", default="81", help="число кадров")
    parser.add_argument("--out", default="task_a.gif", help="файл для анимации")
    parser.add_argument("--no-show", action="store_true", help="не открывать окно с анимацией")
    return vars(parser.parse_args())


def main():
    d = read_input()
    import matplotlib
    if d.get("no_show"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    data = prepare(d, int(d.get("frames", 81)))
    fig = plt.figure(figsize=(9, 6.5))
    save_gif(fig, data, draw_a, d.get("out", "task_a.gif"))
    print(f"анимация сохранена: {d.get('out', 'task_a.gif')}")
    if not d.get("no_show"):
        anim = FuncAnimation(fig, lambda k: draw_a(fig, data, k), frames=len(data["times"]),
                             interval=60, repeat=True)
        plt.show()
        del anim


if __name__ == "__main__":
    main()
