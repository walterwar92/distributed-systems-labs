"""Задание Б: u_t = a^2 u_xx + f(x,t) на полупрямой x > 0 через функцию Грина, анимация на [0, T].

Вариант группы: u_x(0,t) = mu(t) * nu(t)  (II рода, --kind 2, по умолчанию).
Дополнительно: --kind 1 -- условие I рода u(0,t) = mu(t) * nu(t).
Графики: решение u(x, t); функция Грина полупрямой G(x, xi0, t); граничная функция mu*nu.
Запуск без аргументов -- программа спросит данные (Enter = значение в скобках). Пример:

    python task_b.py --a 1 --phi "exp(-(x-3)**2)" --mu "-1" --nu "1+sin(4*t)" --T 3 --xmax 8
"""
import argparse
import sys

from heatlab.halfline import compute_frames
from heatlab.plots import draw_b, save_gif, ylimits
from heatlab.userfunc import make_func

FIELDS = [
    ("a", "коэффициент a (u_t = a^2 u_xx)", "1"),
    ("phi", "начальная температура phi(x), x >= 0", "exp(-(x-3)**2)"),
    ("mu", "mu(t)", "-1"),
    ("nu", "nu(t)", "1 + sin(4*t)"),
    ("kind", "род условия: 2 -- u_x(0,t) = mu*nu (вариант), 1 -- u(0,t) = mu*nu", "2"),
    ("f", "источник f(x,t)", "0"),
    ("T", "длительность T", "3"),
    ("xmax", "правая граница графика", "8"),
    ("xi0", "точка xi0 источника для графика функции Грина", "2"),
]


def prepare(d, frames=81):
    a, T, kind = float(d["a"]), float(d["T"]), int(d["kind"])
    if T <= 0:
        raise ValueError("длительность T должна быть положительной")
    xmax, xi0 = float(d["xmax"]), float(d["xi0"])
    if xmax <= 0 or xi0 < 0:
        raise ValueError("нужно xmax > 0 и ξ0 ≥ 0")
    phi, f = make_func(d["phi"], "x"), make_func(d["f"], "x t")
    mu, nu = make_func(d["mu"], "t"), make_func(d["nu"], "t")
    times, x, U, G, B = compute_frames(a, phi, mu, nu, kind, f, T, xmax, xi0, frames)
    left = "u_x(0,t)" if kind == 2 else "u(0,t)"
    caption = (f"u_t = {a:g}²·u_xx + f, x > 0;  φ = {d['phi']};  f = {d['f']};  "
               f"{left} = ({d['mu']})·({d['nu']})")
    return dict(times=times, x=x, U=U, G=G, B=B, ylim=ylimits(U), caption=caption,
                xi0=xi0, kind=kind, blabel=f"{left} = μ·ν", T=T)


def read_input():
    if len(sys.argv) == 1:
        print("Уравнение теплопроводности на полупрямой x > 0, функция Грина.")
        print("Функции: sin, cos, exp, sqrt, abs, theta (Хевисайд), maximum, pi; степень ** или ^.")
        return {name: input(f"{question} [{default}]: ").strip() or default
                for name, question, default in FIELDS}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name, question, default in FIELDS:
        parser.add_argument(f"--{name}", default=default, help=f"{question} (по умолчанию {default})")
    parser.add_argument("--frames", default="81", help="число кадров")
    parser.add_argument("--out", default="task_b.gif", help="файл для анимации")
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
    save_gif(fig, data, draw_b, d.get("out", "task_b.gif"))
    print(f"анимация сохранена: {d.get('out', 'task_b.gif')}")
    if not d.get("no_show"):
        anim = FuncAnimation(fig, lambda k: draw_b(fig, data, k), frames=len(data["times"]),
                             interval=60, repeat=True)
        plt.show()
        del anim


if __name__ == "__main__":
    main()
