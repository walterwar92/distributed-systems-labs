"""Задание А: решение волнового уравнения на полупрямой и график u(x, t) в момент t.

Запуск без аргументов -- программа спросит все данные по очереди (Enter = значение в скобках).
Запуск с аргументами, например:

    python task_a.py --a 1 --phi "exp(-(x-5)**2)" --psi 0 --bc 1 --mu "sin(2*t)" --g "0.3*t" --t 7 --xmax 15
"""
import argparse
import sys

from wavelab.halfline import solve_halfline
from wavelab.numint import linspace
from wavelab.userfunc import make_func

# правая часть граничного условия по варианту -- сумма двух функций: mu(t) + g(t)
BC_TEXT = {1: "u(0,t) = RHS", 2: "u_x(0,t) = RHS", 3: "u_x(0,t) − h·u(0,t) = RHS"}

# имя, вопрос, значение по умолчанию
FIELDS = [
    ("a", "скорость a", "1"),
    ("phi", "начальное отклонение phi(x)", "exp(-(x-5)**2)"),
    ("psi", "начальная скорость psi(x)", "0"),
    ("bc", "род граничного условия (1, 2 или 3)", "1"),
    ("mu", "граничное условие: mu(t)", "sin(2*t)"),
    ("g", "граничное условие: g(t)", "0.3*t"),
    ("h", "коэффициент h (только для III рода)", "0.5"),
    ("t", "момент времени t", "7"),
    ("xmax", "правая граница графика", "15"),
]


def read_input():
    if len(sys.argv) == 1:                     # диалог
        print("Волновое уравнение u_tt = a^2 u_xx на полупрямой x > 0.")
        print("Функции: sin, cos, exp, sqrt, abs, theta (Хевисайд), maximum, pi; степень ** или ^.")
        return {name: input(f"{question} [{default}]: ").strip() or default
                for name, question, default in FIELDS}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name, question, default in FIELDS:
        parser.add_argument(f"--{name}", default=default, help=f"{question} (по умолчанию {default})")
    parser.add_argument("--out", default="task_a.png", help="файл для графика")
    parser.add_argument("--no-show", action="store_true", help="не открывать окно с графиком")
    return vars(parser.parse_args())


def plot_solution(a, phi, psi, kind, mu, g, h, t, xmax, out="task_a.png", show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = linspace(0.0, xmax, 1201)

    def boundary(time):                        # mu(t) + g(t)
        return mu(time) + g(time)

    u = solve_halfline(x, t, a, phi, psi, kind, boundary, h)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, [phi(v) for v in x], "--", color="gray", label="t = 0")
    ax.plot(x, u, color="tab:blue", lw=2, label=f"t = {t:g}")
    if a * t <= xmax:
        ax.axvline(a * t, color="tab:red", ls=":", label="характеристика x = at")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    bc = BC_TEXT[kind].replace("h", f"{h:g}").replace("RHS", f"{mu.text} + {g.text}")
    ax.set_title(f"a = {a:g};  φ = {phi.text};  ψ = {psi.text};  {bc}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"График сохранён в {out}")
    if show:
        plt.show()
    plt.close(fig)
    return x, u


def main():
    d = read_input()
    plot_solution(
        a=float(d["a"]), phi=make_func(d["phi"], "x"), psi=make_func(d["psi"], "x"),
        kind=int(d["bc"]), mu=make_func(d["mu"], "t"), g=make_func(d["g"], "t"), h=float(d["h"]),
        t=float(d["t"]), xmax=float(d["xmax"]),
        out=d.get("out", "task_a.png"), show=not d.get("no_show", False),
    )


if __name__ == "__main__":
    main()
