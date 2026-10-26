"""Задание Б: задача Коши на прямой через функцию Грина, анимация на отрезке [0, T].

Верхний график -- решение u(x, t), нижний -- функция Грина G(x, xi, t) как функция x.
Запуск без аргументов -- программа спросит все данные по очереди (Enter = значение в скобках).
Запуск с аргументами, например:

    python task_b.py --a 1 --phi "maximum(1-abs(x),0)" --psi 0 --f 0 --T 6 --xmin -8 --xmax 8
"""
import argparse
import sys

from wavelab.green import green, solve_cauchy
from wavelab.numint import linspace
from wavelab.userfunc import make_func

# имя, вопрос, значение по умолчанию
FIELDS = [
    ("a", "скорость a", "1"),
    ("phi", "начальное отклонение phi(x)", "maximum(1-abs(x),0)"),
    ("psi", "начальная скорость psi(x)", "0"),
    ("f", "внешняя сила f(x,t)", "0"),
    ("T", "длительность T", "6"),
    ("xmin", "левая граница графика", "-8"),
    ("xmax", "правая граница графика", "8"),
    ("xi", "точка xi приложения импульса для функции Грина", "0"),
]


def read_input():
    if len(sys.argv) == 1:                     # диалог
        print("Задача Коши для u_tt = a^2 u_xx + f(x,t) на прямой.")
        print("Функции: sin, cos, exp, sqrt, abs, theta (Хевисайд), maximum, pi; степень ** или ^.")
        return {name: input(f"{question} [{default}]: ").strip() or default
                for name, question, default in FIELDS}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name, question, default in FIELDS:
        parser.add_argument(f"--{name}", default=default, help=f"{question} (по умолчанию {default})")
    parser.add_argument("--frames", default="121", help="число кадров")
    parser.add_argument("--out", default="task_b.gif", help="файл для анимации")
    parser.add_argument("--no-show", action="store_true", help="не открывать окно с анимацией")
    return vars(parser.parse_args())


def nonzero(func):
    """None, если пользователь ввёл тождественный нуль: такое слагаемое можно не считать."""
    return None if func.text.strip() in ("0", "0.0") else func


def compute_frames(a, phi, psi, f, T, xmin, xmax, xi0, frames):
    """Решение и функция Грина на всех кадрах: times, x, U[k][i], G[k][i]."""
    psi, f = nonzero(psi), nonzero(f)
    # с внешней силой считается двойной интеграл в каждой точке, поэтому сетка грубее -- иначе долго
    points, n, nt = (401, 81, 3) if f is None else (201, 31, 31)
    x = linspace(xmin, xmax, points)
    times = linspace(0.0, T, frames)
    U = [solve_cauchy(x, t, a, phi, psi, f, n=n, nt=nt) for t in times]
    G = [[green(v, xi0, t, a) for v in x] for t in times]
    return times, x, U, G


def animate(a, phi, psi, f, T, xmin, xmax, xi0, frames=121, out="task_b.gif", show=True):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    times, x, U, G = compute_frames(a, phi, psi, f, T, xmin, xmax, xi0, frames)
    lo, hi = min(min(row) for row in U), max(max(row) for row in U)
    pad = 0.08 * (hi - lo) + 1e-9

    fig, (ax_u, ax_g) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax_u.plot(x, U[0], "--", color="gray", lw=1, label="t = 0")
    line_u, = ax_u.plot(x, U[0], color="tab:blue", lw=2, label="u(x, t)")
    ax_u.set_ylim(lo - pad, hi + pad)
    ax_u.set_ylabel("u(x, t)")
    ax_u.grid(alpha=0.3)
    ax_u.legend(loc="upper right")

    line_g, = ax_g.plot(x, G[0], color="tab:red", lw=2)
    ax_g.axvline(xi0, color="gray", ls=":", lw=1)
    ax_g.set_ylim(-0.05 / a, 0.65 / a)
    ax_g.set_xlabel("x")
    ax_g.set_ylabel(f"G(x, ξ = {xi0:g}, t)")
    ax_g.grid(alpha=0.3)

    def draw(k):
        line_u.set_ydata(U[k])
        line_g.set_ydata(G[k])
        ax_u.set_title(f"a = {a:g};  φ = {phi.text};  ψ = {psi.text};  f = {f.text};   t = {times[k]:.2f}",
                       fontsize=10)
        return line_u, line_g

    draw(0)
    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=frames, interval=50)
    anim.save(out, writer=PillowWriter(fps=20))
    print(f"Анимация сохранена в {out}")
    if show:
        plt.show()
    plt.close(fig)
    return times, x, U, G


def main():
    d = read_input()
    animate(
        a=float(d["a"]), phi=make_func(d["phi"], "x"), psi=make_func(d["psi"], "x"),
        f=make_func(d["f"], "x t"), T=float(d["T"]),
        xmin=float(d["xmin"]), xmax=float(d["xmax"]), xi0=float(d["xi"]),
        frames=int(d.get("frames", 121)), out=d.get("out", "task_b.gif"),
        show=not d.get("no_show", False),
    )


if __name__ == "__main__":
    main()
