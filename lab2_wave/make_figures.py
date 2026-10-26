"""Рисунки для отчёта (report/figures). Запуск: python make_figures.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavelab.green import green, solve_cauchy
from wavelab.halfline import solve_halfline
from wavelab.numint import linspace
from wavelab.userfunc import make_func

OUT = os.path.join("report", "figures")
os.makedirs(OUT, exist_ok=True)
ZERO = make_func("0")


def figure_a():
    """Задание А: четыре примера, по нескольку моментов времени на каждом."""
    hump = make_func("exp(-(x-5)**2)")
    cases = [
        ("I рода: u(0,t) = sin 2t, φ = ψ = 0", ZERO, 1, make_func("sin(2*t)", "t"), 0.0, (1, 3, 6)),
        ("I рода: u(0,t) = 0 (закреплённый конец)", hump, 1, make_func("0", "t"), 0.0, (0, 3, 5, 8)),
        ("II рода: u_x(0,t) = 0 (свободный конец)", hump, 2, make_func("0", "t"), 0.0, (0, 3, 5, 8)),
        ("III рода: u_x(0,t) − 0.5·u(0,t) = 0", hump, 3, make_func("0", "t"), 0.5, (0, 3, 5, 8)),
    ]
    x = linspace(0.0, 15.0, 1201)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, (title, phi, kind, g, h, times) in zip(axes.flat, cases):
        for t in times:
            ax.plot(x, solve_halfline(x, t, 1.0, phi, ZERO, kind, g, h), label=f"t = {t}")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("u(x, t)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "a_examples.png"), dpi=170)
    plt.close(fig)


def figure_b():
    """Задание Б: кадры анимации -- решение (пример лекции 3) и функция Грина."""
    a = 1.0
    phi = make_func("maximum(1-abs(x),0)")
    x = linspace(-8.0, 8.0, 1601)
    times = (0, 0.5, 1, 3, 6)
    fig, (ax_u, ax_g) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for t in times:
        ax_u.plot(x, solve_cauchy(x, t, a, phi), label=f"t = {t}")
        if t > 0:
            ax_g.plot(x, [green(v, 0.0, t, a) for v in x], label=f"t = {t}")
    ax_u.set_title("Решение: a = 1, φ = max(1 − |x|, 0), ψ = 0, f = 0", fontsize=10)
    ax_u.set_ylabel("u(x, t)")
    ax_g.set_title("Функция Грина G(x, ξ = 0, t)", fontsize=10)
    ax_g.set_ylabel("G")
    ax_g.set_xlabel("x")
    ax_g.set_ylim(-0.05, 0.65)
    for ax in (ax_u, ax_g):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=5, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "b_frames.png"), dpi=170)
    plt.close(fig)


def figure_b_terms():
    """Задание Б: вклад начальной скорости и внешней силы."""
    a = 1.0
    psi = make_func("exp(-x**2)")
    f = make_func("exp(-x**2)*sin(3*t)", "x t")
    x = linspace(-8.0, 8.0, 321)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for t in (0.5, 1, 3, 6):
        axes[0].plot(x, solve_cauchy(x, t, a, ZERO, psi, n=101), label=f"t = {t}")
        axes[1].plot(x, solve_cauchy(x, t, a, ZERO, None, f, n=41, nt=41), label=f"t = {t}")
    axes[0].set_title("φ = 0, ψ = exp(−x²), f = 0", fontsize=10)
    axes[1].set_title("φ = ψ = 0, f = exp(−x²)·sin 3t", fontsize=10)
    axes[0].set_ylabel("u(x, t)")
    for ax in axes:
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "b_terms.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    figure_a()
    figure_b()
    figure_b_terms()
    print("Рисунки сохранены в", OUT)
