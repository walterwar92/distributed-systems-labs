"""Рисунки для отчёта и объяснения (report/figures) и анимации task_a.gif, task_b.gif.

Запуск:  python make_figures.py          (всё)
         python make_figures.py --no-gif (без анимаций, быстрее)
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import task_a
import task_b
from heatlab.halfline import HeatHalfLine
from heatlab.kernels import green_half, heat_kernel
from heatlab.line import HeatLine
from heatlab.numint import linspace
from heatlab.plots import draw_a, draw_b, save_gif
from heatlab.userfunc import make_func as F

OUT = os.path.join("report", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=160)
    plt.close(fig)
    print("  ", name)


def snapshots(ax, solver, x, times, title):
    colors = plt.cm.plasma(linspace(0.0, 0.85, len(times)))
    for t, c in zip(times, colors):
        ax.plot(x, solver.row(x, t), color=c, lw=1.8, label=f"t = {t:g}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend(fontsize=8)


# --------------------------------------------------------------------------- отчёт
def fig_a_frames():
    x = linspace(-6, 6, 241)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    snapshots(axes[0], HeatLine(1.0, F("theta(1-abs(x))")), x, [0, 0.05, 0.2, 0.5, 1, 2],
              "φ = θ(1 − |x|), a = 1, формула Пуассона")
    snapshots(axes[1], HeatLine(1.0, F("0"), F("3*exp(-4*x**2)*sin(4*t)**2", "x t"), "green"),
              linspace(-5, 5, 121), [0, 0.4, 0.8, 1.6, 3], "φ = 0, f = 3e^(−4x²)·sin²4t, функция Грина")
    save(fig, "a_frames.png")


def fig_a_check():
    a = 0.7
    x = linspace(-4, 4, 161)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for t, c in ((0.2, "tab:blue"), (1.0, "tab:orange"), (3.0, "tab:green")):
        d = 1 + 4 * a * a * t
        ax.plot(x, [math.exp(-v * v / d) / math.sqrt(d) for v in x], color=c, lw=3, alpha=0.35)
        ax.plot(x, HeatLine(a, F("exp(-x**2)")).row(x, t), "--", color=c, label=f"t = {t:g}")
    ax.set_title("φ = e^(−x²): точное решение (толстые) и расчёт (пунктир)", fontsize=10)
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax = axes[1]
    exact = lambda v, t: 0.5 * (1 + math.erf(v / (2 * a * math.sqrt(t))))
    for method, c in (("poisson", "tab:red"), ("green", "tab:blue")):
        s = HeatLine(a, F("theta(x)"), method=method)
        err = [abs(s.u(v, 0.5) - exact(v, 0.5)) + 1e-17 for v in x]
        ax.semilogy(x, err, color=c, label=method)
    ax.set_title("φ = θ(x), t = 0.5: |u − u_точн|", fontsize=10)
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    save(fig, "a_check.png")


def fig_b_frames():
    x = linspace(0, 8, 201)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    snapshots(axes[0], HeatHalfLine(1.0, F("exp(-(x-3)**2)"), F("-1", "t"), F("1 + sin(4*t)", "t")),
              x, [0, 0.2, 0.5, 1, 2, 3], "u_x(0,t) = (−1)·(1 + sin 4t), φ = e^(−(x−3)²)")
    snapshots(axes[1], HeatHalfLine(1.0, F("theta(2-x)"), F("1", "t"), F("exp(-t)", "t")),
              x, [0, 0.1, 0.5, 1, 2, 3], "u_x(0,t) = 1·e^(−t), φ = θ(2 − x)")
    save(fig, "b_frames.png")


def fig_b_check():
    a = 0.7
    x = linspace(0, 4, 161)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    s = HeatHalfLine(a, F("0"), F("-1", "t"), F("1.5", "t"))
    for t, c in ((0.2, "tab:blue"), (1.0, "tab:orange"), (3.0, "tab:green")):
        ex = []
        for v in x:
            z = v / (2 * a * math.sqrt(t))
            ex.append(2 * 1.5 * (a * math.sqrt(t / math.pi) * math.exp(-z * z) - v / 2 * math.erfc(z)))
        ax.plot(x, ex, color=c, lw=3, alpha=0.35)
        ax.plot(x, s.row(x, t), "--", color=c, label=f"t = {t:g}")
    ax.set_title("u_x(0,t) = (−1)·1.5, φ = 0: точное (толстые) и расчёт", fontsize=10)
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    ax = axes[1]
    s = HeatHalfLine(1.0, F("exp(-(x-2)**2)"), F("sin(t)", "t"), F("2", "t"))
    times = linspace(0.05, 4, 80)
    h = 0.01
    ux = [(-3 * s.u(0, t) + 4 * s.u(h, t) - s.u(2 * h, t)) / (2 * h) for t in times]
    ax.plot(times, [2 * math.sin(t) for t in times], lw=3, alpha=0.4, label="μ(t)·ν(t) = sin t · 2")
    ax.plot(times, ux, "--", label="u_x(0, t) по расчёту")
    ax.set_title("Проверка граничного условия", fontsize=10)
    ax.set_xlabel("t")
    ax.legend(fontsize=8)
    save(fig, "b_check.png")


# --------------------------------------------------------------------------- объяснение
def fig_kernel():
    x = linspace(-5, 5, 401)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t in (0.02, 0.1, 0.5, 1, 3):
        axes[0].plot(x, [heat_kernel(v, 0, t, 1) for v in x], label=f"t = {t:g}")
    axes[0].set_title("K(x, 0, t) при a = 1: от δ-функции к пологому колоколу", fontsize=10)
    for a in (0.3, 0.6, 1, 2):
        axes[1].plot(x, [heat_kernel(v, 0, 1, a) for v in x], label=f"a = {a:g}")
    axes[1].set_title("K(x, 0, 1) при разных a: ширина ~ 2a√t", fontsize=10)
    for ax in axes:
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
    save(fig, "e_kernel.png")


def fig_poisson():
    z = linspace(-4, 4, 401)
    phi = F("theta(1-abs(x))")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(z, [math.exp(-v * v) / math.sqrt(math.pi) for v in z], "k", label="e^(−z²)/√π (вес)")
    for x0, c in ((0, "tab:red"), (1, "tab:blue"), (2, "tab:green")):
        r = 2 * math.sqrt(0.3)
        axes[0].plot(z, [phi(x0 + r * v) * math.exp(-v * v) / math.sqrt(math.pi) for v in z], color=c,
                     label=f"подынтегральная при x = {x0}, t = 0.3")
    axes[0].set_xlabel("z")
    axes[0].set_title("Формула Пуассона после замены ξ = x + 2a√t·z", fontsize=10)
    axes[0].legend(fontsize=8)
    x = linspace(-3, 3, 241)
    for t in (0.0, 0.3):
        axes[1].plot(x, HeatLine(1, phi).row(x, t), label=f"u(x, {t:g})")
    axes[1].set_xlabel("x")
    axes[1].set_title("u(x, t) = площадь под цветной кривой слева", fontsize=10)
    axes[1].legend(fontsize=8)
    save(fig, "e_poisson.png")


def fig_a_effect():
    x = linspace(-6, 6, 241)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for a in (0.3, 0.7, 1.0, 1.5):
        axes[0].plot(x, HeatLine(a, F("theta(1-abs(x))")).row(x, 1.0), label=f"a = {a:g}")
    axes[0].set_title("Влияние a: φ = θ(1 − |x|), t = 1", fontsize=10)
    axes[0].set_xlabel("x")
    axes[0].legend(fontsize=8)
    s = HeatLine(1.0, F("exp(-x**2)"), F("exp(-x**2)", "x t"))
    times = linspace(0, 3, 31)
    axes[1].plot(times, [s.u(0, t) for t in times], label="φ = e^(−x²), f = e^(−x²)")
    axes[1].plot(times, [HeatLine(1.0, F("exp(-x**2)")).u(0, t) for t in times], label="φ = e^(−x²), f = 0")
    axes[1].plot(times, [HeatLine(1.0, F("0"), F("exp(-x**2)", "x t")).u(0, t) for t in times],
                 label="φ = 0, f = e^(−x²)")
    axes[1].set_title("u(0, t): начальное условие + источник = сумма (суперпозиция)", fontsize=10)
    axes[1].set_xlabel("t")
    axes[1].legend(fontsize=8)
    save(fig, "e_a_effect.png")


def fig_images():
    x = linspace(-4, 4, 401)
    xi, t = 1.5, 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    k1 = [heat_kernel(v, xi, t, 1) for v in x]
    k2 = [heat_kernel(v, -xi, t, 1) for v in x]
    axes[0].plot(x, k1, label="K(x, ξ, t) — источник в ξ = 1.5")
    axes[0].plot(x, k2, label="K(x, −ξ, t) — отражённый в −ξ")
    axes[0].plot(x, [p + q for p, q in zip(k1, k2)], "k", lw=2, label="G₂ = сумма (II рода)")
    axes[0].axvspan(-4, 0, color="gray", alpha=0.12)
    axes[0].set_title("Метод отражений, II рода: u_x(0) = 0", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].plot(x, k1, label="K(x, ξ, t)")
    axes[1].plot(x, [-q for q in k2], label="−K(x, −ξ, t)")
    axes[1].plot(x, [p - q for p, q in zip(k1, k2)], "k", lw=2, label="G₁ = разность (I рода)")
    axes[1].axvspan(-4, 0, color="gray", alpha=0.12)
    axes[1].set_title("I рода: u(0) = 0", fontsize=10)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("x")
    save(fig, "e_images.png")


def fig_green_half_time():
    x = linspace(0, 6, 301)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t in (0.05, 0.3, 1, 3):
        axes[0].plot(x, [green_half(v, 1.5, t, 1, 2) for v in x], label=f"t = {t:g}")
        axes[1].plot(x, [green_half(v, 1.5, t, 1, 1) for v in x], label=f"t = {t:g}")
    axes[0].set_title("G₂(x, 1.5, t): тепло отражается от изолированной границы", fontsize=10)
    axes[1].set_title("G₁(x, 1.5, t): тепло уходит через границу u = 0", fontsize=10)
    for ax in axes:
        ax.set_xlabel("x")
        ax.legend(fontsize=8)
    save(fig, "e_green_half.png")


def fig_mu_nu():
    x = linspace(0, 6, 151)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cases = [("-1", "1", "μν = −1 (приток)"), ("1", "1", "μν = +1 (отток)"), ("0", "1", "μν = 0 (изоляция)"),
             ("-2", "exp(-t)", "μν = −2e^(−t)"), ("-1", "1 + sin(4*t)", "μν = −(1 + sin 4t)")]
    times = linspace(0, 3, 61)
    for mu, nu, label in cases:
        s = HeatHalfLine(1.0, F("exp(-(x-2)**2)"), F(mu, "t"), F(nu, "t"))
        axes[0].plot(x, s.row(x, 2.0), label=label)
        axes[1].plot(times, [s.u(0.0, t) for t in times], label=label)
    axes[0].set_title("u(x, 2) при разных μ(t)·ν(t), φ = e^(−(x−2)²)", fontsize=10)
    axes[0].set_xlabel("x")
    axes[1].set_title("Температура на границе u(0, t)", fontsize=10)
    axes[1].set_xlabel("t")
    for ax in axes:
        ax.legend(fontsize=7)
    save(fig, "e_mu_nu.png")


def fig_parts():
    x = linspace(0, 6, 151)
    s = HeatHalfLine(1.0, F("exp(-(x-3)**2)"), F("-1", "t"), F("1 + sin(4*t)", "t"))
    t = 1.0
    ini = [s.initial_part(v, t) for v in x]
    bnd = [s.boundary_part(v, t) for v in x]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, ini, label="∫G₂φ dξ — вклад начального условия")
    ax.plot(x, bnd, label="−a²∫G₂(x,0,t−τ)μν dτ — вклад границы")
    ax.plot(x, [p + q for p, q in zip(ini, bnd)], "k", lw=2, label="u(x, 1) — сумма")
    ax.set_title("Слагаемые решения задания Б, t = 1", fontsize=10)
    ax.set_xlabel("x")
    ax.legend(fontsize=8)
    save(fig, "e_parts.png")


def fig_error():
    a, t, x0 = 1.0, 0.5, 0.3
    d = 1 + 4 * a * a * t
    exact = math.exp(-x0 * x0 / d) / math.sqrt(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    hs = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    green = lambda phi, h, x: HeatLine(a, F(phi), method="green", h=h).u(x, t)
    axes[0].loglog(hs, [abs(green("exp(-x**2)", h, x0) - exact) + 1e-17 for h in hs], "o-",
                   label="гладкая φ = e^(−x²)")
    ex_step = 0.5 * (1 + math.erf((0.36 - 0.0137) / (2 * a * math.sqrt(t))))
    axes[0].loglog(hs, [abs(green("theta(x-0.0137)", h, 0.36) - ex_step) + 1e-17 for h in hs], "s-",
                   label="разрыв θ(x − 0.0137)")
    k0 = heat_kernel(0.36, 0.0137, t, a)
    axes[0].loglog(hs, [k0 * h / 2 for h in hs], "k:", label="оценка K·h/2")
    axes[0].set_xlabel("шаг сетки h")
    axes[0].set_title("Метод Грина (средние прямоугольники): ошибка от h", fontsize=10)
    axes[0].legend(fontsize=8)
    tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    axes[1].loglog(tols, [abs(HeatLine(a, F("theta(x-0.0137)"), tol=tol).u(0.36, t) - ex_step) + 1e-17
                          for tol in tols], "o-", label="θ(x − 0.0137)")
    axes[1].loglog(tols, tols, "k:", label="ошибка = допуск")
    axes[1].set_xlabel("допуск tol")
    axes[1].set_title("Формула Пуассона (адаптивный Симпсон)", fontsize=10)
    axes[1].legend(fontsize=8)
    save(fig, "e_error.png")


def gifs():
    fig = plt.figure(figsize=(9, 6.5))
    data = task_a.prepare({name: default for name, _q, default in task_a.FIELDS}, 61)
    save_gif(fig, data, draw_a, "task_a.gif")
    draw_a(fig, data, 15)
    save(fig, "a_app_frame.png")
    fig = plt.figure(figsize=(9, 6.5))
    data = task_b.prepare({name: default for name, _q, default in task_b.FIELDS}, 61)
    save_gif(fig, data, draw_b, "task_b.gif")
    draw_b(fig, data, 25)
    save(fig, "b_app_frame.png")
    print("   task_a.gif, task_b.gif")


if __name__ == "__main__":
    for job in (fig_a_frames, fig_a_check, fig_b_frames, fig_b_check, fig_kernel, fig_poisson,
                fig_a_effect, fig_images, fig_green_half_time, fig_mu_nu, fig_parts, fig_error):
        job()
    if "--no-gif" not in sys.argv:
        gifs()
