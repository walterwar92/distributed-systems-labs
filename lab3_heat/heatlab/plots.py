"""Рисование кадров заданий А и Б (общее для окна, скриптов и рисунков отчёта).

Данные кадров -- словарь из compute_*: times, x, U, G (и B для задания Б) плюс подписи.
Только рисование: matplotlib получает готовые списки чисел.
"""
from .numint import trapezoid


def ylimits(rows, pad=0.08):
    lo = min(min(r) for r in rows)
    hi = max(max(r) for r in rows)
    d = pad * (hi - lo) + 1e-9
    return lo - d, hi + d


def draw_a(fig, data, k):
    """Задание А: сверху u(x, t_k) и u(x, 0), снизу G(x, xi0, t_k)."""
    fig.clear()
    times, x, U, G = data["times"], data["x"], data["U"], data["G"]
    ax_u, ax_g = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax_u.plot(x, U[0], "--", color="gray", lw=1, label="t = 0")
    ax_u.plot(x, U[k], color="tab:red", lw=2, label=f"t = {times[k]:.3f}")
    ax_u.set_ylim(*data["ylim"])
    ax_u.set_ylabel("u(x, t)")
    mass = trapezoid(U[k], x[1] - x[0])
    ax_u.set_title(f"{data['caption']}\n∫u dx на графике = {mass:.4f}", fontsize=9)
    ax_u.legend(loc="upper right")
    ax_g.plot(x, G[k], color="tab:blue", lw=2)
    ax_g.axvline(data["xi0"], color="gray", ls=":", lw=1)
    ax_g.set_ylabel(f"G(x, ξ={data['xi0']:g}, t)")
    ax_g.set_xlabel("x")
    if k == 0:
        ax_g.text(0.5, 0.5, "t = 0: G = δ(x − ξ)", transform=ax_g.transAxes, ha="center")
    for ax in (ax_u, ax_g):
        ax.grid(alpha=0.3)
    fig.tight_layout()


def draw_b(fig, data, k):
    """Задание Б: u(x, t_k); функция Грина полупрямой; граничная функция mu*nu."""
    fig.clear()
    times, x, U, G, B = data["times"], data["x"], data["U"], data["G"], data["B"]
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[3, 2])
    ax_u = fig.add_subplot(grid[0, :])
    ax_g = fig.add_subplot(grid[1, 0])
    ax_b = fig.add_subplot(grid[1, 1])
    ax_u.plot(x, U[0], "--", color="gray", lw=1, label="t = 0")
    ax_u.plot(x, U[k], color="tab:red", lw=2, label=f"t = {times[k]:.3f}")
    ax_u.set_ylim(*data["ylim"])
    ax_u.set_xlim(x[0], x[-1])
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("u(x, t)")
    ax_u.set_title(data["caption"], fontsize=9)
    ax_u.legend(loc="upper right")
    ax_g.plot(x, G[k], color="tab:blue", lw=2)
    ax_g.axvline(data["xi0"], color="gray", ls=":", lw=1)
    ax_g.set_xlim(x[0], x[-1])
    ax_g.set_xlabel("x")
    ax_g.set_ylabel(f"G{data['kind']}(x, ξ={data['xi0']:g}, t)")
    if k == 0:
        ax_g.text(0.5, 0.5, "t = 0: G = δ(x − ξ)", transform=ax_g.transAxes, ha="center")
    ax_b.plot(times, B, color="tab:green", lw=1.5)
    ax_b.plot([times[k]], [B[k]], "o", color="tab:green")
    ax_b.set_xlabel("t")
    ax_b.set_ylabel(data["blabel"])
    for ax in (ax_u, ax_g, ax_b):
        ax.grid(alpha=0.3)
    fig.tight_layout()


def save_gif(fig, data, draw, path, fps=15):
    """Анимация всех кадров в GIF (PillowWriter из matplotlib)."""
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=90):
        for k in range(len(data["times"])):
            draw(fig, data, k)
            writer.grab_frame()
