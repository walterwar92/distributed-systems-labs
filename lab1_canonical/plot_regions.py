# -*- coding: utf-8 -*-
"""Задание C: карты областей, в которых ДУЧП сохраняет свой тип.

Запуск:  python plot_regions.py
Создаёт regions_tricomi.png и regions_gallery.png.
"""
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sympy import lambdify, simplify, symbols, sympify

from pde_canonical.regions import analyse_regions, region_map

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

x, y = symbols("x y", real=True)

# Палитра проверена scripts/validate_palette.js (режим light, фон #FCFCFB):
# все пять проверок PASS — полоса светлоты, цветность, различимость при
# протанопии/дейтеранопии, порог обычного зрения и контраст к фону.
SURFACE = "#FCFCFB"
INK = "#1C1C1A"
INK_SOFT = "#6B6B66"
GRID = "#E4E4DF"

TYPE_COLOR = {
    "hyperbolic": "#3B5BDB",
    "parabolic": "#B5700C",
    "elliptic": "#0D9488",
}
TYPE_FILL = {  # светлые заливки тех же тонов: цвет — для подписей и границ
    "hyperbolic": "#DDE3F8",
    "parabolic": "#F5E7D2",
    "elliptic": "#D6EDEA",
}
TYPE_RU = {
    "hyperbolic": "гиперболический",
    "parabolic": "параболический",
    "elliptic": "эллиптический",
}
SIGN_TO_TYPE = {1: "hyperbolic", 0: "parabolic", -1: "elliptic"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "text.color": INK,
    }
)


def _draw(ax, a11, a12, a22, xlim, ylim, n=221):
    """Заливает области по знаку D и проводит границу D = 0."""
    xs, ys, grid = region_map(a11, a12, a22, xlim=xlim, ylim=ylim, n=n, x=x, y=y)
    fx = [float(v) for v in xs]
    fy = [float(v) for v in ys]

    # Заливка: цвет клетки определяется знаком D в её центре.
    colors = [[TYPE_FILL[SIGN_TO_TYPE[cell]] for cell in row] for row in grid]
    rgb = [[matplotlib.colors.to_rgb(c) for c in row] for row in colors]
    ax.imshow(
        rgb,
        origin="lower",
        extent=(fx[0], fx[-1], fy[0], fy[-1]),
        aspect="auto",
        interpolation="nearest",
    )

    # Граница D = 0 — та самая кривая, на которой уравнение параболическое.
    disc = simplify(sympify(a12) ** 2 - sympify(a11) * sympify(a22))
    try:
        fast = lambdify((x, y), disc, "math")
        values = [[fast(xv, yv) for xv in fx] for yv in fy]
        ax.contour(
            fx, fy, values, levels=[0], colors=[TYPE_COLOR["parabolic"]], linewidths=1.8
        )
    except Exception:  # noqa: BLE001 — граница не обязательна для читаемости
        pass

    ax.set_xlim(fx[0], fx[-1])
    ax.set_ylim(fy[0], fy[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0, labelpad=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)
    return xs, ys, grid


def _label(ax, text, xy, kind):
    ax.annotate(
        text,
        xy=xy,
        ha="center",
        va="center",
        color=TYPE_COLOR[kind],
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", fc=SURFACE, ec=TYPE_COLOR[kind], lw=0.9),
    )


def _pretty(expr):
    """sympy печатает D как 'x**2 - 1'; в подписи это должно выглядеть как 'x² − 1'."""
    text = str(expr)
    for src, dst in (("**2", "²"), ("**3", "³"), ("*", "·"), ("-", "−")):
        text = text.replace(src, dst)
    return text


def _components(grid):
    """Связные области одного знака — обход в ширину по сетке."""
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    found = []
    for j0 in range(rows):
        for i0 in range(cols):
            if seen[j0][i0]:
                continue
            sign = grid[j0][i0]
            queue, cells = [(i0, j0)], []
            seen[j0][i0] = True
            while queue:
                i, j = queue.pop()
                cells.append((i, j))
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < cols and 0 <= nj < rows and not seen[nj][ni]:
                        if grid[nj][ni] == sign:
                            seen[nj][ni] = True
                            queue.append((ni, nj))
            found.append((sign, cells))
    return found


def _auto_labels(ax, xs, ys, grid, min_share=0.04):
    """Подписывает каждую достаточно крупную область — цвет не единственный признак."""
    fx = [float(v) for v in xs]
    fy = [float(v) for v in ys]
    total = len(grid) * len(grid[0])
    short = {"hyperbolic": "гиперб.", "parabolic": "параб.", "elliptic": "эллипт."}

    for sign, cells in _components(grid):
        if len(cells) < min_share * total:
            continue
        kind = SIGN_TO_TYPE[sign]
        ci = sum(c[0] for c in cells) / len(cells)
        cj = sum(c[1] for c in cells) / len(cells)
        i, j = min(cells, key=lambda c: (c[0] - ci) ** 2 + (c[1] - cj) ** 2)
        ax.annotate(
            short[kind],
            xy=(fx[i], fy[j]),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=TYPE_COLOR[kind],
            bbox=dict(
                boxstyle="round,pad=0.22", fc=SURFACE, ec=TYPE_COLOR[kind], lw=0.8
            ),
        )


def _legend(fig, kinds, ncol=3):
    handles = [
        mpatches.Patch(
            facecolor=TYPE_FILL[k], edgecolor=TYPE_COLOR[k], linewidth=1.2,
            label=TYPE_RU[k],
        )
        for k in kinds
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
    )


def tricomi_figure(path="regions_tricomi.png"):
    """Уравнение Трикоми с наложенными характеристиками."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    _draw(ax, y, 0, 1, xlim=(-3, 3), ylim=(-2.2, 2.2))

    # Характеристики при y < 0:  x ± (2/3)(-y)^{3/2} = C.
    steps = [i / 20.0 for i in range(0, 45)]
    for c in [-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4]:
        for sign in (1, -1):
            xs_c, ys_c = [], []
            for t in steps:
                yv = -t
                xv = c - sign * (2.0 / 3.0) * (t**1.5)
                if -3 <= xv <= 3:
                    xs_c.append(xv)
                    ys_c.append(yv)
            if len(xs_c) > 1:
                ax.plot(
                    xs_c, ys_c, color=TYPE_COLOR["hyperbolic"], lw=0.7, alpha=0.5
                )

    ax.axhline(0, color=TYPE_COLOR["parabolic"], lw=1.8)
    _label(ax, "y < 0\nгиперболический", (-1.55, -1.45), "hyperbolic")
    _label(ax, "y > 0\nэллиптический", (-1.55, 1.35), "elliptic")
    ax.annotate(
        "y = 0 — параболический",
        xy=(1.45, 0), xytext=(1.45, 0.42),
        ha="center", color=TYPE_COLOR["parabolic"], fontsize=9, fontweight="bold",
    )

    ax.set_title(
        "Уравнение Трикоми  y·u$_{xx}$ + u$_{yy}$ = 0        D(x, y) = −y",
        fontsize=11.5, color=INK, pad=12, loc="left",
    )
    fig.text(
        0.01, 0.005,
        "тонкие линии — характеристики  x ± (2/3)(−y)^{3/2} = C  в гиперболической области",
        fontsize=8, color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def gallery_figure(path="regions_gallery.png"):
    cases = [
        ("y·u$_{xx}$ + u$_{yy}$ = 0", y, 0, 1, (-3, 3), (-2, 2)),
        ("u$_{xx}$ + x·u$_{yy}$ = 0", 1, 0, x, (-3, 3), (-2, 2)),
        ("u$_{xx}$ + xy·u$_{yy}$ = 0", 1, 0, x * y, (-2.5, 2.5), (-2.5, 2.5)),
        ("(1 − x²)·u$_{xx}$ + u$_{yy}$ = 0", 1 - x**2, 0, 1, (-2.6, 2.6), (-2, 2)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
    present = []
    for ax, (title, a11, a12, a22, xlim, ylim) in zip(axes.ravel(), cases):
        xs, ys, grid = _draw(ax, a11, a12, a22, xlim, ylim, n=181)
        disc = simplify(sympify(a12) ** 2 - sympify(a11) * sympify(a22))
        ax.set_title(
            f"{title}\nD = {_pretty(disc)}", fontsize=10, color=INK, pad=8, loc="left"
        )
        _auto_labels(ax, xs, ys, grid)
        for row in grid:
            for cell in row:
                kind = SIGN_TO_TYPE[cell]
                if kind not in present:
                    present.append(kind)

    order = [k for k in ("hyperbolic", "parabolic", "elliptic") if k in present]
    _legend(fig, order)
    fig.suptitle(
        "Где уравнение сохраняет свой тип: знак дискриминанта D = a₁₂² − a₁₁a₂₂",
        fontsize=12, color=INK, x=0.012, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    for path in (tricomi_figure(), gallery_figure()):
        print(f"сохранено: {path}")

    print("\nПроверка по областям (уравнение Трикоми):")
    for region in analyse_regions(y, 0, 1).regions:
        print(f"  {region.description:>10}  ->  {region.type_ru}")


if __name__ == "__main__":
    main()
