"""Оконное приложение: ввод данных и просмотр решений заданий А и Б.

Запуск:  python app.py
Вкладка А -- полупрямая (метод характеристик), вкладка Б -- задача Коши (функция Грина).
Слева вводятся данные, кнопка «Построить» (или Enter) считает решение;
ползунок под графиком меняет момент времени, «▶» запускает анимацию.
"""
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.animation import PillowWriter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from task_b import compute_frames
from wavelab.halfline import HalfLine
from wavelab.numint import linspace
from wavelab.userfunc import make_func

HINT = ("Функции: sin, cos, exp, sqrt, abs, maximum, minimum,\n"
        "theta (Хевисайд), pi; степень ** или ^.\n"
        "Примеры: exp(-(x-5)**2), sin(2*t), maximum(1-abs(x),0)")
FRAMES = 121            # число кадров анимации


class Panel(ttk.Frame):
    """Общая часть вкладок: поля ввода слева, график справа, ползунок времени снизу."""
    TITLE = ""
    FIELDS = []         # (ключ, подпись, значение по умолчанию)

    def __init__(self, master):
        super().__init__(master)
        self.T = 1.0
        self.t = 0.0
        self.ready = False
        self.playing = False
        self.moving = False

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        ttk.Label(left, text=self.TITLE, justify="left", font=("Segoe UI", 10, "bold")) \
            .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.vars = {}
        for row, (key, label, default) in enumerate(self.FIELDS, start=1):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 6))
            self.vars[key] = tk.StringVar(value=default)
            entry = ttk.Entry(left, textvariable=self.vars[key], width=28)
            entry.grid(row=row, column=1, sticky="we", pady=2)
            entry.bind("<Return>", lambda _event: self.build())
        row = len(self.FIELDS) + 1
        ttk.Button(left, text="Построить", command=self.build) \
            .grid(row=row, column=0, columnspan=2, sticky="we", pady=(10, 2))
        ttk.Button(left, text=self.SAVE_TEXT, command=self.save) \
            .grid(row=row + 1, column=0, columnspan=2, sticky="we", pady=2)
        self.status = ttk.Label(left, text="", wraplength=340, justify="left")
        self.status.grid(row=row + 2, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(left, text=HINT, justify="left", foreground="#555555") \
            .grid(row=row + 3, column=0, columnspan=2, sticky="w")

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(8, 5.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right, pack_toolbar=True)

        bottom = ttk.Frame(right, padding=(10, 4))
        bottom.pack(fill="x")
        self.play_button = ttk.Button(bottom, text="▶", width=4, command=self.toggle_play)
        self.play_button.pack(side="left")
        self.scale = ttk.Scale(bottom, from_=0.0, to=1.0, command=self.on_slide)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.time_label = ttk.Label(bottom, text="t = 0", width=12)
        self.time_label.pack(side="left")

    # --- действия пользователя -------------------------------------------
    def build(self):
        """Прочитать поля, посчитать решение и показать его."""
        try:
            self.status.config(text="Считаю…", foreground="#555555")
            self.update_idletasks()
            start = self.prepare({key: var.get().strip() for key, var in self.vars.items()})
            self.ready = True
            self.show(start)
            self.status.config(text="Готово.", foreground="#1b7f3b")
        except Exception as error:                      # ошибка ввода -- показать, а не падать
            self.ready = False
            self.status.config(text=f"Ошибка: {error}", foreground="#b00020")

    def on_slide(self, value):
        if self.ready and not self.moving:
            self.show(float(value) * self.T, move_scale=False)

    def toggle_play(self):
        self.playing = not self.playing and self.ready
        self.play_button.config(text="⏸" if self.playing else "▶")
        if self.playing:
            self.tick()

    def tick(self):
        if not self.playing:
            return
        t = self.t + self.T / (FRAMES - 1)
        self.show(0.0 if t > self.T + 1e-9 else t)
        self.after(40, self.tick)

    def show(self, t, move_scale=True):
        self.t = min(max(t, 0.0), self.T)
        self.draw(self.t)
        self.canvas.draw_idle()
        self.time_label.config(text=f"t = {self.t:.2f}")
        if move_scale:
            self.moving = True                          # чтобы set() не вызвал on_slide повторно
            self.scale.set(self.t / self.T)
            self.moving = False

    def save(self):
        if not self.ready:
            self.status.config(text="Сначала нажмите «Построить».", foreground="#b00020")
            return
        self.save_file()

    # --- определяется во вкладках ----------------------------------------
    def prepare(self, d):
        """Разобрать данные d, вернуть начальный момент времени."""
        raise NotImplementedError

    def draw(self, t):
        raise NotImplementedError


class TabA(Panel):
    TITLE = ("Задание А. Полупрямая, метод характеристик\n"
             "u_tt = a²·u_xx,  x > 0;   u(x,0) = φ(x),  u_t(x,0) = ψ(x)\n"
             "I рода:  u(0,t) = μ(t) + g(t)")
    FIELDS = [
        ("a", "скорость a", "1"),
        ("phi", "φ(x)", "exp(-(x-5)**2)"),
        ("psi", "ψ(x)", "0"),
        ("mu", "μ(t)", "sin(2*t)"),
        ("g", "g(t)", "0.3*t"),
        ("kind", "род условия (1, 2, 3)", "1"),
        ("h", "h (для III рода)", "0.5"),
        ("t", "момент времени t", "7"),
        ("T", "предел ползунка по t", "15"),
        ("xmax", "правая граница x", "15"),
    ]
    SAVE_TEXT = "Сохранить график (PNG)"

    def prepare(self, d):
        self.a, self.h, self.kind = float(d["a"]), float(d["h"]), int(d["kind"])
        self.phi, self.psi = make_func(d["phi"], "x"), make_func(d["psi"], "x")
        mu, g = make_func(d["mu"], "t"), make_func(d["g"], "t")
        self.boundary = lambda time: mu(time) + g(time)          # mu(t) + g(t)
        t = float(d["t"])
        self.T = max(float(d["T"]), t, 1e-9)
        xmax = float(d["xmax"])
        self.x = linspace(0.0, xmax, 1001)
        self.start = [self.phi(v) for v in self.x]
        # таблицы бегущих волн строятся один раз -- на все моменты времени до T
        self.solver = HalfLine(self.a, self.phi, self.psi, self.kind, self.boundary, self.h,
                               zmax=xmax + self.a * self.T)
        left = {1: "u(0,t)", 2: "u_x(0,t)", 3: f"u_x(0,t) − {self.h:g}·u(0,t)"}[self.kind]
        self.caption = (f"a = {self.a:g};  φ = {d['phi']};  ψ = {d['psi']};  "
                        f"{left} = {d['mu']} + {d['g']}")
        # общий масштаб по вертикали, чтобы график не прыгал при движении ползунка
        samples = [self.solve(time) for time in linspace(0.0, self.T, 31)]
        lo, hi = min(min(row) for row in samples), max(max(row) for row in samples)
        pad = 0.08 * (hi - lo) + 1e-9
        self.ylim = (lo - pad, hi + pad)
        return t

    def solve(self, t):
        return [self.solver.u(v, t) for v in self.x]

    def draw(self, t):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self.x, self.start, "--", color="gray", label="t = 0")
        ax.plot(self.x, self.solve(t), color="tab:blue", lw=2, label=f"t = {t:.2f}")
        if self.a * t <= self.x[-1]:
            ax.axvline(self.a * t, color="tab:red", ls=":", label="характеристика x = at")
        ax.set_ylim(*self.ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(self.caption, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        self.fig.tight_layout()

    def save_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="task_a.png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=150)
            self.status.config(text=f"Сохранено: {path}", foreground="#1b7f3b")


class TabB(Panel):
    TITLE = ("Задание Б. Задача Коши на прямой, функция Грина\n"
             "u_tt = a²·u_xx + f(x,t);   u(x,0) = φ(x),  u_t(x,0) = ψ(x)\n"
             "G(x,ξ,t) = 1/(2a) при |x − ξ| ≤ at, иначе 0")
    FIELDS = [
        ("a", "скорость a", "1"),
        ("phi", "φ(x)", "maximum(1-abs(x),0)"),
        ("psi", "ψ(x)", "0"),
        ("f", "f(x,t)", "0"),
        ("T", "длительность T", "6"),
        ("xmin", "левая граница x", "-8"),
        ("xmax", "правая граница x", "8"),
        ("xi", "точка импульса ξ", "0"),
    ]
    SAVE_TEXT = "Сохранить анимацию (GIF)"

    def prepare(self, d):
        self.a, self.xi = float(d["a"]), float(d["xi"])
        phi, psi, f = make_func(d["phi"], "x"), make_func(d["psi"], "x"), make_func(d["f"], "x t")
        self.T = max(float(d["T"]), 1e-9)
        # решение и функция Грина на всех кадрах
        self.times, self.x, self.U, self.G = compute_frames(
            self.a, phi, psi, f, self.T, float(d["xmin"]), float(d["xmax"]), self.xi, FRAMES)
        lo, hi = min(min(row) for row in self.U), max(max(row) for row in self.U)
        pad = 0.08 * (hi - lo) + 1e-9
        self.ylim = (lo - pad, hi + pad)
        self.caption = f"a = {self.a:g};  φ = {d['phi']};  ψ = {d['psi']};  f = {d['f']}"
        return 0.0

    def draw(self, t):
        k = int(round(t / self.T * (FRAMES - 1)))
        self.fig.clear()
        ax_u, ax_g = self.fig.subplots(2, 1, sharex=True)
        ax_u.plot(self.x, self.U[0], "--", color="gray", lw=1, label="t = 0")
        ax_u.plot(self.x, self.U[k], color="tab:blue", lw=2, label=f"t = {self.times[k]:.2f}")
        ax_u.set_ylim(*self.ylim)
        ax_u.set_ylabel("u(x, t)")
        ax_u.set_title(self.caption, fontsize=9)
        ax_u.legend(loc="upper right")
        ax_g.plot(self.x, self.G[k], color="tab:red", lw=2)
        ax_g.axvline(self.xi, color="gray", ls=":", lw=1)
        ax_g.set_ylim(-0.05 / self.a, 0.65 / self.a)
        ax_g.set_ylabel(f"G(x, ξ = {self.xi:g}, t)")
        ax_g.set_xlabel("x")
        for ax in (ax_u, ax_g):
            ax.grid(alpha=0.3)
        self.fig.tight_layout()

    def save_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".gif", initialfile="task_b.gif",
                                            filetypes=[("GIF", "*.gif")])
        if not path:
            return
        self.status.config(text="Сохраняю анимацию…", foreground="#555555")
        self.update_idletasks()
        writer = PillowWriter(fps=20)
        with writer.saving(self.fig, path, dpi=100):
            for t in self.times:
                self.draw(t)
                writer.grab_frame()
        self.show(self.t)
        self.status.config(text=f"Сохранено: {path}", foreground="#1b7f3b")


def main():
    root = tk.Tk()
    root.title("Волновое уравнение: метод характеристик и функция Грина")
    root.geometry("1280x760")
    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True)
    panels = [TabA(tabs), TabB(tabs)]
    tabs.add(panels[0], text="  А — полупрямая  ")
    tabs.add(panels[1], text="  Б — функция Грина  ")
    for panel in panels:
        panel.build()
    root.mainloop()


if __name__ == "__main__":
    main()
