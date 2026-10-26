"""Оконное приложение: уравнение теплопроводности, задания А и Б.

Запуск:  python app.py
Вкладка А -- прямая (формула Пуассона или функция Грина), вкладка Б -- полупрямая
(функция Грина, вариант u_x(0,t) = mu(t)*nu(t)).
Слева -- готовые примеры и поля ввода, «Построить» (или Enter) считает все кадры;
ползунок под графиком выбирает момент времени, «▶» запускает анимацию.
"""
import threading
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import task_a
import task_b
from heatlab.plots import draw_a, draw_b, save_gif

HINT = ("Функции: sin, cos, exp, sqrt, log, abs, maximum, minimum,\n"
        "theta (Хевисайд), sign, pi; степень ** или ^.\n"
        "φ — от x;  f — от x и t;  μ, ν — от t.")


class Panel(ttk.Frame):
    """Общая часть вкладок: примеры и поля слева, график справа, ползунок времени снизу."""
    TITLE = ""
    FIELDS = []         # (ключ, подпись, значение по умолчанию, варианты для списка или None)
    PRESETS = {}        # название -> {ключ: значение}
    TASK = None         # модуль с prepare()
    DRAW = None

    def __init__(self, master):
        super().__init__(master)
        self.data = None
        self.k = 0
        self.playing = False
        self.moving = False

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        ttk.Label(left, text=self.TITLE, justify="left", font=("TkDefaultFont", 10, "bold")) \
            .grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(left, text="Пример").grid(row=1, column=0, sticky="w", padx=(0, 6))
        self.preset = ttk.Combobox(left, values=list(self.PRESETS), state="readonly", width=34)
        self.preset.grid(row=1, column=1, sticky="we", pady=(0, 6))
        self.preset.bind("<<ComboboxSelected>>", self.apply_preset)

        self.vars = {}
        for row, (key, label, default, choices) in enumerate(self.FIELDS, start=2):
            ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 6))
            self.vars[key] = tk.StringVar(value=default)
            if choices:
                widget = ttk.Combobox(left, textvariable=self.vars[key], values=choices,
                                      state="readonly", width=34)
            else:
                widget = ttk.Entry(left, textvariable=self.vars[key], width=36)
                widget.bind("<Return>", lambda _event: self.build())
            widget.grid(row=row, column=1, sticky="we", pady=2)
        row = len(self.FIELDS) + 2
        ttk.Label(left, text="кадров").grid(row=row, column=0, sticky="w", pady=2)
        self.frames = tk.StringVar(value="81")
        ttk.Spinbox(left, from_=11, to=301, increment=10, textvariable=self.frames, width=8) \
            .grid(row=row, column=1, sticky="w", pady=2)
        ttk.Label(left, text="скорость, мс/кадр").grid(row=row + 1, column=0, sticky="w", pady=2)
        self.delay = tk.IntVar(value=60)
        ttk.Scale(left, from_=10, to=300, variable=self.delay, orient="horizontal") \
            .grid(row=row + 1, column=1, sticky="we", pady=2)

        self.build_button = ttk.Button(left, text="Построить", command=self.build)
        self.build_button.grid(row=row + 2, column=0, columnspan=2, sticky="we", pady=(10, 2))
        ttk.Button(left, text="Сохранить кадр (PNG)", command=self.save_png) \
            .grid(row=row + 3, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Button(left, text="Сохранить анимацию (GIF)", command=self.save_anim) \
            .grid(row=row + 4, column=0, columnspan=2, sticky="we", pady=2)
        self.status = ttk.Label(left, text="", wraplength=360, justify="left")
        self.status.grid(row=row + 5, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(left, text=HINT, justify="left", foreground="#555555") \
            .grid(row=row + 6, column=0, columnspan=2, sticky="w")

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(8.4, 6.2), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right, pack_toolbar=True)

        bottom = ttk.Frame(right, padding=(10, 4))
        bottom.pack(fill="x")
        self.play_button = ttk.Button(bottom, text="▶", width=4, command=self.toggle_play)
        self.play_button.pack(side="left")
        self.scale = ttk.Scale(bottom, from_=0.0, to=1.0, command=self.on_slide)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.time_label = ttk.Label(bottom, text="t = 0", width=14)
        self.time_label.pack(side="left")

    # --- ввод ---------------------------------------------------------------
    def apply_preset(self, _event=None):
        for key, value in self.PRESETS[self.preset.get()].items():
            self.vars[key].set(value)
        self.build()

    def read(self):
        d = {key: var.get().strip() for key, var in self.vars.items()}
        for key, _label, _default, choices in self.FIELDS:
            if choices:                                 # "poisson — формула Пуассона" -> "poisson"
                d[key] = d[key].split()[0]
        return d

    # --- счёт в отдельном потоке, чтобы окно не замирало -----------------------
    def build(self):
        if self.build_button.instate(["disabled"]):
            return
        self.stop()
        try:
            d, frames = self.read(), int(self.frames.get())
        except ValueError as error:
            self.set_status(f"Ошибка: {error}", "#b00020")
            return
        self.build_button.state(["disabled"])
        self.set_status("Считаю…", "#555555")
        result = {}

        def work():
            try:
                result["data"] = self.TASK.prepare(d, max(frames, 2))
            except Exception as error:                  # ошибка ввода -- показать, а не падать
                result["error"] = error

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self.wait(thread, result)

    def wait(self, thread, result):
        if thread.is_alive():
            self.after(100, self.wait, thread, result)
            return
        self.build_button.state(["!disabled"])
        if "error" in result:
            self.set_status(f"Ошибка: {result['error']}", "#b00020")
            return
        self.data = result["data"]
        self.show(0)
        self.set_status(f"Готово: {len(self.data['times'])} кадров, {len(self.data['x'])} точек по x.",
                        "#1b7f3b")

    def set_status(self, text, color):
        self.status.config(text=text, foreground=color)

    # --- просмотр -------------------------------------------------------------
    def show(self, k, move_scale=True):
        n = len(self.data["times"])
        self.k = min(max(k, 0), n - 1)
        self.DRAW(self.fig, self.data, self.k)
        self.canvas.draw_idle()
        self.time_label.config(text=f"t = {self.data['times'][self.k]:.3f}")
        if move_scale:
            self.moving = True                          # чтобы set() не вызвал on_slide повторно
            self.scale.set(self.k / (n - 1))
            self.moving = False

    def on_slide(self, value):
        if self.data and not self.moving:
            self.show(round(float(value) * (len(self.data["times"]) - 1)), move_scale=False)

    def toggle_play(self):
        self.playing = not self.playing and self.data is not None
        self.play_button.config(text="⏸" if self.playing else "▶")
        if self.playing:
            self.tick()

    def stop(self):
        self.playing = False
        self.play_button.config(text="▶")

    def tick(self):
        if not self.playing:
            return
        k = self.k + 1
        self.show(0 if k >= len(self.data["times"]) else k)
        self.after(int(self.delay.get()), self.tick)

    # --- сохранение -----------------------------------------------------------
    def save_png(self):
        if not self.data:
            return self.set_status("Сначала нажмите «Построить».", "#b00020")
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                            initialfile=f"{self.TASK.__name__}.png")
        if path:
            self.fig.savefig(path, dpi=150)
            self.set_status(f"Сохранено: {path}", "#1b7f3b")

    def save_anim(self):
        if not self.data:
            return self.set_status("Сначала нажмите «Построить».", "#b00020")
        path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF", "*.gif")],
                                            initialfile=f"{self.TASK.__name__}.gif")
        if not path:
            return
        self.stop()
        self.set_status("Сохраняю анимацию…", "#555555")
        self.update_idletasks()
        save_gif(self.fig, self.data, self.DRAW, path)
        self.show(self.k)
        self.set_status(f"Сохранено: {path}", "#1b7f3b")


class TabA(Panel):
    TITLE = ("Задание А. Задача Коши на прямой\n"
             "u_t = a²·u_xx + f(x,t),   u(x,0) = φ(x)\n"
             "u = ∫G(x,ξ,t)φ(ξ)dξ + ∫₀ᵗ∫G(x,ξ,t−τ)f(ξ,τ)dξdτ")
    FIELDS = [
        ("method", "способ", "poisson — формула Пуассона",
         ["poisson — формула Пуассона", "green — функция Грина"]),
        ("a", "a (температуропроводность)", "1", None),
        ("phi", "φ(x)", "theta(1-abs(x))", None),
        ("f", "f(x,t)", "0", None),
        ("T", "длительность T", "2", None),
        ("xmin", "левая граница x", "-6", None),
        ("xmax", "правая граница x", "6", None),
        ("xi0", "ξ₀ для графика G", "0", None),
    ]
    PRESETS = {
        "Прямоугольный импульс": dict(phi="theta(1-abs(x))", f="0", a="1", T="2", xmin="-6", xmax="6"),
        "Гауссов пучок (есть точное решение)": dict(phi="exp(-x**2)", f="0", a="1", T="2",
                                                   xmin="-6", xmax="6"),
        "Два пика": dict(phi="exp(-4*(x+2)**2) + 0.6*exp(-4*(x-2)**2)", f="0", a="0.8", T="2",
                         xmin="-6", xmax="6"),
        "Ступенька θ(x)": dict(phi="theta(x)", f="0", a="1", T="2", xmin="-6", xmax="6"),
        "sin(x): затухание e^(−a²t)": dict(phi="sin(x)", f="0", a="1", T="2", xmin="-6.3", xmax="6.3"),
        "Нагрев источником": dict(phi="0", f="exp(-x**2)", a="1", T="2", xmin="-6", xmax="6"),
        "Пульсирующий источник": dict(phi="0", f="3*exp(-4*x**2)*sin(4*t)**2", a="0.7", T="3",
                                      xmin="-5", xmax="5"),
    }
    TASK, DRAW = task_a, staticmethod(draw_a)


class TabB(Panel):
    TITLE = ("Задание Б. Полупрямая x > 0, функция Грина\n"
             "u_t = a²·u_xx + f,  u(x,0) = φ(x)\n"
             "вариант:  u_x(0,t) = μ(t)·ν(t)")
    FIELDS = [
        ("kind", "граничное условие", "2 — u_x(0,t) = μ·ν (вариант)",
         ["2 — u_x(0,t) = μ·ν (вариант)", "1 — u(0,t) = μ·ν"]),
        ("a", "a (температуропроводность)", "1", None),
        ("phi", "φ(x)", "exp(-(x-3)**2)", None),
        ("mu", "μ(t)", "-1", None),
        ("nu", "ν(t)", "1 + sin(4*t)", None),
        ("f", "f(x,t)", "0", None),
        ("T", "длительность T", "3", None),
        ("xmax", "правая граница x", "8", None),
        ("xi0", "ξ₀ для графика G", "2", None),
    ]
    PRESETS = {
        "Пульсирующий приток тепла": dict(phi="exp(-(x-3)**2)", mu="-1", nu="1 + sin(4*t)", f="0",
                                          a="1", T="3", xmax="8"),
        "Теплоизоляция (μ·ν = 0)": dict(phi="exp(-(x-1)**2)", mu="0", nu="1", f="0", a="1", T="3",
                                        xmax="8"),
        "Постоянный поток (точное решение)": dict(phi="0", mu="-1", nu="1", f="0", a="1", T="3",
                                                  xmax="6"),
        "Отвод тепла через границу": dict(phi="theta(2-x)", mu="1", nu="exp(-t)", f="0", a="1",
                                          T="3", xmax="8"),
        "μ·ν = t·sin(3t)": dict(phi="0", mu="t", nu="sin(3*t)", f="0", a="0.8", T="4", xmax="6"),
        "Источник внутри + изоляция": dict(phi="0", mu="0", nu="1", f="exp(-(x-2)**2)", a="1",
                                           T="3", xmax="8"),
        "cos(x): затухание e^(−a²t)": dict(phi="cos(x)", mu="0", nu="1", f="0", a="1", T="2",
                                           xmax="9.42"),
    }
    TASK, DRAW = task_b, staticmethod(draw_b)


def main():
    root = tk.Tk()
    root.title("Уравнение теплопроводности: формула Пуассона и функция Грина")
    root.geometry("1360x800")
    tabs = ttk.Notebook(root)
    tabs.pack(fill="both", expand=True)
    panels = [TabA(tabs), TabB(tabs)]
    tabs.add(panels[0], text="  А — прямая  ")
    tabs.add(panels[1], text="  Б — полупрямая  ")
    for panel in panels:
        panel.build()
    root.mainloop()


if __name__ == "__main__":
    main()
