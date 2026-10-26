# 08. Инструмент `tools/pde.py` — справка с проверенными примерами

Помощник на sympy для проверки выкладок.
- `tools/pde.py` — функции и CLI;
- `tools/test_pde.py` — 35 тестов на примерах лекций и методички. Запуск: `python -X utf8 tools/test_pde.py`.

## Правила записи

- Производные как в лекциях: `u_xx`, `u_xy` (= `u_yx`), `u_x`, `u`; порядок не выше второго.
- Умножение только через `*`; степень `**` (или `^`); дроби вида `u_y/2`; доступны `sqrt`, `exp`, `sin`, `pi` и т. п.
- Правую часть можно отделить `=`: `"u_tt = a**2*u_xx + f(x, t)"`.
- Переменные задаются `--vars` (по умолчанию `"x y"`), например `"x t"`, `"x y z"`, `"ξ η"`. **Порядок важен**: первая играет роль x, вторая — y (характеристики dy/dx).
- Все прочие имена (a, k, l, T0, E, S …) — **положительные параметры**. Имя со скобкой — произвольная функция: `phi(x**2 + y**2)`, `F(y + 2*x)`.
- Знаки переменных: `--pos "x y"`, `--neg y`. Нужны, если тип зависит от области.
- Запускать всегда как `python -X utf8 …` (в выводе греческие буквы), из корня папки.

## Команды (реальный вывод)

### classify — тип уравнения

```
> python -X utf8 tools/pde.py classify "y*u_xx + u_yy"
D = a12² − a11·a22 = -y
гиперболический: y < 0
параболический: Eq(y, 0)
эллиптический: 0 < y

> python -X utf8 tools/pde.py classify "u_xy - 2*u_xz + u_yz + u_x + u_y/2" --vars "x y z"
инерция (+, −, 0): (2, 1, 0)
тип: гиперболический
```

### canonical — канонический вид

Две переменные — через характеристики:

```
> python -X utf8 tools/pde.py canonical "u_xx - 2*u_xy + u_yy + 9*u_x + 9*u_y - 9*u"
тип: параболический | D = 0
характеристики: x + y = C
ξ = x + y
η = x
v_ηη + 9*v_η + 18*v_ξ - 9*v = 0

> python -X utf8 tools/pde.py canonical "u_tt = a**2*u_xx" --vars "x t"
тип: гиперболический | D = a**2
характеристики: t + x/a = C; t - x/a = C
ξ = t + x/a
η = t - x/a
v_ξη = 0
```

Переменные коэффициенты (нужны знаки переменных):

```
> python -X utf8 tools/pde.py canonical "x**2*u_xx - y**2*u_yy" --pos "x y"
тип: гиперболический | D = x**2*y**2
характеристики: y/x = C; x*y = C
ξ = y/x
η = x*y
v_ξη - v_ξ/(2*η) = 0

> python -X utf8 tools/pde.py canonical "y*u_xx + u_yy" --neg y
тип: гиперболический | D = -y
характеристики: -3*x/2 + (-y)**(3/2) = C; 3*x/2 + (-y)**(3/2) = C
ξ = -3*x/2 + (-y)**(3/2)
η = 3*x/2 + (-y)**(3/2)
v_ξη + v_η/(6*(η + ξ)) + v_ξ/(6*(η + ξ)) = 0
```

⚠ Инструмент может выбрать характеристики с другим множителем или в другом порядке, чем в лекции. В последнем примере ξ = −(3/2)·η_лекц и η = (3/2)·ξ_лекц, поэтому уравнение эквивалентно лекционному v_ξη − (v_ξ − v_η)/(6(ξ − η)) = 0. Чтобы получить ровно переменные лекции, используйте `change` с явной заменой.

Без знака переменной `canonical` откажется работать: `ValueError: D = -y: тип зависит от области — задайте знаки переменных …`.

n переменных (или `--lagrange` для двух) — через квадратичную форму:

```
> python -X utf8 tools/pde.py canonical "u_xy - 2*u_xz + u_yz + u_x + u_y/2" --vars "x y z"
тип: гиперболический | β = [1, -1, 1]
P = [[1, -1, -sqrt(2)/2], [1, 1, sqrt(2)], [0, 0, sqrt(2)/2]]
B = Pᵀ = [[1, 1, 0], [-1, 1, 0], [-sqrt(2)/2, sqrt(2), sqrt(2)/2]]
ξ = x + y
η = -x + y
ω = -sqrt(2)*x/2 + sqrt(2)*y + sqrt(2)*z/2
v_ξξ + v_ωω - v_ηη - v_η/2 + 3*v_ξ/2 = 0
```

В методичке η = x − y, поэтому там стоит +½v_η; это тот же ответ.

```
> python -X utf8 tools/pde.py canonical "u_xx + 2*u_xy + 5*u_yy - 32*u" --lagrange
тип: эллиптический | β = [1, 1]
P = [[1, -1/2], [0, 1/2]]
B = Pᵀ = [[1, 0], [-1/2, 1/2]]
ξ = x
η = -x/2 + y/2
v_ηη + v_ξξ - 32*v = 0

> python -X utf8 tools/pde.py canonical "u_xx + 2*u_xy + u_yy + u_zz" --vars "x y z"
тип: параболический | β = [1, 1, 0]
…
ξ = x
η = z
ω = -x + y
v_ηη + v_ξξ = 0
```

### change — своя замена переменных

```
> python -X utf8 tools/pde.py change "y*u_xx + u_yy" --pos y --new "ξ=x" "η=2*y**(3/2)/3" --inverse "x=ξ" "y=(3*η/2)**(2/3)" --normalize
v_ηη + v_ξξ + v_η/(3*η) = 0
```

- `--inverse` нужен, если коэффициенты зависят от старых переменных. Без него обратная замена ищется через solve; если решение не найдено, коэффициенты остаются в старых переменных.
- `--normalize` делит уравнение на коэффициент первой ненулевой старшей производной.

### lower — убрать младшие производные (постоянные коэффициенты)

```
> python -X utf8 tools/pde.py lower "u_ηη + 18*u_ξ + 9*u_η - 9*u" --vars "ξ η"
u = w·exp(-9*η/2 + 13*ξ/8)
w_ηη + 18*w_ξ = 0
```

### check — подставить функцию в уравнение

```
> python -X utf8 tools/pde.py check "y*u_x - x*u_y" "phi(x**2 + y**2)"
невязка: 0 → решение
> python -X utf8 tools/pde.py check "u_xx + 2*u_xy - 3*u_yy" "3*x**2 + y**2"
невязка: 0 → решение
```

Начальные и граничные условия так не проверяются — для них нужен Python (`u.subs(...)`, см. ниже).

### dalembert, sine, cosine

```
> python -X utf8 tools/pde.py dalembert "sin(x)" "0" --a a
u(x, t) = sin(x)*cos(a*t)
> python -X utf8 tools/pde.py sine "x*(l-x)" --l l
b_n = 4*l**2*(1 - (-1)**n)/(pi**3*n**3)
> python -X utf8 tools/pde.py cosine "x" --l pi --n 0
a_n = pi
```

У `dalembert` источник задаётся через `--f "x"`. У `cosine` при `--n 0` печатается a₀ (ряд a₀/2 + Σ aₙcos).

## Python API — для нестандартных случаев

Скрипт класть в scratchpad и запускать `python -X utf8 script.py` из корня папки:

```python
import sys; sys.path.insert(0, "tools")
import sympy as sp
from pde import parse, classify, canonical2, canonical_n, change_vars, remove_lower, check, dalembert, show

y = sp.Symbol("y", negative=True)
r = canonical2("y*u_xx + u_yy", ["x", y])          # type, D, slopes, characteristics, new_vars, equation
print(r["type"], r["new_vars"], show(r["equation"]))

r = canonical_n("u_xy - 2*u_xz + u_yz", "x y z")   # betas, P, B, new_vars, equation, type
p = parse("u_tt = a**2*u_xx", "x t")               # p.vars, p.A, p.b, p.c, p.f

x, t = sp.symbols("x t", real=True)                # переменные инструмента вещественные
u = dalembert("exp(-x**2)", "x/(1 + x**2)", 3)
print(sp.simplify(u.subs(t, 0)), sp.simplify(sp.diff(u, t).subs(t, 0)))   # проверка начальных условий
```

## Ограничения

- только линейные (квазилинейные) уравнения не выше второго порядка;
- для переменных коэффициентов `canonical` вызывает dsolve и может не справиться. Тогда найти характеристики вручную и воспользоваться `change`;
- тип по областям даёт `classify`; `canonical` требует знаков переменных (`--pos` / `--neg`);
- `canonical_n` и `lower` работают только с постоянными коэффициентами;
- параметры считаются положительными. Если знак параметра важен, разбирать вручную или через Python с `Symbol`.
