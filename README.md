# Управление распределёнными системами — лабораторные работы

Курс: ДУЧП 2-го порядка и уравнения математической физики (лектор — Сорокина Н. В.).
Каждая лабораторная лежит в своей папке и запускается из неё; общие для курса
заметки и инструменты — в корне.

## Лабораторные

| Папка | Работа | Отчёт |
|---|---|---|
| [`lab1_canonical/`](lab1_canonical/) | Классификация ДУЧП 2-го порядка и приведение к каноническому виду (задания A, B, C) | `report/otchet.pdf`, `ОТЧЁТ.docx` |
| [`lab2_wave/`](lab2_wave/) | Одномерное волновое уравнение: метод характеристик на полупрямой и функция Грина (задания А, Б) | `report/otchet2.pdf` |
| [`lab3_heat/`](lab3_heat/) | Уравнение теплопроводности: формула Пуассона / функция Грина на прямой, полупрямая с u_x(0,t) = μ·ν (задания А, Б) | `report/otchet3.pdf`, `report/explanation3.pdf` |

Новая работа — новая папка `labN_<тема>/` с тем же устройством: свой пакет с расчётами,
`tests/`, `README.md`, `report/` с `.tex` и `figures/`.

## Общие материалы

| Папка | Что внутри |
|---|---|
| [`notes/`](notes/) | конспекты по темам курса: классификация, канонический вид, младшие члены, модели, методы решения |
| [`tools/`](tools/) | `pde.py` — проверка выкладок на sympy (только для самопроверки, в сдаваемый код не входит), `test_pde.py` — 35 тестов |

## Структура

```
.
├── lab1_canonical/        лабораторная 1
│   ├── pde_canonical/     метод Лагранжа, характеристики, области
│   ├── widget/            виджет (index.html)
│   ├── tests/             96 тестов
│   ├── report/            otchet.tex → otchet.pdf, figures/
│   ├── demo.py, plot_regions.py, build_docx.py
│   └── README.md, ОТЧЁТ.md, ОТЧЁТ.docx
├── lab2_wave/             лабораторная 2
│   ├── wavelab/           полупрямая, функция Грина, численные методы
│   ├── tests/             22 теста
│   ├── report/            otchet2.tex → otchet2.pdf, figures/
│   └── app.py, task_a.py, task_b.py, make_figures.py, README.md
├── lab3_heat/             лабораторная 3
│   ├── heatlab/           функции Грина, прямая, полупрямая, численные методы
│   ├── tests/             23 теста
│   ├── report/            otchet3.tex, explanation3.tex → PDF, figures/
│   └── app.py, task_a.py, task_b.py, make_figures.py, README.md
├── notes/                 конспекты курса
└── tools/                 pde.py — проверка на sympy
```

## Запуск тестов

```bash
cd lab1_canonical && python -m unittest discover -s tests -t .
cd lab2_wave      && python -m unittest discover -s tests -t .
cd lab3_heat      && python -m unittest discover -s tests -t .
python -X utf8 tools/test_pde.py          # из корня
```
