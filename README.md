# О проекте
Реализация группового метода парных сравнений для ранжирования объектов с поддержкой различных алгоритмов вычисления агрегированных матриц. 
Проект включает:
- Скрипт для проведения экспериментов с разными методами агрегации экспертных оценок
- Инструмент анализа результатов через корреляционный коэффициент Спирмена
  
# Участники
1. Руководитель и соавтор проекта Владимир А. Пархоменко, старший преподаватель ИКНК СПбПУ.
1. Разработчик программы ранжирования и анализа результатов Любовь А. Лаврова, студентка ИКНК СПбПУ
# Структура проекта
- PairwiseComparison.py - модуль для ранжирования объектов разными методами.
- example_usage.py - скрипт для запуска экспериментов на всех данных из директории и расчета корреляционных метрик
- example_usage2.py - скрипт для запуска экспериментов на данных из файла и расчета корреляционных метрик.
- /data - настоящие и синтетические наборы данных для экспериментов
# Быстрый старт
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate
python example_usage.py
```
# Гарантия
Разработчики не дают никаких гарантий по поводу использования данного программного обеспечения.
# Лицензия
Эта программа открыта для использования и распространяется под лицензией MIT.
# About
Implementation of a group method of paired comparisons for ranking objects with support for various algorithms for calculating aggregated matrices. 
The project includes:
- A script for conducting experiments with different methods of aggregation of expert assessments
- A tool for analyzing results through the Spearman correlation coefficient
# Persons
1. The head and co-author of the project, Vladimir A. Parkhomenko, senior lecturer at the ICNK SPbPU.
2. The developer of the ranking and results analysis program Lyubov A. Lavrova, a student of the ICNK SPbPU
# Structure
- PairwiseComparison.py - a module for ranking objects using different methods.
- example_usage.py - a script for running experiments on all data from the directory and calculating correlation metrics
- example_usage2.py - a script for running experiments on data from a file and calculating correlation metrics.
- /data - real and synthetic datasets for experiments
# Quick start
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate
python example_usage.py
```
# Warranty
The contributors give no warranty for the using of the software.
# License 
This program is open to use anywhere and is licensed under the MIT license.
