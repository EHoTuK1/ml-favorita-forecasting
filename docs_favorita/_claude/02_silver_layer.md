# 02. Саммари треда: решения по очистке и переход bronze → silver

Дата: 2026-10-24.

## Решения (журнал: `docs_favorita/0_raw_data_exploration/cleaning_decisions.md`)

- C1: вариант A, нули восстанавливаются только с 2015-06-01.
- C2: NULL в onpromotion остаётся, флаг `onpromotion_known`.
- C3: `unit_sales_raw` (исходное), `unit_sales = max(raw, 0)`, флаг `is_return`.
- C4: выбросы только флагами: `is_outlier` (ln-z > 5 по паре, x ≥ 100, у пары ≥ 30 продаж) и `is_event_spike` (2016-04-18…21, 19–24 декабря).
- Группа B оформлена таблицей в журнале.
- Праздники: `holidays_count` и `holiday_locale` (Local > Regional > National), оба поля.
- Плотная сетка: конец окна 2017-08-15 для всех пар; дни без продаж у магазина — `is_store_closed`, `unit_sales = NULL`; пары без продаж в окне не входят.

## Реализация

- Код: `scripts/silver_build.py` (build_* и verify_*, пороги — константы модуля).
- Запуск: `notebooks/04_silver.ipynb`, полный прогон около 2 минут при memory_limit 8 ГБ.
- Результат в `data/silver/`: stores, items, oil, transactions, holiday_features, sales_history (125,5 млн строк, 739 МБ), sales_daily/ (119,3 млн строк, файл на месяц, около 350 МБ).
- Все 31 проверка пройдены; суммы продаж сверяются точно через DECIMAL.

## Цифры sales_daily

119 312 458 строк; восстановлено нулей 40 889 984 (34,3 %); закрытых дней магазина 823 471; возвратов 5 015; `is_outlier` 9 642.

## Находки

- Максимум 89 440 (магазин 39, товар 1976284) не выброс: у пары 25 продаж с медианой 3 817, оптовая позиция.
- approx_quantile по 175 тыс. пар не помещается в 8 ГБ, поэтому выбросы считаются через среднее и std логарифма.
- duckdb 2.x: после `execute(...).fetchone()` соединение блокируется до дочитывания; в silver_build используется fetchall.

## Следующие шаги

- Gold: витрины для EDA и обучения поверх sales_daily + holiday_features + oil + transactions.
- Мелочи из саммари 01 (ER-схема, тип test.store_nbr, sha256 landing) не трогались.
