"""Построение слоя silver (data/silver) из bronze (data/bronze).

Решения по очистке — в docs_favorita/0_raw_data_exploration/cleaning_decisions.md;
номера в комментариях (C1, S3 и т. д.) ссылаются на него.

Принципы:
- значения bronze не теряются: исходные продажи лежат в `unit_sales_raw`,
  проблемы помечаются флагами, отбор делается в gold или в модели;
- все шаги детерминированы и повторяемы: повторный запуск перезаписывает результат;
- запись идёт во временный файл (каталог), который переименовывается только после
  успеха, поэтому после сбоя не остаётся полузаписанного parquet;
- плотная таблица продаж пишется по месяцам: в памяти одновременно только один месяц.

Все пути относительны корня проекта (в ноутбуках он задан jupyter.notebookFileRoot).
"""

import contextlib
import os
import shutil
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from scripts.parquet_convert import BRONZE_DIR, Check, VerificationReport

__all__ = [
    "BRONZE_TABLES",
    "SILVER_DIR",
    "SILVER_TABLES",
    "TEST_END",
    "TRAIN_END",
    "WINDOW_START",
    "build_all",
    "build_holiday_features",
    "build_items",
    "build_oil",
    "build_sales_daily",
    "build_sales_history",
    "build_stores",
    "build_transactions",
    "register_bronze",
    "register_silver",
    "verify_all",
]

SILVER_DIR = Path("data/silver")

# C1: с этой даты ассортимент в train записан полностью, нули восстанавливаются только с неё
WINDOW_START = date(2015, 6, 1)
TRAIN_END = date(2017, 8, 15)
TEST_END = date(2017, 8, 31)

# C4: точечный выброс — продажа сильно выше обычной для пары (магазин, товар).
# Сравнение в логарифмах: у товаров с продажами 1–5 шт. и 500–1000 шт. разброс сопоставим.
OUTLIER_Z = 5.0  # порог: ln(1 + x) > среднее + OUTLIER_Z * std по паре
OUTLIER_MIN_SALES = 100  # мелкие значения не считаем выбросами, даже если z велик
OUTLIER_MIN_ROWS = 30  # по короткой истории std ненадёжна, такие пары не помечаем

# C4: событийные всплески — реальный спрос, помечаются отдельно от точечных выбросов
EARTHQUAKE_SPIKE = (date(2016, 4, 18), date(2016, 4, 21))
CHRISTMAS_SPIKE_DAYS = (19, 24)  # 19–24 декабря каждого года

# 4.3: «весовой» товар — дробные продажи больше чем в половине строк
WEIGHTED_SHARE = 0.5

BRONZE_TABLES = ("train", "stores", "items", "transactions", "oil", "holidays_events")
SILVER_TABLES = (
    "stores",
    "items",
    "oil",
    "transactions",
    "holiday_features",
    "sales_history",
    "sales_daily",
)

# Временная таблица со статистикой пар, общая для sales_history и sales_daily
_PAIR_STATS = "silver_pair_stats"


# --- Служебные функции -------------------------------------------------------


def _quote_literal(value: str) -> str:
    """Экранирует строковый литерал для SQL."""
    return "'" + value.replace("'", "''") + "'"


def _date_sql(d: date) -> str:
    return f"DATE '{d.isoformat()}'"


def _fetch_row(con: duckdb.DuckDBPyConnection, query: str) -> tuple[Any, ...]:
    # fetchall, а не fetchone: в duckdb 2.x недочитанный результат блокирует соединение
    rows = con.execute(query).fetchall()
    if not rows:
        raise RuntimeError(f"Запрос не вернул строк:\n{query}")
    return rows[0]


def _scalar(con: duckdb.DuckDBPyConnection, query: str) -> Any:
    return _fetch_row(con, query)[0]


def _copy_to_parquet(con: duckdb.DuckDBPyConnection, query: str, target: Path) -> None:
    """Пишет результат запроса в parquet через временный файл и атомарную замену."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.unlink(missing_ok=True)  # остаток прошлого неудачного запуска
    try:
        con.execute(
            f"COPY ({query}) TO {_quote_literal(tmp.as_posix())} "
            "(FORMAT parquet, COMPRESSION 'zstd')"
        )
        # На Windows упадёт, если target открыт другим процессом
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def _silver_source(table: str, silver_dir: Path) -> str:
    """SQL-источник таблицы silver: файл <table>.parquet или каталог <table>/ с частями."""
    f = silver_dir / f"{table}.parquet"
    if f.is_file():
        return f"read_parquet({_quote_literal(f.as_posix())})"
    d = silver_dir / table
    if d.is_dir():
        return f"read_parquet({_quote_literal((d / '*.parquet').as_posix())})"
    raise FileNotFoundError(f"Не найден silver для таблицы {table!r} в {silver_dir}")


def _event_spike_sql(col: str = "date") -> str:
    """Флаг событийного всплеска по дате (C4)."""
    q_start, q_end = EARTHQUAKE_SPIKE
    d_from, d_to = CHRISTMAS_SPIKE_DAYS
    return (
        f"({col} BETWEEN {_date_sql(q_start)} AND {_date_sql(q_end)} "
        f"OR (month({col}) = 12 AND day({col}) BETWEEN {d_from} AND {d_to}))"
    )


def _outlier_sql(sales: str, stats: str) -> str:
    """Флаг точечного выброса (C4); `stats` — алиас таблицы статистик пары.

    CASE вместо AND: ln от отрицательных продаж (возвратов) не вычисляется.
    """
    return (
        f"coalesce(CASE WHEN {sales} >= {OUTLIER_MIN_SALES} AND {stats}.n >= {OUTLIER_MIN_ROWS} "
        f"THEN ln(1 + {sales}) > {stats}.mu + {OUTLIER_Z} * {stats}.sd END, false)"
    )


def _month_starts(start: date, end: date) -> list[date]:
    months: list[date] = []
    d = start.replace(day=1)
    while d <= end:
        months.append(d)
        d = d.replace(year=d.year + 1, month=1) if d.month == 12 else d.replace(month=d.month + 1)
    return months


def _month_end(d: date) -> date:
    nxt = d.replace(year=d.year + 1, month=1) if d.month == 12 else d.replace(month=d.month + 1)
    return date.fromordinal(nxt.toordinal() - 1)


# --- Регистрация источников --------------------------------------------------


def register_bronze(con: duckdb.DuckDBPyConnection, bronze_dir: Path = BRONZE_DIR) -> None:
    """Создаёт представления bronze_<table> поверх parquet из bronze."""
    for t in BRONZE_TABLES:
        path = (bronze_dir / f"{t}.parquet").as_posix()
        con.execute(
            f"CREATE OR REPLACE VIEW bronze_{t} AS "
            f"SELECT * FROM read_parquet({_quote_literal(path)})"
        )


def register_silver(con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR) -> None:
    """Создаёт представления silver_<table> поверх готового silver."""
    for t in SILVER_TABLES:
        con.execute(
            f"CREATE OR REPLACE VIEW silver_{t} AS SELECT * FROM {_silver_source(t, silver_dir)}"
        )


def _ensure_pair_stats(con: duckdb.DuckDBPyConnection) -> None:
    """Статистика пары (магазин, товар) по всей истории train.

    first_date / last_date — границы продаж пары (окно S3);
    n, mu, sd — число положительных продаж, среднее и std ln(1 + x) (выбросы C4).
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {_PAIR_STATS} AS
        SELECT
            store_nbr,
            item_nbr,
            min(date)                                                      AS first_date,
            max(date)                                                      AS last_date,
            count(*) FILTER (WHERE unit_sales > 0)                         AS n,
            avg(ln(1 + unit_sales)) FILTER (WHERE unit_sales > 0)          AS mu,
            stddev_samp(ln(1 + unit_sales)) FILTER (WHERE unit_sales > 0)  AS sd
        FROM bronze_train
        GROUP BY store_nbr, item_nbr
    """)


# --- Справочники и внешние ряды (S1) -----------------------------------------


def build_stores(con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR) -> Path:
    """stores без изменений: только порядок строк."""
    target = silver_dir / "stores.parquet"
    _copy_to_parquet(con, "SELECT * FROM bronze_stores ORDER BY store_nbr", target)
    return target


def build_items(con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR) -> Path:
    """items + is_weighted (4.3): доля строк train с дробными продажами > WEIGHTED_SHARE.

    Товары без продаж в train получают is_weighted = false.
    """
    target = silver_dir / "items.parquet"
    query = f"""
        WITH frac AS (
            SELECT item_nbr, avg(CAST(unit_sales <> floor(unit_sales) AS INTEGER)) AS share
            FROM bronze_train
            GROUP BY item_nbr
        )
        SELECT i.*, coalesce(f.share > {WEIGHTED_SHARE}, false) AS is_weighted
        FROM bronze_items i
        LEFT JOIN frac f USING (item_nbr)
        ORDER BY item_nbr
    """
    _copy_to_parquet(con, query, target)
    return target


def build_oil(con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR) -> Path:
    """oil на полном календаре (3.2).

    dcoilwtico — заполнение только из прошлого (последнее известное значение);
    dcoilwtico_raw — исходное значение; is_imputed — значение взято из прошлого.
    Первая дата (2013-01-01) в bronze NULL, заполнять её нечем: остаётся NULL.
    """
    target = silver_dir / "oil.parquet"
    query = """
        WITH cal AS (
            SELECT CAST(unnest(generate_series(
                (SELECT min(date) FROM bronze_oil),
                (SELECT max(date) FROM bronze_oil),
                INTERVAL 1 DAY
            )) AS DATE) AS date
        ),
        filled AS (
            SELECT
                c.date,
                o.dcoilwtico AS dcoilwtico_raw,
                last_value(o.dcoilwtico IGNORE NULLS) OVER (
                    ORDER BY c.date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS dcoilwtico
            FROM cal c
            LEFT JOIN bronze_oil o USING (date)
        )
        SELECT
            date,
            dcoilwtico,
            dcoilwtico_raw,
            dcoilwtico_raw IS NULL AND dcoilwtico IS NOT NULL AS is_imputed
        FROM filled
        ORDER BY date
    """
    _copy_to_parquet(con, query, target)
    return target


def build_transactions(con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR) -> Path:
    """transactions + пары (date, store_nbr), где продажи в train есть, а чеков нет (3.3).

    Для таких пар transactions = NULL и transactions_missing = true: это потеря записи,
    а не закрытый магазин, поэтому 0 не ставится. Сравнение — на диапазоне дат transactions.
    """
    target = silver_dir / "transactions.parquet"
    query = """
        WITH sold AS (
            SELECT DISTINCT date, store_nbr
            FROM bronze_train
            WHERE date BETWEEN (SELECT min(date) FROM bronze_transactions)
                           AND (SELECT max(date) FROM bronze_transactions)
        )
        SELECT
            date,
            store_nbr,
            t.transactions,
            t.transactions IS NULL AS transactions_missing
        FROM bronze_transactions t
        FULL JOIN sold USING (date, store_nbr)
        ORDER BY date, store_nbr
    """
    _copy_to_parquet(con, query, target)
    return target


# --- Праздники (S2) ----------------------------------------------------------


def build_holiday_features(
    con: duckdb.DuckDBPyConnection,
    silver_dir: Path = SILVER_DIR,
    *,
    end: date = TEST_END,
) -> Path:
    """Признаки праздников на сетке (date, store_nbr): от начала train до `end`.

    Правила (S2, 1.2, 2.4):
    - область: National — все магазины, Regional — stores.state, Local — stores.city;
      соединение строго по locale, иначе Esmeraldas и Loja дали бы ложные пары;
    - is_holiday: Holiday (не перенесённый), Transfer (фактическая дата переноса), Additional;
    - is_transferred_from: исходная дата перенесённого праздника (в этот день работают);
    - is_bridge, is_work_day, is_event — отдельные флаги, не праздники;
    - holidays_count — число праздничных записей в этот день, holiday_locale — самая
      узкая область среди них (Local > Regional > National);
    - days_since_prev_holiday / days_to_next_holiday — по магазину, только по is_holiday,
      0 в сам праздник. Считаются на всём диапазоне holidays_events (2012–2017-12),
      поэтому на краях сетки расстояния тоже известны. Календарь праздников известен
      заранее, утечки из будущего нет.
    """
    target = silver_dir / "holiday_features.parquet"
    query = f"""
        WITH ev AS (
            SELECT
                h.date,
                s.store_nbr,
                h.type,
                h.locale,
                h.type IN ('Holiday', 'Transfer', 'Additional') AND NOT h.transferred AS is_hol
            FROM bronze_holidays_events h
            JOIN bronze_stores s
              ON h.locale = 'National'
              OR (h.locale = 'Regional' AND h.locale_name = s.state)
              OR (h.locale = 'Local' AND h.locale_name = s.city)
        ),
        agg AS (
            SELECT
                date,
                store_nbr,
                bool_or(is_hol)                                  AS is_holiday,
                bool_or(type = 'Holiday' AND NOT is_hol)         AS is_transferred_from,
                bool_or(type = 'Bridge')                         AS is_bridge,
                bool_or(type = 'Work Day')                       AS is_work_day,
                bool_or(type = 'Event')                          AS is_event,
                count(*) FILTER (WHERE is_hol)                   AS holidays_count,
                max(CASE WHEN is_hol THEN
                        CASE locale WHEN 'Local' THEN 3 WHEN 'Regional' THEN 2 ELSE 1 END
                    END)                                         AS locale_rank
            FROM ev
            GROUP BY date, store_nbr
        ),
        cal AS (
            SELECT CAST(unnest(generate_series(
                least((SELECT min(date) FROM bronze_holidays_events),
                      (SELECT min(date) FROM bronze_train)),
                greatest((SELECT max(date) FROM bronze_holidays_events), {_date_sql(end)}),
                INTERVAL 1 DAY
            )) AS DATE) AS date
        ),
        grid AS (
            SELECT
                c.date,
                s.store_nbr,
                coalesce(a.is_holiday, false)           AS is_holiday,
                coalesce(a.is_transferred_from, false)  AS is_transferred_from,
                coalesce(a.is_bridge, false)            AS is_bridge,
                coalesce(a.is_work_day, false)          AS is_work_day,
                coalesce(a.is_event, false)             AS is_event,
                CAST(coalesce(a.holidays_count, 0) AS SMALLINT) AS holidays_count,
                CASE a.locale_rank
                    WHEN 3 THEN 'Local' WHEN 2 THEN 'Regional' WHEN 1 THEN 'National'
                END                                     AS holiday_locale
            FROM cal c
            CROSS JOIN bronze_stores s
            LEFT JOIN agg a ON a.date = c.date AND a.store_nbr = s.store_nbr
        ),
        dist AS (
            SELECT
                *,
                date_diff('day',
                    max(CASE WHEN is_holiday THEN date END) OVER (
                        PARTITION BY store_nbr ORDER BY date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                    date) AS days_since_prev_holiday,
                date_diff('day',
                    date,
                    min(CASE WHEN is_holiday THEN date END) OVER (
                        PARTITION BY store_nbr ORDER BY date
                        ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)
                    ) AS days_to_next_holiday
            FROM grid
        )
        SELECT *
        FROM dist
        WHERE date BETWEEN (SELECT min(date) FROM bronze_train) AND {_date_sql(end)}
        ORDER BY date, store_nbr
    """
    _copy_to_parquet(con, query, target)
    return target


# --- Продажи -----------------------------------------------------------------


def build_sales_history(con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR) -> Path:
    """Разреженная история продаж: все строки train с флагами (C1–C4), без восстановления нулей.

    Нужна для EDA по всему периоду; для обучения — sales_daily.
    - unit_sales = max(unit_sales_raw, 0), is_return — исходное значение < 0 (C3);
    - onpromotion_known — onpromotion не NULL (C2);
    - is_outlier, is_event_spike — выбросы (C4);
    - is_reliable_period — дата не раньше WINDOW_START (C1).
    """
    _ensure_pair_stats(con)
    target = silver_dir / "sales_history.parquet"
    query = f"""
        SELECT
            t.id,
            t.date,
            t.store_nbr,
            t.item_nbr,
            greatest(t.unit_sales, 0)                     AS unit_sales,
            t.unit_sales                                  AS unit_sales_raw,
            t.onpromotion,
            t.onpromotion IS NOT NULL                     AS onpromotion_known,
            t.unit_sales < 0                              AS is_return,
            {_outlier_sql("t.unit_sales", "p")}           AS is_outlier,
            {_event_spike_sql("t.date")}                  AS is_event_spike,
            t.date >= {_date_sql(WINDOW_START)}           AS is_reliable_period
        FROM bronze_train t
        LEFT JOIN {_PAIR_STATS} p USING (store_nbr, item_nbr)
        ORDER BY t.date, t.store_nbr, t.item_nbr
    """
    _copy_to_parquet(con, query, target)
    return target


def _sales_daily_month_sql(month_start: date, month_end: date) -> str:
    """Плотная сетка продаж за один месяц (S3)."""
    lo, hi = _date_sql(month_start), _date_sql(month_end)
    return f"""
        WITH cal AS (
            SELECT CAST(d AS DATE) AS date
            FROM (SELECT unnest(generate_series({lo}, {hi}, INTERVAL 1 DAY)) AS d)
            WHERE NOT (month(d) = 12 AND day(d) = 25)
        ),
        pairs AS (
            SELECT store_nbr, item_nbr, greatest(first_date, {_date_sql(WINDOW_START)}) AS start,
                   n, mu, sd
            FROM {_PAIR_STATS}
            WHERE last_date >= {_date_sql(WINDOW_START)}
        ),
        sales AS (
            SELECT date, store_nbr, item_nbr, unit_sales, onpromotion
            FROM bronze_train
            WHERE date BETWEEN {lo} AND {hi}
        ),
        store_open AS (
            SELECT DISTINCT date, store_nbr, true AS is_open
            FROM sales
        )
        SELECT
            c.date,
            p.store_nbr,
            p.item_nbr,
            CASE WHEN s.unit_sales IS NOT NULL THEN greatest(s.unit_sales, 0)
                 WHEN o.is_open THEN 0 END                          AS unit_sales,
            CASE WHEN s.unit_sales IS NOT NULL THEN s.unit_sales
                 WHEN o.is_open THEN 0 END                          AS unit_sales_raw,
            s.onpromotion,
            s.onpromotion IS NOT NULL                               AS onpromotion_known,
            s.unit_sales IS NULL AND coalesce(o.is_open, false)     AS is_restored,
            NOT coalesce(o.is_open, false)                          AS is_store_closed,
            coalesce(s.unit_sales < 0, false)                       AS is_return,
            {_outlier_sql("s.unit_sales", "p")}                     AS is_outlier,
            {_event_spike_sql("c.date")}                            AS is_event_spike
        FROM pairs p
        JOIN cal c ON c.date >= p.start
        LEFT JOIN sales s
               ON s.date = c.date AND s.store_nbr = p.store_nbr AND s.item_nbr = p.item_nbr
        LEFT JOIN store_open o
               ON o.date = c.date AND o.store_nbr = p.store_nbr
        ORDER BY c.date, p.store_nbr, p.item_nbr
    """


def build_sales_daily(
    con: duckdb.DuckDBPyConnection,
    silver_dir: Path = SILVER_DIR,
    *,
    verbose: bool = True,
) -> Path:
    """Плотная таблица продаж (магазин × товар × дата) с восстановленными нулями (S3).

    Окно пары: от max(первая продажа пары, WINDOW_START) до TRAIN_END. Пары, у которых
    последняя продажа раньше WINDOW_START, в сетку не входят: в окне у них одни нули.
    - 25 декабря исключено из календаря (магазины закрыты, строк нет нигде);
    - день, когда у магазина нет ни одной продажи, — закрытие (1 января, периоды молчания):
      строка остаётся в сетке, но unit_sales = NULL и is_store_closed = true, ноль не ставится;
    - отсутствие строки в открытый день — ноль: unit_sales = 0, is_restored = true,
      onpromotion = NULL (акция в этот день неизвестна).

    Результат — каталог sales_daily/ с файлом на каждый месяц (YYYY-MM.parquet).
    """
    _ensure_pair_stats(con)
    target = silver_dir / "sales_daily"
    tmp = silver_dir / "sales_daily.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        for m in _month_starts(WINDOW_START, TRAIN_END):
            m_end = min(_month_end(m), TRAIN_END)
            path = tmp / f"{m:%Y-%m}.parquet"
            con.execute(
                f"COPY ({_sales_daily_month_sql(m, m_end)}) TO "
                f"{_quote_literal(path.as_posix())} (FORMAT parquet, COMPRESSION 'zstd')"
            )
            if verbose:
                print(f"{m:%Y-%m} записан", flush=True)
        shutil.rmtree(target, ignore_errors=True)
        os.replace(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


# --- Всё вместе --------------------------------------------------------------


def build_all(
    con: duckdb.DuckDBPyConnection,
    tables: Iterable[str] | None = None,
    *,
    silver_dir: Path = SILVER_DIR,
) -> dict[str, Path]:
    """Строит таблицы silver (по умолчанию все из SILVER_TABLES) в фиксированном порядке."""
    builders = {
        "stores": build_stores,
        "items": build_items,
        "oil": build_oil,
        "transactions": build_transactions,
        "holiday_features": build_holiday_features,
        "sales_history": build_sales_history,
        "sales_daily": build_sales_daily,
    }
    wanted = set(tables or SILVER_TABLES)
    return {t: builders[t](con, silver_dir) for t in SILVER_TABLES if t in wanted}


# --- Проверка ----------------------------------------------------------------

# Ожидаемые числа из профилирования (02_profiling.ipynb, таблица «Выводы»)
EXPECTED_WEIGHTED_ITEMS = 293
EXPECTED_MISSING_TRANSACTIONS = 118
EXPECTED_RETURNS = 7_795
EXPECTED_PROMO_NULLS = 21_657_651


def _dec_sum(col: str) -> str:
    """Точная сумма: DECIMAL не зависит от порядка сложения, в отличие от DOUBLE."""
    return f"sum(CAST({col} AS DECIMAL(18, 3)))"


def _duplicates(con: duckdb.DuckDBPyConnection, source: str, key: str) -> int:
    return int(
        _scalar(
            con,
            f"""
        SELECT count(*) FROM (SELECT {key} FROM {source} GROUP BY {key} HAVING count(*) > 1)
    """,
        )
    )


def verify_stores(
    con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR
) -> VerificationReport:
    src = _silver_source("stores", silver_dir)
    checks = [
        Check(
            "rows",
            _scalar(con, "SELECT count(*) FROM bronze_stores"),
            _scalar(con, f"SELECT count(*) FROM {src}"),
        ),
        Check("duplicates.store_nbr", 0, _duplicates(con, src, "store_nbr")),
    ]
    return VerificationReport("stores", tuple(checks))


def verify_items(
    con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR
) -> VerificationReport:
    src = _silver_source("items", silver_dir)
    checks = [
        Check(
            "rows",
            _scalar(con, "SELECT count(*) FROM bronze_items"),
            _scalar(con, f"SELECT count(*) FROM {src}"),
        ),
        Check("duplicates.item_nbr", 0, _duplicates(con, src, "item_nbr")),
        Check(
            "is_weighted",
            EXPECTED_WEIGHTED_ITEMS,
            _scalar(con, f"SELECT count(*) FILTER (WHERE is_weighted) FROM {src}"),
        ),
    ]
    return VerificationReport("items", tuple(checks))


def verify_oil(con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR) -> VerificationReport:
    src = _silver_source("oil", silver_dir)
    days = _scalar(con, "SELECT date_diff('day', min(date), max(date)) + 1 FROM bronze_oil")
    checks = [
        Check("rows = календарные дни", days, _scalar(con, f"SELECT count(*) FROM {src}")),
        Check("duplicates.date", 0, _duplicates(con, src, "date")),
        # NULL допустим только до первого известного значения
        Check(
            "nulls после первого значения",
            0,
            _scalar(
                con,
                f"""
            SELECT count(*) FROM {src}
            WHERE dcoilwtico IS NULL
              AND date > (SELECT min(date) FROM {src} WHERE dcoilwtico_raw IS NOT NULL)
        """,
            ),
        ),
        Check(
            "исходные значения не изменены",
            0,
            _scalar(
                con,
                f"""
            SELECT count(*) FROM {src}
            WHERE dcoilwtico_raw IS NOT NULL AND dcoilwtico <> dcoilwtico_raw
        """,
            ),
        ),
        Check(
            "непустые значения = bronze",
            _scalar(con, "SELECT count(dcoilwtico) FROM bronze_oil"),
            _scalar(con, f"SELECT count(dcoilwtico_raw) FROM {src}"),
        ),
    ]
    return VerificationReport("oil", tuple(checks))


def verify_transactions(
    con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR
) -> VerificationReport:
    src = _silver_source("transactions", silver_dir)
    bronze_rows = _scalar(con, "SELECT count(*) FROM bronze_transactions")
    checks = [
        Check(
            "rows = bronze + пропуски",
            bronze_rows + EXPECTED_MISSING_TRANSACTIONS,
            _scalar(con, f"SELECT count(*) FROM {src}"),
        ),
        Check(
            "transactions_missing",
            EXPECTED_MISSING_TRANSACTIONS,
            _scalar(con, f"SELECT count(*) FILTER (WHERE transactions_missing) FROM {src}"),
        ),
        Check("duplicates.(date, store_nbr)", 0, _duplicates(con, src, "date, store_nbr")),
        Check(
            "sum(transactions) = bronze",
            _scalar(con, "SELECT sum(transactions) FROM bronze_transactions"),
            _scalar(con, f"SELECT sum(transactions) FROM {src}"),
        ),
    ]
    return VerificationReport("transactions", tuple(checks))


def verify_holiday_features(
    con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR, *, end: date = TEST_END
) -> VerificationReport:
    src = _silver_source("holiday_features", silver_dir)
    days = _scalar(
        con, f"SELECT date_diff('day', min(date), {_date_sql(end)}) + 1 FROM bronze_train"
    )
    stores = _scalar(con, "SELECT count(*) FROM bronze_stores")
    checks = [
        Check("rows = дни × магазины", days * stores, _scalar(con, f"SELECT count(*) FROM {src}")),
        Check("duplicates.(date, store_nbr)", 0, _duplicates(con, src, "date, store_nbr")),
        Check(
            "nulls в расстояниях до праздников",
            0,
            _scalar(
                con,
                f"""
            SELECT count(*) FROM {src}
            WHERE days_since_prev_holiday IS NULL OR days_to_next_holiday IS NULL
        """,
            ),
        ),
        # Национальный праздник обязан дойти до всех магазинов
        Check(
            "национальные праздники во всех магазинах",
            0,
            _scalar(
                con,
                f"""
            SELECT count(*) FROM bronze_holidays_events h
            JOIN {src} f USING (date)
            WHERE h.locale = 'National' AND h.type IN ('Holiday', 'Transfer', 'Additional')
              AND NOT h.transferred AND NOT f.is_holiday
        """,
            ),
        ),
    ]
    return VerificationReport("holiday_features", tuple(checks))


def verify_sales_history(
    con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR
) -> VerificationReport:
    src = _silver_source("sales_history", silver_dir)
    b_rows, b_sum, b_pos = _fetch_row(
        con,
        f"""
        SELECT count(*), {_dec_sum("unit_sales")}, {_dec_sum("greatest(unit_sales, 0)")}
        FROM bronze_train
    """,
    )
    s_rows, s_raw, s_sum, s_ret, s_unknown = _fetch_row(
        con,
        f"""
        SELECT count(*), {_dec_sum("unit_sales_raw")}, {_dec_sum("unit_sales")},
               count(*) FILTER (WHERE is_return), count(*) FILTER (WHERE NOT onpromotion_known)
        FROM {src}
    """,
    )
    checks = [
        Check("rows = bronze", b_rows, s_rows),
        Check("sum(unit_sales_raw) = bronze", b_sum, s_raw),
        Check("sum(unit_sales) = bronze без возвратов", b_pos, s_sum),
        Check("is_return", EXPECTED_RETURNS, s_ret),
        Check("onpromotion_known = false", EXPECTED_PROMO_NULLS, s_unknown),
    ]
    return VerificationReport("sales_history", tuple(checks))


def verify_sales_daily(
    con: duckdb.DuckDBPyConnection, silver_dir: Path = SILVER_DIR
) -> VerificationReport:
    """Инварианты S3: размер сетки, ключ, сумма продаж и состав строк."""
    _ensure_pair_stats(con)
    src = _silver_source("sales_daily", silver_dir)
    ws, te = _date_sql(WINDOW_START), _date_sql(TRAIN_END)

    # Ожидаемый размер сетки: для каждой пары — число дней календаря (без 25.12) в окне
    expected_rows = _scalar(
        con,
        f"""
        WITH cal AS (
            SELECT CAST(d AS DATE) AS date
            FROM (SELECT unnest(generate_series({ws}, {te}, INTERVAL 1 DAY)) AS d)
            WHERE NOT (month(d) = 12 AND day(d) = 25)
        )
        SELECT count(*)
        FROM {_PAIR_STATS} p
        JOIN cal c ON c.date >= greatest(p.first_date, {ws})
        WHERE p.last_date >= {ws}
    """,
    )
    b_rows, b_sum = _fetch_row(
        con,
        f"""
        SELECT count(*), {_dec_sum("unit_sales")} FROM bronze_train WHERE date >= {ws}
    """,
    )
    (rows, raw_sum, kept, min_date, dec25, closed_with_sales, restored_with_promo) = _fetch_row(
        con,
        f"""
        SELECT
            count(*),
            {_dec_sum("unit_sales_raw")},
            count(*) FILTER (WHERE NOT is_restored AND NOT is_store_closed),
            min(date),
            count(*) FILTER (WHERE month(date) = 12 AND day(date) = 25),
            count(*) FILTER (WHERE is_store_closed AND unit_sales IS NOT NULL),
            count(*) FILTER (WHERE is_restored AND onpromotion IS NOT NULL)
        FROM {src}
    """,
    )
    checks = [
        Check("rows = размер сетки", expected_rows, rows),
        Check(
            "duplicates.(date, store_nbr, item_nbr)",
            0,
            _duplicates(con, src, "date, store_nbr, item_nbr"),
        ),
        Check("sum(unit_sales_raw) = bronze в окне", b_sum, raw_sum),
        Check("исходные строки = bronze в окне", b_rows, kept),
        Check("min(date) >= WINDOW_START", True, min_date >= WINDOW_START),
        Check("строк за 25.12", 0, dec25),
        Check("закрытые дни без продаж", 0, closed_with_sales),
        Check("восстановленные строки без onpromotion", 0, restored_with_promo),
    ]
    return VerificationReport("sales_daily", tuple(checks))


def verify_all(
    con: duckdb.DuckDBPyConnection,
    tables: Iterable[str] | None = None,
    *,
    silver_dir: Path = SILVER_DIR,
) -> list[VerificationReport]:
    """Проверяет таблицы silver (по умолчанию все из SILVER_TABLES)."""
    verifiers = {
        "stores": verify_stores,
        "items": verify_items,
        "oil": verify_oil,
        "transactions": verify_transactions,
        "holiday_features": verify_holiday_features,
        "sales_history": verify_sales_history,
        "sales_daily": verify_sales_daily,
    }
    wanted = set(tables or SILVER_TABLES)
    return [verifiers[t](con, silver_dir) for t in SILVER_TABLES if t in wanted]
