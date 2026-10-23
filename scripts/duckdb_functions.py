"""Вспомогательные функции для разведочного анализа данных через DuckDB.

Во всех функциях `source` — готовый SQL-фрагмент для FROM: имя таблицы,
путь к файлу в кавычках ("'data/landing/stores.csv'") или подзапрос в скобках.
Он подставляется в запрос как есть, поэтому не передавайте сюда внешний ввод.
"""

import math
from typing import Any

import duckdb
import pandas as pd

__all__ = [
    "columns",
    "show_columns",
    "num_stats",
    "show_num_stats",
    "cat_stats",
    "show_cat_stats",
    "date_stats",
    "show_date_stats",
]

_INT_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
}
_FLOAT_TYPES = {"FLOAT", "REAL", "DOUBLE"}
_CAT_PREFIXES = ("VARCHAR", "ENUM", "BOOLEAN")
# DATE, TIMESTAMP, TIMESTAMP_S/MS/NS, TIMESTAMP WITH TIME ZONE
_DATE_PREFIXES = ("DATE", "TIMESTAMP")


# --- Служебные функции -------------------------------------------------------

def _quote_ident(name: str) -> str:
    """Экранирует имя колонки для подстановки в SQL."""
    return '"' + name.replace('"', '""') + '"'


def _fetch_row(con: duckdb.DuckDBPyConnection, query: str) -> tuple[Any, ...]:
    """Выполняет запрос и возвращает единственную строку результата."""
    row = con.execute(query).fetchone()
    if row is None:
        raise RuntimeError(f"Запрос не вернул строк:\n{query}")
    return row


def _kind(dtype: str) -> str:
    """Грубая классификация типа DuckDB: int / float / decimal / date / cat / other."""
    if dtype in _INT_TYPES:
        return "int"
    if dtype in _FLOAT_TYPES:
        return "float"
    if dtype.startswith("DECIMAL"):
        return "decimal"
    if dtype.startswith(_DATE_PREFIXES):
        return "date"
    if dtype.startswith(_CAT_PREFIXES):
        return "cat"
    return "other"


def _column_type(con: duckdb.DuckDBPyConnection, source: str, col: str) -> str:
    """Тип колонки по схеме, а не по данным: работает и на пустой таблице."""
    return _fetch_row(con, f"DESCRIBE SELECT {_quote_ident(col)} FROM {source}")[1]


# --- Схема -------------------------------------------------------------------

def columns(con: duckdb.DuckDBPyConnection, source: str) -> list[dict[str, Any]]:
    """Список колонок источника с типами и их грубой классификацией."""
    rows = con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    # DESCRIBE: column_name, column_type, null, key, default, extra
    return [
        {"column": name, "dtype": dtype, "kind": _kind(dtype)}
        for name, dtype, *_ in rows
    ]


def show_columns(con: duckdb.DuckDBPyConnection, source: str) -> pd.DataFrame:
    info = columns(con, source)
    # Конструктор с явным index вместо set_index: у set_index в pandas
    # параметр keys без аннотации, и strict-pyright считает его Unknown
    return pd.DataFrame(
        [{k: v for k, v in c.items() if k != "column"} for c in info],
        index=[c["column"] for c in info],
    )


# --- Числовые колонки --------------------------------------------------------

def num_stats(con: duckdb.DuckDBPyConnection, source: str, col: str) -> dict[str, Any]:
    """Полнота, дубликаты, диапазон и непрерывность числовой колонки."""
    dtype = _column_type(con, source, col)
    kind = _kind(dtype)
    if kind not in ("int", "float", "decimal"):
        raise TypeError(f"Колонка {col!r} не числовая: {dtype}")

    # У целых типов дробной части быть не может, лишняя проверка не нужна
    frac_expr = "false" if kind == "int" else "coalesce(bool_or(c <> floor(c)), false)"

    total, not_null, distinct, mn, mx, has_fraction = _fetch_row(con, f"""
        SELECT count(*), count(c), count(DISTINCT c), min(c), max(c), {frac_expr}
        FROM (SELECT {_quote_ident(col)} AS c FROM {source})
    """)

    gaps = continuous = None
    finite = not_null > 0 and (
        kind != "float" or (math.isfinite(mn) and math.isfinite(mx))
    )
    if finite and not has_fraction:
        # Считаем в Python: int без переполнения, в отличие от BIGINT в SQL
        span = int(mx) - int(mn) + 1
        gaps = span - distinct
        continuous = gaps == 0

    return {
        "column": col,
        "dtype": dtype,
        "total": total,
        "not_null": not_null,
        "nulls": total - not_null,
        "duplicates": not_null - distinct,
        "min": mn,
        "max": mx,
        "has_fraction": bool(has_fraction),
        "continuous": continuous,
        "missing_in_range": gaps,
    }


def show_num_stats(con: duckdb.DuckDBPyConnection, source: str, col: str) -> pd.DataFrame:
    return pd.DataFrame(num_stats(con, source, col), index=[col])


# --- Колонки с датами -------------------------------------------------------

def date_stats(con: duckdb.DuckDBPyConnection, source: str, col: str) -> dict[str, Any]:
    """Полнота, дубликаты, диапазон и непрерывность колонки с датами.

    Непрерывность проверяется с шагом в один день: TIMESTAMP приводится к DATE
    (у TIMESTAMP WITH TIME ZONE — в часовом поясе сессии). Если пропусков
    несколько, возвращается первый по времени.
    """
    dtype = _column_type(con, source, col)
    if _kind(dtype) != "date":
        raise TypeError(f"Колонка {col!r} не дата: {dtype}")

    col_q = _quote_ident(col)

    # Полнота и дубликаты — по исходным значениям (для TIMESTAMP — с точностью до времени)
    total, not_null, distinct, mn, mx = _fetch_row(con, f"""
        SELECT count(*), count(c), count(DISTINCT c), min(c), max(c)
        FROM (SELECT {col_q} AS c FROM {source})
    """)

    # Непрерывность — по уникальным дням: шаг между соседними днями больше 1 = разрыв
    span, days, gaps, gap_start, gap_end = _fetch_row(con, f"""
        WITH days AS (
            SELECT DISTINCT CAST(c AS DATE) AS d
            FROM (SELECT {col_q} AS c FROM {source})
            WHERE c IS NOT NULL
        ),
        steps AS (
            SELECT lag(d) OVER (ORDER BY d) AS prev, d
            FROM days
        )
        SELECT
            date_diff('day', min(d), max(d)) + 1,
            count(*),
            count(*) FILTER (WHERE date_diff('day', prev, d) > 1),
            min(prev) FILTER (WHERE date_diff('day', prev, d) > 1) + 1,
            arg_min(d, prev) FILTER (WHERE date_diff('day', prev, d) > 1) - 1
        FROM steps
    """)

    has_data = not_null > 0
    return {
        "column": col,
        "dtype": dtype,
        "total": total,
        "not_null": not_null,
        "nulls": total - not_null,
        "duplicates": not_null - distinct,
        "min": mn,
        "max": mx,
        "span_days": span,
        "distinct_days": days,
        "continuous": gaps == 0 if has_data else None,
        "missing_in_range": span - days if has_data else None,
        "gaps": gaps if has_data else None,
        # Первый пропущенный интервал целиком: [first_gap_start, first_gap_end]
        "first_gap_start": gap_start,
        "first_gap_end": gap_end,
    }


def show_date_stats(con: duckdb.DuckDBPyConnection, source: str, col: str) -> pd.DataFrame:
    return pd.DataFrame(date_stats(con, source, col), index=[col])


# --- Категориальные колонки --------------------------------------------------

def cat_stats(con: duckdb.DuckDBPyConnection, source: str, col: str) -> dict[str, Any]:
    """Полнота, дубликаты и кардинальность категориальной колонки."""
    dtype = _column_type(con, source, col)
    if _kind(dtype) != "cat":
        raise TypeError(f"Колонка {col!r} не категориальная: {dtype}")

    total, not_null, distinct = _fetch_row(con, f"""
        SELECT count(*), count(c), count(DISTINCT c)
        FROM (SELECT {_quote_ident(col)} AS c FROM {source})
    """)

    return {
        "column": col,
        "dtype": dtype,
        "total": total,
        "not_null": not_null,
        "nulls": total - not_null,
        "duplicates": not_null - distinct,
        "cardinality": distinct,
        # Доля от not_null: NULL не считается категорией
        "cardinality_pct": 100 * distinct / not_null if not_null else None,
    }


def show_cat_stats(con: duckdb.DuckDBPyConnection, source: str, col: str) -> pd.DataFrame:
    df = pd.DataFrame(cat_stats(con, source, col), index=[col])
    # В аннотации pandas у round параметр Series без type args — со стороны не исправить
    return df.round({"cardinality_pct": 2})  # pyright: ignore[reportUnknownMemberType]
