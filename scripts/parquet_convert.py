"""Конвертация CSV из data/landing в слой bronze (data/bronze) в формате parquet.

Принципы:
- типы колонок задаются явно в `TABLES` (источник — data_dictionary.md),
  автоопределение типов DuckDB не используется;
- bronze хранит данные как есть: значения только приводятся к типам,
  без заполнения NULL, фильтрации и прочей очистки (это задача silver);
- строки сортируются по ключу таблицы: результат детерминирован, а статистики
  min/max в row group позволяют пропускать ненужные части файла при фильтре по дате;
- запись идёт во временный файл, который переименовывается только после успеха,
  поэтому после сбоя не остаётся полузаписанного parquet.

Все пути относительны корня проекта (в ноутбуках он задан jupyter.notebookFileRoot).
"""

import contextlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

__all__ = [
    "BRONZE_DIR",
    "LANDING_DIR",
    "TABLES",
    "Check",
    "TableSpec",
    "VerificationReport",
    "convert_all",
    "convert_table",
    "verify_all",
    "verify_table",
]

LANDING_DIR = Path("data/landing")
BRONZE_DIR = Path("data/bronze")

_COMPRESSIONS = {"zstd", "snappy", "gzip", "uncompressed"}


@dataclass(frozen=True)
class TableSpec:
    """Схема таблицы bronze.

    columns: имя колонки -> тип DuckDB; порядок обязан совпадать с заголовком CSV.
    sort_by: ключ сортировки; колонки без NULL, лучше уникальный ключ —
             тогда порядок строк полностью детерминирован.
    """

    columns: Mapping[str, str]
    sort_by: tuple[str, ...]


# Типы — из колонки «Тип в bronze» data_dictionary.md.
# Все они переносимы в Spark: SMALLINT -> ShortType, INTEGER -> IntegerType,
# DOUBLE -> DoubleType, BOOLEAN -> BooleanType, DATE -> DateType, VARCHAR -> StringType.
TABLES: dict[str, TableSpec] = {
    "train": TableSpec(
        columns={
            "id": "INTEGER",
            "date": "DATE",
            "store_nbr": "SMALLINT",
            "item_nbr": "INTEGER",
            "unit_sales": "DOUBLE",
            "onpromotion": "BOOLEAN",
        },
        sort_by=("date", "store_nbr", "item_nbr"),
    ),
    "test": TableSpec(
        columns={
            "id": "INTEGER",
            "date": "DATE",
            "store_nbr": "SMALLINT",
            "item_nbr": "INTEGER",
            "onpromotion": "BOOLEAN",
        },
        sort_by=("date", "store_nbr", "item_nbr"),
    ),
    "stores": TableSpec(
        columns={
            "store_nbr": "SMALLINT",
            "city": "VARCHAR",
            "state": "VARCHAR",
            "type": "VARCHAR",
            "cluster": "SMALLINT",
        },
        sort_by=("store_nbr",),
    ),
    "items": TableSpec(
        columns={
            "item_nbr": "INTEGER",
            "family": "VARCHAR",
            "class": "SMALLINT",
            "perishable": "BOOLEAN",  # в CSV 0/1, DuckDB приводит '0'/'1' к false/true
        },
        sort_by=("item_nbr",),
    ),
    "transactions": TableSpec(
        columns={
            "date": "DATE",
            "store_nbr": "SMALLINT",
            "transactions": "INTEGER",
        },
        sort_by=("date", "store_nbr"),
    ),
    "oil": TableSpec(
        columns={
            "date": "DATE",
            "dcoilwtico": "DOUBLE",
        },
        sort_by=("date",),
    ),
    "holidays_events": TableSpec(
        columns={
            "date": "DATE",
            "type": "VARCHAR",
            "locale": "VARCHAR",
            "locale_name": "VARCHAR",
            "description": "VARCHAR",
            "transferred": "BOOLEAN",
        },
        # Естественный ключ не определён, поэтому сортировка по всем колонкам:
        # иначе порядок строк с одинаковой датой не детерминирован.
        sort_by=("date", "type", "locale", "locale_name", "description", "transferred"),
    ),
}


# --- Служебные функции -------------------------------------------------------


def _quote_ident(name: str) -> str:
    """Экранирует идентификатор для SQL."""
    return '"' + name.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    """Экранирует строковый литерал для SQL."""
    return "'" + value.replace("'", "''") + "'"


def _spec(table: str) -> TableSpec:
    try:
        return TABLES[table]
    except KeyError:
        raise KeyError(f"Нет схемы для таблицы {table!r}. Известные: {sorted(TABLES)}") from None


def _csv_path(table: str, landing_dir: Path) -> Path:
    return landing_dir / f"{table}.csv"


def _parquet_path(table: str, bronze_dir: Path) -> Path:
    return bronze_dir / f"{table}.parquet"


def _check_header(path: Path, spec: TableSpec) -> None:
    """Сверяет заголовок CSV со схемой.

    При auto_detect = false DuckDB просто пропускает строку заголовка и раскладывает
    значения по позиции. Если порядок колонок в файле изменится, данные молча
    окажутся не в тех колонках, поэтому заголовок проверяется явно.
    """
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    expected = list(spec.columns)
    if header != expected:
        raise ValueError(f"Заголовок {path} не совпадает со схемой: {header} != {expected}")


def _read_csv_sql(path: Path, columns: Mapping[str, str]) -> str:
    """SQL-фрагмент read_csv с фиксированным диалектом и явными типами."""
    struct = ", ".join(f"{_quote_literal(n)}: {_quote_literal(t)}" for n, t in columns.items())
    return (
        f"read_csv({_quote_literal(path.as_posix())}, "
        "header = true, auto_detect = false, "
        "delim = ',', quote = '\"', escape = '\"', "
        "dateformat = '%Y-%m-%d', "
        f"columns = {{{struct}}})"
    )


def _read_parquet_sql(path: Path, *, file_row_number: bool = False) -> str:
    extra = ", file_row_number = true" if file_row_number else ""
    return f"read_parquet({_quote_literal(path.as_posix())}{extra})"


def _fetch_row(con: duckdb.DuckDBPyConnection, query: str) -> tuple[Any, ...]:
    row = con.execute(query).fetchone()
    if row is None:
        raise RuntimeError(f"Запрос не вернул строк:\n{query}")
    return row


# --- Конвертация -------------------------------------------------------------


def convert_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    landing_dir: Path = LANDING_DIR,
    bronze_dir: Path = BRONZE_DIR,
    compression: str = "zstd",
) -> Path:
    """Конвертирует data/landing/<table>.csv в data/bronze/<table>.parquet.

    Повторный запуск перезаписывает результат. Если какое-то значение не приводится
    к объявленному типу, DuckDB падает с ошибкой — это намеренно: ошибка видна сразу,
    а не превращается в NULL.

    Для train сортировка 125 млн строк выходит за память, поэтому у соединения
    должны быть заданы memory_limit и temp_directory (DuckDB сбросит часть на диск).
    """
    if compression not in _COMPRESSIONS:
        raise ValueError(f"Неизвестное сжатие {compression!r}, допустимые: {sorted(_COMPRESSIONS)}")

    spec = _spec(table)
    source = _csv_path(table, landing_dir)
    _check_header(source, spec)

    bronze_dir.mkdir(parents=True, exist_ok=True)
    target = _parquet_path(table, bronze_dir)
    tmp = target.with_name(target.name + ".tmp")
    tmp.unlink(missing_ok=True)  # остаток прошлого неудачного запуска

    columns = ", ".join(_quote_ident(c) for c in spec.columns)
    order_by = ", ".join(_quote_ident(c) for c in spec.sort_by)
    query = f"""
        COPY (
            SELECT {columns}
            FROM {_read_csv_sql(source, spec.columns)}
            ORDER BY {order_by}
        ) TO {_quote_literal(tmp.as_posix())}
        (FORMAT parquet, COMPRESSION {_quote_literal(compression)})
    """
    try:
        con.execute(query)
        # Атомарная замена. На Windows упадёт, если target открыт другим процессом.
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    return target


def convert_all(
    con: duckdb.DuckDBPyConnection,
    tables: Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, Path]:
    """Конвертирует несколько таблиц (по умолчанию все из TABLES)."""
    return {t: convert_table(con, t, **kwargs) for t in (tables or TABLES)}


# --- Проверка ----------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """Одна проверка: ожидаемое значение (из CSV или схемы) против фактического (parquet)."""

    name: str
    expected: Any
    actual: Any

    @property
    def passed(self) -> bool:
        return bool(self.expected == self.actual)


@dataclass(frozen=True)
class VerificationReport:
    table: str
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "table": [self.table] * len(self.checks),
                "check": [c.name for c in self.checks],
                "expected": [str(c.expected) for c in self.checks],
                "actual": [str(c.actual) for c in self.checks],
                "passed": [c.passed for c in self.checks],
            }
        )

    def raise_if_failed(self) -> None:
        if not self.ok:
            lines = [
                f"  {c.name}: ожидалось {c.expected}, получено {c.actual}" for c in self.failed()
            ]
            raise AssertionError(f"Проверка {self.table} не пройдена:\n" + "\n".join(lines))


def _counts_and_nulls(
    con: duckdb.DuckDBPyConnection, source: str, columns: Iterable[str]
) -> tuple[int, list[int]]:
    """Число строк и число NULL по каждой колонке за один проход."""
    cols = list(columns)
    nulls = ", ".join(f"count(*) - count({_quote_ident(c)})" for c in cols)
    row = _fetch_row(con, f"SELECT count(*), {nulls} FROM {source}")
    return int(row[0]), [int(v) for v in row[1:]]


def _content_hash(
    con: duckdb.DuckDBPyConnection, source: str, columns: Iterable[str]
) -> int | None:
    """Отпечаток содержимого, не зависящий от порядка строк: сумма хешей строк.

    Сумма, а не XOR: при XOR одинаковые строки взаимно гасятся, и потеря
    дубликата не была бы замечена.
    """
    cols = ", ".join(_quote_ident(c) for c in columns)
    value = _fetch_row(con, f"SELECT sum(hash({cols})::HUGEINT) FROM {source}")[0]
    return None if value is None else int(value)


def _sort_violations(con: duckdb.DuckDBPyConnection, path: Path, sort_by: tuple[str, ...]) -> int:
    """Число строк parquet, которые меньше предыдущей по ключу сортировки."""
    keys = [_quote_ident(k) for k in sort_by]
    prev = [_quote_ident(f"_prev_{k}") for k in sort_by]
    lags = ", ".join(f"lag({k}) OVER w AS {p}" for k, p in zip(keys, prev, strict=True))
    # Лексикографическое «строка < предыдущей»:
    # k1 < p1 OR (k1 = p1 AND k2 < p2) OR ...
    terms: list[str] = []
    for i in range(len(keys)):
        eq = [f"{keys[j]} = {prev[j]}" for j in range(i)]
        terms.append("(" + " AND ".join([*eq, f"{keys[i]} < {prev[i]}"]) + ")")
    query = f"""
        SELECT count(*) FROM (
            SELECT {", ".join(keys)}, {lags}
            FROM {_read_parquet_sql(path, file_row_number=True)}
            WINDOW w AS (ORDER BY file_row_number)
        )
        WHERE {" OR ".join(terms)}
    """
    return int(_fetch_row(con, query)[0])


def verify_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    *,
    landing_dir: Path = LANDING_DIR,
    bronze_dir: Path = BRONZE_DIR,
    check_sort: bool = True,
) -> VerificationReport:
    """Проверяет, что parquet совпадает с исходным CSV.

    Проверки:
    - schema: имена, порядок и типы колонок parquet совпадают с TABLES;
    - rows: число строк parquet = числу строк CSV;
    - nulls.<col>: число NULL совпадает с числом пустых значений в сыром CSV
      (CSV читается как VARCHAR, без приведения типов). Так видно, если приведение
      типа превратило значение в NULL;
    - content_hash: отпечаток содержимого parquet = отпечатку CSV, прочитанного
      с теми же типами. Ловит потерю, дублирование и искажение строк при записи;
    - sorted: строки parquet идут в порядке sort_by (0 нарушений). Для train это
      отдельный проход с оконной функцией, его можно отключить check_sort=False.

    Проверки не оценивают сами типы по смыслу (например, что 'True' стало true):
    это гарантирует строгое приведение типов при чтении CSV.
    """
    spec = _spec(table)
    csv_path = _csv_path(table, landing_dir)
    parquet_path = _parquet_path(table, bronze_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} не найден — сначала convert_table")

    raw_csv = _read_csv_sql(csv_path, dict.fromkeys(spec.columns, "VARCHAR"))
    typed_csv = _read_csv_sql(csv_path, spec.columns)
    parquet = _read_parquet_sql(parquet_path)
    cols = list(spec.columns)

    checks: list[Check] = []

    schema_rows = con.execute(f"DESCRIBE SELECT * FROM {parquet}").fetchall()
    actual_schema = [(str(r[0]), str(r[1])) for r in schema_rows]
    checks.append(Check("schema", list(spec.columns.items()), actual_schema))

    csv_rows, csv_nulls = _counts_and_nulls(con, raw_csv, cols)
    pq_rows, pq_nulls = _counts_and_nulls(con, parquet, cols)
    checks.append(Check("rows", csv_rows, pq_rows))
    checks.extend(
        Check(f"nulls.{c}", e, a) for c, e, a in zip(cols, csv_nulls, pq_nulls, strict=True)
    )

    checks.append(
        Check(
            "content_hash",
            _content_hash(con, typed_csv, cols),
            _content_hash(con, parquet, cols),
        )
    )

    if check_sort:
        checks.append(Check("sorted", 0, _sort_violations(con, parquet_path, spec.sort_by)))

    return VerificationReport(table, tuple(checks))


def verify_all(
    con: duckdb.DuckDBPyConnection,
    tables: Iterable[str] | None = None,
    **kwargs: Any,
) -> list[VerificationReport]:
    """Проверяет несколько таблиц (по умолчанию все из TABLES)."""
    return [verify_table(con, t, **kwargs) for t in (tables or TABLES)]
