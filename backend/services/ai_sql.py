import logging
import sqlite3
import time
from contextlib import closing
from math import isfinite
from pathlib import Path

from backend.core.database import DATABASE_PATH

logger = logging.getLogger(__name__)
MAX_ROWS = 100
ALLOWED_FUNCTIONS = {
    "abs", "avg", "coalesce", "count", "ifnull", "length", "like", "lower",
    "max", "min", "nullif", "round", "substr", "substring", "sum", "total",
    "trim", "ltrim", "rtrim", "upper",
}
COLUMNS = {"item_id", "category", "item_name", "price", "stock_quantity", "supplier"}
SCHEMA = (
    "SQLite table inventory(item_id INTEGER PRIMARY KEY, category TEXT, item_name TEXT, "
    "price REAL NULL, stock_quantity INTEGER, supplier TEXT). Price is unit price; "
    "inventory value is price * stock_quantity. Low stock is 1-49, zero is out of "
    "stock, negative stock is a data issue. No sales/history/cost/timestamp fields exist."
)


class SQLQueryError(ValueError):
    pass


def validate(sql: str) -> str:
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    if len(sql) > 12000:
        raise SQLQueryError("SQL exceeds the query length limit")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (SqlglotError, RecursionError) as exc:
        raise SQLQueryError("The agent generated invalid SQLite SQL; please rephrase") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise SQLQueryError("Only one read-only SELECT query is allowed")
    query = statements[0]
    if any(query.find_all(exp.With, exp.Join, exp.Into, exp.Lock, exp.Union, exp.Intersect, exp.Except)):
        raise SQLQueryError("Use a single-table SELECT; joins, CTEs and set operations are not supported")
    tables = list(query.find_all(exp.Table))
    if not tables or any(
        table.name.lower() != "inventory" or table.db or table.catalog for table in tables
    ):
        raise SQLQueryError("SQL may read only the inventory table")
    return query.sql(dialect="sqlite", comments=False)


def authorize(operation: int, arg1: str | None, arg2: str | None, database: str | None, source: str | None) -> int:
    if operation == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    # SQLite may omit the database name for an optimized column-free row read.
    if (operation == sqlite3.SQLITE_READ and arg1 == "inventory"
            and ((database == "main" and (not arg2 or arg2 in COLUMNS))
                 or (database is None and arg2 == ""))):
        return sqlite3.SQLITE_OK
    if operation == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ALLOWED_FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def result_item_ids(data: dict) -> list[int]:
    import sqlglot
    from sqlglot import exp

    query = sqlglot.parse_one(data["sql"], read="sqlite")
    source = query.args.get("from_")
    if (data["truncated"] or query.args.get("group") or source is None
            or not isinstance(source.this, exp.Table)
            or any(column.find(exp.AggFunc, exp.Window) for column in query.expressions)):
        return []
    indexes = []
    for index, projection in enumerate(query.expressions):
        column = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(column, exp.Star) or (isinstance(column, exp.Column) and column.is_star):
            if len(query.expressions) == 1 and data["columns"].count("item_id") == 1:
                indexes.append(data["columns"].index("item_id"))
        elif isinstance(column, exp.Column) and column.name.lower() == "item_id":
            if not any(item.is_star for item in query.expressions):
                indexes.append(index)
    if len(indexes) != 1:
        return []
    values = {row[indexes[0]] for row in data["rows"]}
    if not values or len(values) > 25 or any(type(value) is not int or value <= 0 for value in values):
        return []
    return sorted(values)


def execute(sql: str, path: Path | None = None) -> dict:
    path = path or DATABASE_PATH
    validated = validate(sql)
    started = time.monotonic()
    ticks = 0

    def progress() -> int:
        nonlocal ticks
        ticks += 1
        return int(ticks >= 1000 or time.monotonic() - started > 1)

    try:
        # Independent connection: read-only at the filesystem and SQLite authorization layers.
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            if not hasattr(connection, "setlimit"):
                raise SQLQueryError("Text-to-SQL requires Python 3.11 or newer for SQLite resource limits")
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 100_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 24_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 50)
            connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(authorize)
            connection.set_progress_handler(progress, 1000)
            cursor = connection.execute(f"SELECT * FROM ({validated}) AS agent_result LIMIT {MAX_ROWS + 1}")
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
        if any(
            isinstance(cell, bytes) or (isinstance(cell, float) and not isfinite(cell))
            for row in rows for cell in row
        ):
            raise SQLQueryError("SQL results must be finite numbers, text or null; binary data is unsupported")
        return {
            "sql": validated, "columns": columns, "rows": [list(row) for row in rows[:MAX_ROWS]],
            "row_count": min(len(rows), MAX_ROWS), "truncated": len(rows) > MAX_ROWS,
            "max_rows": MAX_ROWS,
        }
    except sqlite3.Error as exc:
        logger.warning("Read-only AI SQL failed: %s", type(exc).__name__)
        raise SQLQueryError(
            "SQL execution was rejected or exceeded a resource limit. Only bounded inventory "
            "SELECT queries with approved functions are supported; please rephrase."
        ) from exc
