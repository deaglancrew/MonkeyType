import psycopg2
from monkeytype.db.sqlite import DBApi2Store, QueryValue, ParameterizedQuery
import datetime
from typing import (
    Iterable,
    List,
    Optional,
)

from monkeytype.db.base import (
    CallTraceThunk,
)
from monkeytype.encoding import (
    CallTraceRow,
    serialize_traces,
)
from monkeytype.tracing import CallTrace


def make_query(table: str, module: str, qualname: Optional[str], limit: int) -> ParameterizedQuery:
    raw_query = """
    SELECT
        module, qualname, arg_types, return_type, yield_type
    FROM {table}
    WHERE
        module LIKE %s
    """.format(table=table)
    values: List[QueryValue] = [module]
    if qualname is not None:
        raw_query += " AND qualname LIKE %s"
        values.append(f"{qualname}%")
    raw_query += """
    GROUP BY
        module, qualname, arg_types, return_type, yield_type
    LIMIT %s
    """
    values.append(limit)
    return raw_query, values


class PsycopgStore(DBApi2Store):

    def add(self, traces: Iterable[CallTrace]) -> None:
        values = []
        for row in serialize_traces(traces):
            values.append((datetime.datetime.now(), row.module, row.qualname,
                           row.arg_types, row.return_type, row.yield_type))
        with self.conn:
            cur = self.conn.cursor()
            cur.executemany(
                'INSERT INTO {table} VALUES (%s, %s, %s, %s, %s, %s)'.format(table=self.table),
                values
            )

    def filter(
            self,
            module: str,
            qualname_prefix: Optional[str] = None,
            limit: int = 2000
    ) -> List[CallTraceThunk]:
        sql_query, values = make_query(self.table, module, qualname_prefix, limit)
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(sql_query, values)
            return [CallTraceRow(*row) for row in cur.fetchall()]

    @staticmethod
    def make_connection(connection_string):
        return psycopg2.connect(connection_string)
