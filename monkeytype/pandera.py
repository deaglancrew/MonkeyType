from types import new_class
from typing import Union

import pandas as pd
import pandera as pa
from numpy import object_
from pandera.schemas import SeriesSchema, DataFrameSchema
from pandera.typing import DataFrame, Series

from monkeytype.typing import TypeHook

DUMMY_MODEL_NAME = "DUMMY_PANDERA_MODEL"


def _get_series(column):
    new_series = _series_schema_to_type(column)
    new_series._name = None
    return new_series


def _schema_to_model(schema: Union[DataFrameSchema, SeriesSchema]):
    if isinstance(schema, SeriesSchema):
        return _series_schema_to_type(schema)
    annotations = {column.name: _get_series(column) for column in schema.columns.values()}

    def update_namespace(ns):
        ns["__annotations__"] = annotations

    model = new_class(DUMMY_MODEL_NAME, bases=(pa.SchemaModel,), exec_body=update_namespace)
    model.__module__ = None
    return DataFrame[model]


def _series_schema_to_type(schema: pa.schemas.SeriesSchema):
    try:
        return Series[schema.dtype.type.type]
    except TypeError:
        return Series[object_]


class PanderaDataFrame(TypeHook):
    def handles(self, typ) -> bool:
        return isinstance(typ, pd.DataFrame) or isinstance(typ, pd.Series)

    def convert_object(self, obj):
        schema = pa.infer_schema(obj)
        return _schema_to_model(schema)
