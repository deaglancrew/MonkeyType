import datetime
from string import ascii_lowercase
from types import new_class
from typing import List, Iterable, NamedTuple, Tuple, Union

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series
from pandera.schemas import SeriesSchema, DataFrameSchema
from monkeytype.config import DefaultConfig
from monkeytype.typing import TypeHook


def _gen_attribute_names():
    count = 0
    while True:
        for letter in ascii_lowercase:
            yield f"{letter}{count if count else ''}"
        count += 1


DUMMY_TUPLE_NAME = 'DUMMY_NAMED_TUPLE'

DUMMY_MODEL_NAME = 'DUMMY_PANDERA_MODEL'


def basic_type(type_name):
    for name, t in [('int', int), ('float', float), ('complex', complex), ('datetime', datetime.datetime),
                    ('timedelta', datetime.timedelta), ('object', object)]:
        if name in type_name.lower():
            return t


def _get_series(column):
    new_series = pa.model.Series[column.dtype.type.type]
    new_series._name = None
    return new_series


def _schema_to_model(schema: Union[DataFrameSchema, SeriesSchema]):
    if isinstance(schema, SeriesSchema):
        return _series_schema_to_type(schema)
    annotations = {column.name: _get_series(column) for column in schema.columns.values()}

    def update_namespace(ns):
        ns['__annotations__'] = annotations

    model = new_class(DUMMY_MODEL_NAME,
                      bases=(pa.SchemaModel,),
                      exec_body=update_namespace)
    model.__module__ = None
    return DataFrame[model]


def _series_schema_to_type(schema: pa.schemas.SeriesSchema):
    return Series[schema.dtype.type.type]


class PanderaDataFrame(TypeHook):
    def handles(self, typ) -> bool:
        return isinstance(typ, pd.DataFrame) or isinstance(typ, pd.Series)

    def convert_object(self, obj):
        schema = pa.infer_schema(obj)
        return _schema_to_model(schema)

    def can_shrink(self, types: Iterable) -> bool:
        return False

    def shrink_types(self, types: list):
        return types


class DummyNamedTuple(TypeHook):
    def handles(self, typ) -> bool:
        return isinstance(typ, Tuple)

    def convert_object(self, obj):
        fields = {name: self._getter.get_type(value) for name, value in zip(_gen_attribute_names(), obj)}
        result = NamedTuple(DUMMY_TUPLE_NAME, **fields)
        result.__orig_bases__ = (NamedTuple,)
        result.__module__ = None
        return result

    def can_shrink(self, types: Iterable) -> bool:
        return False

    def shrink_types(self, types: list):
        return types


class WithNamedTuple(DefaultConfig):
    def type_hooks(self) -> List[TypeHook]:
        return super().type_hooks() + [DummyNamedTuple(), PanderaDataFrame()]


CONFIG = WithNamedTuple()
