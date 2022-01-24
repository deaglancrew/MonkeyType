from types import new_class

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from monkeytype.typing import TypeHook

DUMMY_MODEL_NAME = "DUMMY_PANDERA_MODEL"


from pandera.typing import Series, Index
from pandera.engines import numpy_engine
import builtins


def get_column_type(column, example=None):
    if isinstance(column.dtype, numpy_engine.Object) and example is not None:
        numpy_engine_result = numpy_engine.Engine.dtype(example.__class__)
        if hasattr(builtins, str(numpy_engine_result)):
            return getattr(builtins, str(numpy_engine_result))
        return numpy_engine_result
    return column.dtype.__class__


def get_indices(df):
    df_schema = pa.infer_schema(df)
    index = df_schema.index
    if hasattr(index, 'indexes'):
        yield from index.indexes
    else:
        yield index


def convert_indices_to_annotations(df):
    for level, index in enumerate(get_indices(df)):
        index_type = get_column_type(index, example=df.index[0][level])
        name = f"INDEX_{level}_{index.name or 'idx'}"
        yield name, index_type


def get_index_annotations(df):
    return {name: Index[index_type] for name, index_type in convert_indices_to_annotations(df)}


def df_to_model(df):
    annotations = {}
    for column_name in df:
        annotations[column_name] = series_to_type(df[column_name])

    annotations.update(get_index_annotations(df))

    def update_namespace(ns):
        ns["__annotations__"] = annotations

    model = new_class(DUMMY_MODEL_NAME, bases=(pa.SchemaModel,), exec_body=update_namespace)
    model.__module__ = None
    return DataFrame[model]


def series_to_type(series):
    series_schema = pa.infer_schema(series)
    example = None
    if len(series) > 0:
        example = series[0]
    output = Series[get_column_type(series_schema, example)]
    return output


class PanderaDataFrame(TypeHook):
    def handles(self, typ) -> bool:
        return isinstance(typ, pd.DataFrame) or isinstance(typ, pd.Series)

    def convert_object(self, obj):
        if isinstance(obj, pd.DataFrame):
            print(obj)
            return df_to_model(obj)
        return series_to_type(obj)

