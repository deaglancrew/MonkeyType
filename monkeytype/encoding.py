# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import abc
import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Type,
    TypeVar,
)

from mypy_extensions import TypedDict

from monkeytype.compat import is_any, is_union, is_generic, qualname_of_generic
from monkeytype.db.base import CallTraceThunk
from monkeytype.exceptions import InvalidTypeError
from monkeytype.tracing import CallTrace
from monkeytype.typing import NoneType, NotImplementedType, is_typed_dict, mappingproxy
from monkeytype.util import (
    get_func_in_module,
    get_name_in_module,
)
from types import new_class
logger = logging.getLogger(__name__)

TypeDict = Dict[str, Any]


class TypeTrace(abc.ABC):
    def __init__(self, typ, typename: Optional[str] = None):
        self.type = typ
        self.typename = typename or getattr(typ,  '__qualname__', repr(typ))

    @abc.abstractmethod
    def to_dict(self) -> TypeDict:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def from_dict(d: TypeDict):
        raise NotImplementedError


def _get_type_from_dict(d: TypeDict):
    if d['module'] == 'builtins' and d['qualname'] in _HIDDEN_BUILTIN_TYPES:
        return _HIDDEN_BUILTIN_TYPES[d['qualname']]
    output = get_name_in_module(d['module'], d['qualname'])
    if not (
            isinstance(output, type) or
            is_any(output) or
            is_generic(output)
    ):
        raise InvalidTypeError(
            f"Attribute specified by '{d['qualname']}' in module '{d['module']}' "
            f"is of type {type(output)}, not type."
        )
    return output


class SimpleTypeTrace(TypeTrace):
    def to_dict(self) -> TypeDict:
        return {'module': self.type.__module__,
                'qualname': self.typename}

    @staticmethod
    def from_dict(d: TypeDict):
        return _get_type_from_dict(d)


class ParameterizedTypeTrace(TypeTrace):
    def to_dict(self) -> TypeDict:
        elements = []
        elem_types = getattr(self.type, '__args__', None)
        if elem_types and is_generic(self.type):
            # empty typing.Tuple is weird; the spec says it should be Tuple[()],
            # which results in __args__ of `((),)`
            if elem_types == ((),):
                elem_types = ()
            elements = [type_to_dict(t) for t in elem_types]
        return {'module': self.type.__module__,
                'qualname': self.typename,
                'elem_types': elements}

    @staticmethod
    def from_dict(d: TypeDict):
        _type = _get_type_from_dict(d)
        return _type[tuple(type_from_dict(elem) for elem in d['elem_types'])]


def _get_baseclass(d: TypeDict):
    if d.get('is_typed_dict', False):
        return TypedDict
    if d.get('base_class', False):
        return get_name_in_module(**d['base_class'])
    return type

def _get_bases(d: TypeDict):
    if d.get('bases'):
        return tuple(get_name_in_module(**base) for base in d.get('bases'))
    return tuple()

class ClassicalTypeTrace(TypeTrace):
    def to_dict(self) -> TypeDict:
        output = {
            'module': self.type.__module__,
            'qualname': self.typename,
            'elem_types': {k: type_to_dict(v) for k, v in self.type.__annotations__.items()},
        }
        if is_typed_dict(self.type):
            output['is_typed_dict'] = True
        elif hasattr(self.type, '__orig_bases__'):
            base_type = self.type.__orig_bases__[0]
            output['base_class'] = {'module': base_type.__module__, 'qualname': base_type.__qualname__}
        elif hasattr(self.type, '__bases__'):
            bases = self.type.__bases__
            output['bases'] = [{'module': base.__module__, 'qualname': base.__qualname__} for base in bases]
        return output

    @staticmethod
    def from_dict(d: TypeDict):
        _class = _get_baseclass(d)
        if _class is not type:
            reconstruction = _class(d['qualname'],
                          **{k: type_from_dict(v)
                             for k, v in d['elem_types'].items()})
            if not hasattr(reconstruction, '__orig_bases__'):
                reconstruction.__orig_bases__ = (_class,)
        else:
            def add_annotations(ns):
                ns['__annotations__'] = {k: type_from_dict(v) for k, v in d['elem_types'].items()}
            reconstruction = new_class(d['qualname'], bases=_get_bases(d), exec_body=add_annotations)
        reconstruction.__module__ = None
        return reconstruction


# Types are converted to dictionaries of the following form before
# being JSON encoded and sent to storage:
#
#     {
#         'module': '<module>',
#         'qualname': '<qualname>',
#         'elem_types': [type_dict],
#     }
#
# The corresponding type alias should actually be
#
#     TypeDict = Dict[str, Union[str, TypeDict]]
#
# (or better, a TypedDict) but mypy does not support recursive type aliases:
#  https://github.com/python/mypy/issues/731


def _get_typetrace(typ: type, typename=None):
    if is_typed_dict(typ) or (hasattr(typ, '__annotations__') and typ.__module__ is None):
        return ClassicalTypeTrace(typ, typename=typename)
    if hasattr(typ, '__args__'):
        return ParameterizedTypeTrace(typ, typename=typename)
    return SimpleTypeTrace(typ, typename=typename)


def get_typetrace(typ: type):
    if is_union(typ):
        return _get_typetrace(typ, typename='Union')
    if is_generic(typ):
        return _get_typetrace(typ, typename=qualname_of_generic(typ))
    if is_any(typ):
        return _get_typetrace(typ, typename='Any')
    return _get_typetrace(typ)


def type_to_dict(typ: type) -> TypeDict:
    """Convert a type into a dictionary representation that we can store.

    The dictionary must:
        1. Be encodable as JSON
        2. Contain enough information to let us reify the type
    """
    return get_typetrace(typ).to_dict()


_HIDDEN_BUILTIN_TYPES: Dict[str, type] = {
    # Types that are inaccessible by their names in the builtins module.
    'NoneType': NoneType,
    'NotImplementedType': NotImplementedType,
    'mappingproxy': mappingproxy,
}


def type_from_dict(d: TypeDict) -> type:
    """Given a dictionary produced by type_to_dict, return the equivalent type.

    Raises:
        NameLookupError if we can't reify the specified type
        InvalidTypeError if the named type isn't actually a type
    """
    elem_types = d.get('elem_types', None)
    if isinstance(elem_types, dict):
        return ClassicalTypeTrace.from_dict(d)
    if isinstance(elem_types, list):
        return ParameterizedTypeTrace.from_dict(d)
    return SimpleTypeTrace.from_dict(d)


def type_to_json(typ: type) -> str:
    """Encode the supplied type as json using type_to_dict."""
    type_dict = type_to_dict(typ)
    return json.dumps(type_dict, sort_keys=True)


def type_from_json(typ_json: str) -> type:
    """Reify a type from the format produced by type_to_json."""
    type_dict = json.loads(typ_json)
    return type_from_dict(type_dict)


def arg_types_to_json(arg_types: Dict[str, type]) -> str:
    """Encode the supplied argument types as json"""
    type_dict = {name: type_to_dict(typ) for name, typ in arg_types.items()}
    return json.dumps(type_dict, sort_keys=True)


def arg_types_from_json(arg_types_json: str) -> Dict[str, type]:
    """Reify the encoded argument types from the format produced by arg_types_to_json."""
    arg_types = json.loads(arg_types_json)
    return {name: type_from_dict(type_dict) for name, type_dict in arg_types.items()}


TypeEncoder = Callable[[type], str]


def maybe_encode_type(encode: TypeEncoder, typ: Optional[type]) -> Optional[str]:
    if typ is None:
        return None
    return encode(typ)


TypeDecoder = Callable[[str], type]


def maybe_decode_type(decode: TypeDecoder, encoded: Optional[str]) -> Optional[type]:
    if (encoded is None) or (encoded == 'null'):
        return None
    return decode(encoded)


CallTraceRowT = TypeVar('CallTraceRowT', bound='CallTraceRow')


class CallTraceRow(CallTraceThunk):
    """A semi-structured call trace where each field has been json encoded."""

    def __init__(
        self,
        module: str,
        qualname: str,
        arg_types: str,
        return_type: Optional[str],
        yield_type: Optional[str]
    ) -> None:
        self.module = module
        self.qualname = qualname
        self.arg_types = arg_types
        self.return_type = return_type
        self.yield_type = yield_type

    @classmethod
    def from_trace(cls: Type[CallTraceRowT], trace: CallTrace) -> CallTraceRowT:
        module = trace.func.__module__
        qualname = trace.func.__qualname__
        arg_types = arg_types_to_json(trace.arg_types)
        return_type = maybe_encode_type(type_to_json, trace.return_type)
        yield_type = maybe_encode_type(type_to_json, trace.yield_type)
        return cls(module, qualname, arg_types, return_type, yield_type)

    def to_trace(self) -> CallTrace:
        function = get_func_in_module(self.module, self.qualname)
        arg_types = arg_types_from_json(self.arg_types)
        return_type = maybe_decode_type(type_from_json, self.return_type)
        yield_type = maybe_decode_type(type_from_json, self.yield_type)
        return CallTrace(function, arg_types, return_type, yield_type)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CallTraceRow):
            return (
                self.module,
                self.qualname,
                self.arg_types,
                self.return_type,
                self.yield_type,
            ) == (
                other.module,
                other.qualname,
                other.arg_types,
                other.return_type,
                other.yield_type,
            )
        return NotImplemented


def serialize_traces(traces: Iterable[CallTrace]) -> Iterable[CallTraceRow]:
    """Serialize an iterable of CallTraces to an iterable of CallTraceRow.

    Catches and logs exceptions, so a failure to serialize one CallTrace doesn't
    lose all traces.

    """
    for trace in traces:
        try:
            yield CallTraceRow.from_trace(trace)
        except Exception:
            logger.exception(f"Failed to serialize trace for {trace.func}")
