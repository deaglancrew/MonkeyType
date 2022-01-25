from monkeytype.encoding import SimpleTypeTrace
from pandera.engines.numpy_engine import DataType


def test_can_serialise_numpy_datatype():
    assert SimpleTypeTrace(DataType('float64')).to_dict() == {'module': 'numpy', 'qualname': 'float64'}
