import numpy as np
from tiny_transformer.main import encode, decode


def test_encode_decode_roundtrip_nonempty():
    s = "hello world"
    arr = encode(s)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    out = decode(arr)
    assert isinstance(out, str)
    assert "hello" in out and "world" in out
