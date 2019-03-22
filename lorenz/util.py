
import numpy as np
from collections import namedtuple
from numpy.lib.stride_tricks import as_strided

_Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Transition(_Transition):

    @classmethod
    def transpose(cls, trans):
        if isinstance(trans, cls):
            return list(map(lambda x: cls(*x), zip(*trans)))
        elif isinstance(trans, list):
            # Transpose the batch (see http://stackoverflow.com/a/19343/3343043
            # for detailed explanation).
            return cls(*zip(*trans))
        else:
            raise TypeError("`trans` needs to be either `Transition` or `list`")


_bytes_per_dtype = {
    np.dtype('float32'): 4,
    np.dtype('float64'): 8,
}

def unfold(x: np.ndarray, dim: int, size: int, step: int) -> np.ndarray:
    """returns consecutive sequences of array `x`

    dim : dimension that indexes across sequences
    size : length of the sequence
    step : step between sequence
    """
    b = _bytes_per_dtype[x.dtype]
    # compute shape
    assert dim == 0
    shape = ((1+x.size-size)//step, size)
    # compute strides
    strides = (step*b, step*b)
    return as_strided(x, shape, strides)
