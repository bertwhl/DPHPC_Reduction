import dace
from util import *

H = dace.symbol('H')
W = dace.symbol('W')
X = dace.symbol('X')

#B1-A1-B2 -> A1
@dace.program
def RowReduceNarrow(inputs: dace.float64[X, H, W], gridDim_y: dace.int64, gridDim_x: dace.int64, blockDim_x: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_y, 0:gridDim_x]:
        for threadIdx_x in dace.map[0:blockDim_x]:
            value = dace.float64(0)
            rowIdx = blockIdx_x*blockDim_x+threadIdx_x
            if rowIdx < H:
                for colIdx in dace.map[0:W]:
                    value += inputs[blockIdx_y,rowIdx,colIdx]
                outputs[rowIdx] += value
    return outputs

if __name__ == '__main__':
    import numpy as np
    from dace.transformation.interstate import GPUTransformSDFG
    # Transform to GPU, keep thread-block map
    sdfg = RowReduceNarrow.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    x = 12
    h = 1000
    w = 12

    # Test
    inputs = np.random.rand(x, h, w)
    outputs = sdfg(X=x, H=h, W=w, inputs=inputs, gridDim_y=12, gridDim_x=4, blockDim_x=256)
    compared = np.sum(inputs, axis=(0,2))
    assert np.allclose(outputs, compared)