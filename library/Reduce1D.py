import dace
from util import *

W = dace.symbol('W')

@dace.program
def Reduce(inputs: dace.float64[W], gridDim_x: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([1], dtype=dace.float64)
    outputs[0] = 0
    # blocks mapping
    for blockIdx_x in dace.map[0:gridDim_x]:
        # thread mapping
        for threadIdx_y, threadIdx_x in dace.map[0:16, 0:32]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexes
            Idx = 512*blockIdx_x + 32*threadIdx_y + threadIdx_x
            delta = 512*gridDim_x
            # sum up the values
            for loopIdx in dace.map[0:loopNum]:
                if Idx<W:
                    value += inputs[Idx]
                Idx += delta
                # write back to global memory with atomic add
            reduced = warpReduce_sum(value)
            if threadIdx_x == 0:
                outputs[0] += reduced
    return outputs

if __name__ == '__main__':
    import numpy as np
    from dace.transformation.interstate import GPUTransformSDFG
    # Transform to GPU, keep thread-block map
    sdfg = Reduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    w = 1024*1024*128
    BlockNum = 256
    WarpPerBlock = 16
    # other params
    loopNum = (w+512*BlockNum-1)//(512*BlockNum)

    # Test
    inputs = np.random.rand(w)
    outputs = sdfg(W=w, inputs=inputs, gridDim_x=BlockNum, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)