import dace
import numpy as np
from util import *
from dace.transformation.interstate import GPUTransformSDFG
from dace.frontend.common import op_repository as oprepo

X = dace.symbol('X')
H = dace.symbol('H')
W = dace.symbol('W')

#A1-B1-A2 -> A1-A2
#using more blocks
@dace.program
def ColReduce(inputs: dace.float64[X, H, W], gridDim_x: dace.int64, gridDim_y: dace.int64, blockDim_x: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([X,W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_z, blockIdx_y, blockIdx_x in dace.map[0:X, 0:gridDim_y, 0:gridDim_x]:
        # thread mapping
        for threadIdx_x in dace.map[0:blockDim_x]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexes
            levelIdx = blockIdx_z
            rowIdx = blockIdx_y
            colIdx = blockDim_x * blockIdx_x + threadIdx_x
            # sum up the values
            if colIdx<W:
                delta = gridDim_y
                for loopIdx in dace.map[0:loopNum]:
                    if rowIdx<H:
                        value += inputs[levelIdx, rowIdx, colIdx]
                    rowIdx += delta
                # write back to global memory with atomic add
                outputs[levelIdx,colIdx] += value
    return outputs


# use warp read warp reduce for narrow corner cases
@dace.program
def ColReduceNarrow(inputs: dace.float64[X, H, W], gridDim_x: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([X,W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_y, blockIdx_x in dace.map[0:X, 0:gridDim_x]:
        shared = dace.ndarray([1024], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        for threadIdx_y, threadIdx_x in dace.map[0:W, 0:32]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexs
            threadIdx = 32*threadIdx_y + threadIdx_x
            colIdx = threadIdx%W
            rowIdx_offset = dace.int64(threadIdx/W)
            rowIdx = 32*blockIdx_x + rowIdx_offset
            rowIdx_delta = 32*gridDim_x
            for loopIdx in dace.map[0:loopNum]:
                # add the value
                if rowIdx<H:
                    value += inputs[blockIdx_y, rowIdx, colIdx]
                rowIdx += rowIdx_delta
            # write the value into shared memory
            shared[W*rowIdx_offset + colIdx] = value
        # synchronize here
        for threadIdx_y, threadIdx_x in dace.map[0:W, 0:32]:
            rowIdx = threadIdx_x
            colIdx = threadIdx_y
            # warp reduce
            reduced = warpReduce_sum(shared[W*rowIdx + colIdx])
            # write back to global memory with atomic add
            if threadIdx_x==0:
                outputs[blockIdx_y,colIdx] += reduced
    return outputs

test = None
if test == 'normal':
    sdfg = ColReduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    x = 256
    h = 8192
    w = 30
    gridDim_y = 8
    loopNum = int(np.ceil(h/gridDim_y))

    # Test
    inputs = np.random.rand(x, h, w)
    for i in range(4):
        outputs = sdfg(X=x, H=h, W=w, inputs=inputs, gridDim_x=1, gridDim_y=gridDim_y, blockDim_x=w, loopNum=loopNum)
    compared = np.sum(inputs, axis=1)
    assert np.allclose(outputs, compared)
if test == 'narrow':
    sdfg = ColReduceNarrow.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    x = 256
    h = 8192
    w = 30
    gridDim_x = 1
    loopNum = int(np.ceil(h/gridDim_x/32))

    # Test
    inputs = np.random.rand(x, h, w)
    for i in range(4):
        outputs = sdfg(X=x, H=h, W=w, inputs=inputs, gridDim_x=gridDim_x, loopNum=loopNum)
    compared = np.sum(inputs, axis=1)
    assert np.allclose(outputs, compared)
