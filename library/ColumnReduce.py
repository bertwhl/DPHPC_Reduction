import numpy as np
import dace
from dace.transformation.interstate import GPUTransformSDFG, StateFusion
from util import *

H = dace.symbol('H')
W = dace.symbol('W')

@dace.program
def WarpReadWarpReduce(inputs: dace.float64[H, W], gridDim_x: dace.int64, gridDim_y: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_y, 0:gridDim_x]:
        shared = dace.ndarray([32,32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        for threadIdx_y, threadIdx_x in dace.map[0:32,0:32]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexs
            rowIdx = 32*blockIdx_y + threadIdx_y
            colIdx = 32*blockIdx_x + threadIdx_x
            delta = 32*gridDim_y
            for loopIdx in dace.map[0:loopNum]:
                # add the value
                if (rowIdx<H) and (colIdx<W):
                    value += inputs[rowIdx, colIdx]
                rowIdx += delta
            # write the value into shared memory
            shared[threadIdx_x,threadIdx_y] = value
        # synchronize here
        for threadIdx_y, threadIdx_x in dace.map[0:32,0:32]:
            # warp reduce
            reduced = warpReduce_sum(shared[threadIdx_y,threadIdx_x])
            # write back to global memory with atomic add
            colIdx = 32*blockIdx_x + threadIdx_y
            if (threadIdx_x==0) and (colIdx<W):
                outputs[colIdx] += reduced
    return outputs

@dace.program
def ThreadReduce(inputs: dace.float64[H, W], gridDim_x: dace.int64, blockDim_x: dace.int64, loopNum: dace.int64, rowNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_x in dace.map[0:gridDim_x]:
        # thread mapping
        for threadIdx_x in dace.map[0:blockDim_x]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexs
            Idx = blockDim_x * blockIdx_x + threadIdx_x
            rowIdx = dace.int64(Idx/W)
            colIdx = Idx%W
            # sum up the values
            if rowIdx<rowNum:
                delta = rowNum
                for loopIdx in dace.map[0:loopNum]:
                    if rowIdx<H:
                        value += inputs[rowIdx, colIdx]
                    rowIdx += delta
                # write back to global memory with atomic add
                outputs[colIdx] += value
    return outputs