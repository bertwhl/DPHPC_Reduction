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
def WarpReadWarpReduceCornerOld(inputs: dace.float64[H, W], gridDim_x: dace.int64, rowNum_warp: dace.int64, rowNum_block: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_x in dace.map[0:gridDim_x]:
        shared = dace.ndarray([1024], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        for threadIdx_z, threadIdx_y, threadIdx_x in dace.map[0:W,0:rowNum_warp,0:32]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexs
            threadIdx = rowNum_block*threadIdx_z + 32*threadIdx_y + threadIdx_x
            colIdx = threadIdx%W
            rowIdx_offset = dace.int64(threadIdx/W)
            rowIdx = rowNum_block*blockIdx_x + rowIdx_offset
            rowIdx_delta = rowNum_block*gridDim_x
            for loopIdx in dace.map[0:loopNum]:
                # add the value
                if rowIdx<H:
                    value += inputs[rowIdx, colIdx]
                rowIdx += rowIdx_delta
            # write the value into shared memory
            shared[W*rowIdx_offset + colIdx] = value
        # synchronize here
        for threadIdx_z, threadIdx_y, threadIdx_x in dace.map[0:W,0:rowNum_warp,0:32]:
            rowIdx = 32*threadIdx_y + threadIdx_x
            colIdx = threadIdx_z
            # warp reduce
            reduced = warpReduce_sum(shared[W*rowIdx + colIdx])
            # write back to global memory with atomic add
            if threadIdx_x==0:
                outputs[colIdx] += reduced
    return outputs

@dace.program
def WarpReadWarpReduceCorner(inputs: dace.float64[H, W], gridDim_x: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_x in dace.map[0:gridDim_x]:
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
                    value += inputs[rowIdx, colIdx]
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

@dace.program
def ThreadReduceAlign(inputs: dace.float64[H, W], gridDim_x: dace.int64, gridDim_y: dace.int64, blockDim_x: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_y, 0:gridDim_x]:
        # thread mapping
        for threadIdx_x in dace.map[0:blockDim_x]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexes
            rowIdx = blockIdx_y
            colIdx = blockDim_x * blockIdx_x + threadIdx_x
            # sum up the values
            if colIdx<W:
                delta = gridDim_y
                for loopIdx in dace.map[0:loopNum]:
                    if rowIdx<H:
                        value += inputs[rowIdx, colIdx]
                    rowIdx += delta
                # write back to global memory with atomic add
                outputs[colIdx] += value
    return outputs

@dace.program
def ThreadReduceAlign1(inputs: dace.float64[H, W], gridDim_x: dace.int64, blockDim_x: dace.int64, blockDim_y: dace.int64, loopNum: dace.int64):
    # define and initialize the output vector
    outputs = dace.ndarray([W], dtype=dace.float64)
    outputs[:] = 0
    # blocks mapping
    for blockIdx_x in dace.map[0:gridDim_x]:
        # thread mapping
        for threadIdx_y, threadIdx_x in dace.map[0:blockDim_y, 0:blockDim_x]:
            # initialize value
            value = dace.float64(0)
            # calculate the indexes
            rowIdx = blockIdx_x * blockDim_y + threadIdx_y
            colIdx = threadIdx_x
            # sum up the values
            delta = gridDim_x * blockDim_y
            for loopIdx in dace.map[0:loopNum]:
                if rowIdx<H:
                    value += inputs[rowIdx, colIdx]
                rowIdx += delta
            # write back to global memory with atomic add
            outputs[colIdx] += value
    return outputs

if __name__ == '__main__':
    # Transform to GPU, keep thread-block map
    sdfg = ThreadReduceAlign1.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    h = 1024
    w = 17
    BlockPerColumn = 32
    RowPerBlock = 16
    loopNum = (h+BlockPerColumn*RowPerBlock-1)//(BlockPerColumn*RowPerBlock)

    # Test
    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerColumn, blockDim_x=w, blockDim_y=RowPerBlock, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)