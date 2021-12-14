import dace
from util import *

H = dace.symbol('H')
W = dace.symbol('W')

# use aligned version of thread reduce to deal with ordinaray column reduce
@dace.program
def ColReduce(inputs: dace.float64[H, W], gridDim_x: dace.int64, gridDim_y: dace.int64, blockDim_x: dace.int64, loopNum: dace.int64):
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

# use warp read warp reduce for narrow corner cases
@dace.program
def ColReduceNarrow(inputs: dace.float64[H, W], gridDim_x: dace.int64, loopNum: dace.int64):
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

# multiple warps per row, multiple rows per block
@dace.program
def RowReduceMulti(inputs: dace.float64[H, W], gridDim_x: dace.int64, loopNum: dace.int64, blockDim_z: dace.int64, blockDim_y: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for block_id in dace.map[0:gridDim_x]:
        _blockDim_z = dace.int32(blockDim_z)
        _blockDim_y = dace.int32(blockDim_y)
        for warp_id_x, warp_id_y, thread_id in dace.map[0:blockDim_z, 0:blockDim_y, 0:32]:
            value = dace.float64(0)
            row_id = block_id * _blockDim_z + warp_id_x
            col_id = warp_id_y * 32 + thread_id
            delta = _blockDim_y * 32
            for i in dace.map[0:loopNum]:
                if row_id < H and col_id < W:
                    value += inputs[row_id, col_id]
                col_id += delta
            reduced = warpReduce_sum(value)
            if thread_id==0:
                outputs[row_id] += reduced
    return outputs

# no-loop version
@dace.program
def RowReduceNoLoop(inputs: dace.float64[H, W], blockDim_y: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for block_id in dace.map[0:H]:
        for warp_id, thread_id in dace.map[0:blockDim_y, 0:32]:
            col_id = warp_id*32+thread_id
            if col_id < W:
                reduced = warpReduce_sum(inputs[block_id, col_id])
            if thread_id == 0:
                outputs[block_id] += reduced
    return outputs

# atomic add to global memory
@dace.program
def RowReduceGlobal(inputs: dace.float64[H, W], gridDim_y: dace.int64, loopNum: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_y, 0:H]:
        for warp_id, thread_id in dace.map[0:32,0:32]:
            row_id = blockIdx_x
            col_id = 1024*blockIdx_y + 32*warp_id +thread_id
            delta = 1024*gridDim_y
            value = dace.float64(0)
            for loopIdx in dace.map[0:loopNum]:
                if col_id<W:
                    value += inputs[row_id, col_id]
                col_id += delta
            reduced = warpReduce_sum(value)
            if thread_id == 0:
                outputs[row_id] += reduced
    return outputs

# atomic add to shared memory first, then global memory
@dace.program
def RowReduceShared(inputs: dace.float64[H, W], gridDim_y: dace.int64, loopNum: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_y, blockIdx_x in dace.map[0:gridDim_y, 0:H]:
        shared = dace.ndarray([32], dtype=dace.float64, storage=dace.StorageType.GPU_Shared)
        row_id = blockIdx_x
        for warp_id, thread_id in dace.map[0:32,0:32]:
            col_id = 1024*blockIdx_y + 32*warp_id +thread_id
            delta = 1024*gridDim_y
            value = dace.float64(0)
            for loopIdx in dace.map[0:loopNum]:
                if col_id<W:
                    value += inputs[row_id, col_id]
                col_id += delta
            reduced = warpReduce_sum(value)
            if thread_id == 0:
                shared[warp_id] = reduced
        for warp_id, thread_id in dace.map[0:32,0:32]:
            if warp_id == 0:
                value = shared[thread_id]
                reduced = warpReduce_sum(value)
                if thread_id == 0:
                    outputs[row_id] += reduced
    return outputs

# narrow corner case for row reduce
@dace.program
def RowReduceNarrow(inputs: dace.float64[H, W], gridDim_x: dace.int64, blockDim_x: dace.int64):
    outputs = dace.ndarray([H], dtype=dace.float64)
    outputs[:] = 0
    for blockIdx_x in dace.map[0:gridDim_x]:
        for threadIdx_x in dace.map[0:blockDim_x]:
            value = dace.float64(0)
            rowIdx = blockIdx_x*blockDim_x+threadIdx_x
            if rowIdx < H:
                for colIdx in dace.map[0:W]:
                    value += inputs[rowIdx,colIdx]
                outputs[rowIdx] = value
    return outputs

if __name__ == '__main__':
    import numpy as np
    from dace.transformation.interstate import GPUTransformSDFG
    # Transform to GPU, keep thread-block map
    sdfg = ColReduce.to_sdfg()
    sdfg.apply_transformations(GPUTransformSDFG, {'sequential_innermaps': False})

    h = 1024
    w = 1024
    BlockPerRow = 4
    BlockPerColumn = 16
    ThreadPerBlock = 256
    # other params
    loopNum = (h+BlockPerColumn-1)//BlockPerColumn

    # Test
    inputs = np.random.rand(h, w)
    outputs = sdfg(H=h, W=w, inputs=inputs, gridDim_x=BlockPerRow, gridDim_y=BlockPerColumn, blockDim_x=ThreadPerBlock, loopNum=loopNum)
    compared = np.sum(inputs, axis=0)
    assert np.allclose(outputs, compared)